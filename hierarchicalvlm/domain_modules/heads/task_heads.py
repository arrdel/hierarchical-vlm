"""
Task-Specific Heads for Different Video Understanding Tasks

This module implements prediction heads for three core video understanding tasks:
- Action Detection: Temporal localization of actions in videos
- Visual Question Answering (VQA): Answering questions about video content
- Video Captioning: Generating descriptions for videos

Each head is designed to work with the shared backbone from HierarchicalVLM
and can be fine-tuned with LoRA adapters for domain-specific optimization.
"""

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionDetectionHead(nn.Module):
    """Head for temporal action detection in videos.
    
    Predicts action class and temporal location for each frame/segment.
    Outputs both class predictions and confidence scores.
    
    Args:
        input_dim: Dimension of input features
        num_classes: Number of action classes
        num_frames: Number of video frames/segments
        dropout: Dropout probability
        use_temporal_conv: Use temporal convolution for context
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 150,
        num_frames: int = 32,
        dropout: float = 0.1,
        use_temporal_conv: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_frames = num_frames
        
        # Temporal context layer
        if use_temporal_conv:
            # 1D temporal convolution for capturing temporal context
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(input_dim),
                nn.ReLU()
            )
        else:
            self.temporal_conv = nn.Identity()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Feature refinement
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.bn1 = nn.BatchNorm1d(input_dim // 2)
        self.relu = nn.ReLU()
        
        # Classification head
        self.class_head = nn.Linear(input_dim // 2, num_classes)
        
        # Confidence/IoU prediction head
        self.conf_head = nn.Linear(input_dim // 2, 1)
    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Predict action detections.
        
        Args:
            x: Input features of shape (batch, num_frames, input_dim)
            masks: Attention masks of shape (batch, num_frames)
        
        Returns:
            Dictionary with:
            - 'class_logits': (batch, num_frames, num_classes)
            - 'confidence': (batch, num_frames, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Transpose for temporal convolution: (batch, dim, seq_len)
        x_t = x.transpose(1, 2)
        
        # Apply temporal convolution
        x_t = self.temporal_conv(x_t)
        
        # Transpose back: (batch, seq_len, dim)
        x = x_t.transpose(1, 2)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Feature refinement
        x = self.fc1(x)
        x = self.bn1(x.view(-1, x.shape[-1])).view(batch_size, seq_len, -1)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Prediction heads
        class_logits = self.class_head(x)  # (batch, seq_len, num_classes)
        confidence = self.conf_head(x)     # (batch, seq_len, 1)
        
        # Apply mask if provided
        if masks is not None:
            # Expand mask: (batch, seq_len) -> (batch, seq_len, 1)
            mask_expanded = masks.unsqueeze(-1)
            confidence = confidence.masked_fill(~mask_expanded.bool(), float('-inf'))
        
        return {
            'class_logits': class_logits,
            'confidence': confidence.squeeze(-1)
        }


class VideoQAHead(nn.Module):
    """Head for Video Question Answering.
    
    Combines video features with question embeddings to predict answer.
    Supports multiple answer types: classification, counting, descriptive.
    
    Args:
        input_dim: Dimension of video features
        question_dim: Dimension of question embeddings
        num_answers: Number of possible answers (for multi-choice QA)
        dropout: Dropout probability
        use_attention: Use attention to focus on relevant frames
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        question_dim: int = 768,
        num_answers: int = 1000,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.question_dim = question_dim
        self.num_answers = num_answers
        
        # Frame-question attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.attention = None
        
        # Fusion layer for combining video and question
        self.fusion_fc = nn.Linear(input_dim + question_dim, input_dim)
        self.fusion_bn = nn.BatchNorm1d(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Reasoning layers
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.bn2 = nn.BatchNorm1d(input_dim // 2)
        
        # Answer prediction
        self.answer_head = nn.Linear(input_dim // 2, num_answers)
        
        # Confidence estimation
        self.confidence_head = nn.Linear(input_dim // 2, 1)
    
    def forward(
        self,
        video_features: torch.Tensor,
        question_embedding: torch.Tensor,
        video_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Predict answer to video question.
        
        Args:
            video_features: Video features of shape (batch, num_frames, input_dim)
            question_embedding: Question embedding of shape (batch, question_dim)
            video_masks: Attention masks for video
        
        Returns:
            Dictionary with:
            - 'answer_logits': (batch, num_answers)
            - 'confidence': (batch,)
        """
        batch_size, num_frames, _ = video_features.shape
        
        # Apply attention to focus on relevant frames
        if self.attention is not None:
            # Use question as query
            q_expanded = question_embedding.unsqueeze(1)  # (batch, 1, question_dim)
            
            # Attend over video frames
            attended, _ = self.attention(
                q_expanded,
                video_features,
                video_features,
                key_padding_mask=None
            )  # (batch, 1, input_dim)
            attended = attended.squeeze(1)
        else:
            # Simple global average pooling
            attended = video_features.mean(dim=1)  # (batch, input_dim)
        
        # Fuse video and question information
        # Expand question to match batch
        fused = torch.cat([attended, question_embedding], dim=-1)  # (batch, input_dim + question_dim)
        
        fused = self.fusion_fc(fused)
        fused = self.fusion_bn(fused)
        fused = self.dropout(fused)
        
        # Reasoning
        x = self.fc1(fused)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Predict answer
        answer_logits = self.answer_head(x)  # (batch, num_answers)
        confidence = self.confidence_head(x)  # (batch, 1)
        
        return {
            'answer_logits': answer_logits,
            'confidence': confidence.squeeze(-1)
        }


class VideoCaptioningHead(nn.Module):
    """Head for Video Captioning (Dense Video Captioning).
    
    Predicts caption for each temporal segment and generates temporal boundaries.
    Can work with a language decoder or generate embeddings for decoder.
    
    Args:
        input_dim: Dimension of video features
        vocab_size: Size of vocabulary
        caption_dim: Dimension of caption embeddings
        max_caption_len: Maximum caption length
        dropout: Dropout probability
        use_decoder: Include decoder or just embeddings
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        vocab_size: int = 10000,
        caption_dim: int = 512,
        max_caption_len: int = 20,
        dropout: float = 0.1,
        use_decoder: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.caption_dim = caption_dim
        self.max_caption_len = max_caption_len
        
        # Feature projection to caption dimension
        self.feature_proj = nn.Linear(input_dim, caption_dim)
        self.proj_bn = nn.BatchNorm1d(caption_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Segment boundary prediction (for dense captioning)
        self.boundary_fc = nn.Linear(caption_dim, 2)  # Start and end probabilities
        
        # Caption generation
        if use_decoder:
            # LSTM decoder for caption generation
            self.embedding = nn.Embedding(vocab_size, caption_dim)
            self.decoder = nn.LSTM(
                input_size=caption_dim,
                hidden_size=caption_dim,
                num_layers=2,
                dropout=dropout,
                batch_first=True
            )
            self.decoder_fc = nn.Linear(caption_dim, vocab_size)
        else:
            self.embedding = None
            self.decoder = None
            self.decoder_fc = None
        
        # Confidence estimation
        self.confidence_head = nn.Linear(caption_dim, 1)
    
    def forward(
        self,
        video_features: torch.Tensor,
        caption_tokens: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Generate video captions.
        
        Args:
            video_features: Video features of shape (batch, num_frames, input_dim)
            caption_tokens: Ground truth caption tokens for training
            teacher_forcing: Use teacher forcing during training
        
        Returns:
            Dictionary with:
            - 'boundary_logits': (batch, num_frames, 2) - start/end predictions
            - 'caption_logits': (batch, max_len, vocab_size) if decoder enabled
            - 'confidence': (batch,)
        """
        batch_size, num_frames, _ = video_features.shape
        
        # Project features to caption dimension
        x = self.feature_proj(video_features)
        x = self.proj_bn(x.view(-1, x.shape[-1])).view(batch_size, num_frames, -1)
        x = self.dropout(x)
        
        # Predict segment boundaries
        boundary_logits = self.boundary_fc(x)  # (batch, num_frames, 2)
        
        # Generate captions if decoder is available
        caption_logits = None
        if self.decoder is not None and caption_tokens is not None:
            # Get segment features (use average for now)
            segment_features = x.mean(dim=1)  # (batch, caption_dim)
            
            if teacher_forcing and caption_tokens is not None:
                # Teacher forcing: use ground truth tokens
                embedded = self.embedding(caption_tokens)  # (batch, max_len, caption_dim)
                decoder_out, _ = self.decoder(embedded)
                caption_logits = self.decoder_fc(decoder_out)  # (batch, max_len, vocab_size)
            else:
                # Autoregressive generation
                caption_logits = self._generate_captions(
                    segment_features, 
                    self.max_caption_len
                )
        
        # Confidence estimation
        segment_features = x.mean(dim=1)  # (batch, caption_dim)
        confidence = self.confidence_head(segment_features)  # (batch, 1)
        
        return {
            'boundary_logits': boundary_logits,
            'caption_logits': caption_logits,
            'confidence': confidence.squeeze(-1)
        }
    
    def _generate_captions(
        self,
        features: torch.Tensor,
        max_len: int
    ) -> torch.Tensor:
        """Generate captions autoregressively.
        
        Args:
            features: Segment features of shape (batch, caption_dim)
            max_len: Maximum caption length
        
        Returns:
            Caption logits of shape (batch, max_len, vocab_size)
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Start token (assuming token 0 is <START>)
        prev_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        caption_logits = []
        hidden = None
        
        for _ in range(max_len):
            # Embed token
            embedded = self.embedding(prev_token)  # (batch, 1, caption_dim)
            
            # Decoder step
            decoder_out, hidden = self.decoder(embedded, hidden)
            logits = self.decoder_fc(decoder_out)  # (batch, 1, vocab_size)
            caption_logits.append(logits)
            
            # Sample next token
            prev_token = logits.argmax(dim=-1)  # (batch, 1)
        
        # Concatenate all timesteps
        caption_logits = torch.cat(caption_logits, dim=1)  # (batch, max_len, vocab_size)
        
        return caption_logits


class MultiTaskHead(nn.Module):
    """Combined head for multi-task learning.
    
    Shares a backbone but has separate task-specific heads for:
    - Action Detection
    - Video QA
    - Video Captioning
    
    Args:
        input_dim: Dimension of input features
        num_action_classes: Number of action classes
        num_qa_answers: Number of possible QA answers
        num_frames: Number of video frames
        vocab_size: Size of vocabulary for captioning
        task_weights: Loss weights for each task
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_action_classes: int = 150,
        num_qa_answers: int = 1000,
        num_frames: int = 32,
        vocab_size: int = 10000,
        task_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Task-specific heads
        self.action_head = ActionDetectionHead(
            input_dim=input_dim,
            num_classes=num_action_classes,
            num_frames=num_frames
        )
        
        self.qa_head = VideoQAHead(
            input_dim=input_dim,
            question_dim=input_dim,
            num_answers=num_qa_answers
        )
        
        self.captioning_head = VideoCaptioningHead(
            input_dim=input_dim,
            vocab_size=vocab_size,
            caption_dim=input_dim
        )
        
        # Task weights
        self.task_weights = task_weights or {
            'action': 1.0,
            'qa': 1.0,
            'captioning': 1.0
        }
    
    def forward(
        self,
        video_features: torch.Tensor,
        question_embedding: Optional[torch.Tensor] = None,
        caption_tokens: Optional[torch.Tensor] = None,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Predict outputs for specified tasks.
        
        Args:
            video_features: Video features of shape (batch, num_frames, input_dim)
            question_embedding: Question embedding for QA task
            caption_tokens: Caption tokens for captioning task
            tasks: List of tasks to run ['action', 'qa', 'captioning']
        
        Returns:
            Dictionary mapping task names to their outputs
        """
        results = {}
        
        if tasks is None or 'action' in tasks:
            results['action'] = self.action_head(video_features)
        
        if (tasks is None or 'qa' in tasks) and question_embedding is not None:
            results['qa'] = self.qa_head(video_features, question_embedding)
        
        if tasks is None or 'captioning' in tasks:
            results['captioning'] = self.captioning_head(
                video_features,
                caption_tokens=caption_tokens
            )
        
        return results
