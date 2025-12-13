"""
Task-specific prediction heads for different video understanding tasks.
"""

import torch
import torch.nn as nn


class ActionDetectionHead(nn.Module):
    """
    Action detection head for temporal action localization.
    
    Outputs action class probabilities and temporal boundaries for each frame.
    """
    
    def __init__(self, input_dim: int = 768, num_classes: int = 200, 
                 hidden_dim: int = 512):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Classification branch
        self.classify = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Temporal localization branch
        self.localize = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Start and end logits
        )
        
    def forward(self, x):
        """
        Args:
            x: Feature tensor (batch, seq_len, input_dim)
            
        Returns:
            action_logits: (batch, seq_len, num_classes)
            boundary_logits: (batch, seq_len, 2)
        """
        action_logits = self.classify(x)
        boundary_logits = self.localize(x)
        
        return {
            "action_logits": action_logits,
            "boundary_logits": boundary_logits,
        }


class VideoQAHead(nn.Module):
    """
    Video Question Answering head.
    
    Fuses question embeddings with video features for answer prediction.
    """
    
    def __init__(self, video_dim: int = 768, text_dim: int = 768, 
                 hidden_dim: int = 512, num_answers: int = 1000):
        super().__init__()
        
        self.video_dim = video_dim
        self.text_dim = text_dim
        
        # Cross-modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(video_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Answer prediction
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_answers)
        )
        
    def forward(self, video_features, question_embedding):
        """
        Args:
            video_features: (batch, seq_len, video_dim) or (batch, video_dim)
            question_embedding: (batch, text_dim)
            
        Returns:
            answer_logits: (batch, num_answers)
        """
        # Average pool video features if sequence length is present
        if video_features.dim() == 3:
            video_features = video_features.mean(dim=1)
        
        # Concatenate video and question features
        fused = torch.cat([video_features, question_embedding], dim=-1)
        
        # Fuse modalities
        fused = self.fusion(fused)
        
        # Predict answer
        answer_logits = self.answer_head(fused)
        
        return answer_logits


class VideoCaptioningHead(nn.Module):
    """
    Video Captioning head for generating video descriptions.
    """
    
    def __init__(self, input_dim: int = 768, vocab_size: int = 10000, 
                 max_len: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # LSTM decoder
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(input_dim, vocab_size)
        
    def forward(self, video_features, teacher_forcing=None):
        """
        Args:
            video_features: (batch, seq_len, input_dim)
            teacher_forcing: Optional ground truth tokens for training
            
        Returns:
            logits: (batch, max_len, vocab_size)
        """
        batch_size = video_features.size(0)
        
        # Decode video features to sequence of tokens
        output, _ = self.decoder(video_features)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        return logits
