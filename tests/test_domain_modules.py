"""
Comprehensive Test Suite for Domain Modules

Tests cover:
- LoRA adapters (LinearLoRA, AttentionLoRA, LoRAAdapter)
- Task-specific heads (ActionDetection, VideoQA, VideoCaptioning)
- Domain experts (DomainExpert, MultiDomainAdapter, DomainRouter)
- Integration tests for multi-task learning

Run with: pytest tests/test_domain_modules.py -v
"""

import pytest
import torch
import torch.nn as nn

# Import domain modules
from hierarchicalvlm.domain_modules.adapters.lora import (
    LinearLoRA, AttentionLoRA, LoRAAdapter, LoRALayerWrapper
)
from hierarchicalvlm.domain_modules.heads.task_heads import (
    ActionDetectionHead, VideoQAHead, VideoCaptioningHead, MultiTaskHead
)
from hierarchicalvlm.domain_modules.domain_experts.domain_expert import (
    DomainExpert, MultiDomainAdapter, DomainRouter, SpecializedTransformer
)


class TestLinearLoRA:
    """Tests for LinearLoRA adapter."""
    
    def test_linear_lora_initialization(self):
        """Test LinearLoRA initialization with correct shapes."""
        lora = LinearLoRA(in_features=768, out_features=768, rank=8, alpha=16.0)
        
        assert lora.lora_a.in_features == 768
        assert lora.lora_a.out_features == 8
        assert lora.lora_b.in_features == 8
        assert lora.lora_b.out_features == 768
        assert lora.scaling == 16.0 / 8
    
    def test_linear_lora_forward(self):
        """Test LinearLoRA forward pass."""
        batch_size, seq_len, dim = 2, 4, 768
        lora = LinearLoRA(in_features=dim, out_features=dim, rank=8)
        
        x = torch.randn(batch_size, seq_len, dim)
        output = lora(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_linear_lora_gradient_flow(self):
        """Test gradient flow through LinearLoRA."""
        lora = LinearLoRA(in_features=768, out_features=768, rank=8)
        
        x = torch.randn(2, 4, 768, requires_grad=True)
        output = lora(x)
        loss = output.sum()
        loss.backward()
        
        assert lora.lora_a.weight.grad is not None
        assert lora.lora_b.weight.grad is not None
        assert x.grad is not None
    
    def test_linear_lora_different_dimensions(self):
        """Test LinearLoRA with different input/output dimensions."""
        lora = LinearLoRA(in_features=512, out_features=1024, rank=16)
        
        x = torch.randn(4, 512)
        output = lora(x)
        
        assert output.shape == (4, 1024)
    
    def test_linear_lora_dropout(self):
        """Test LinearLoRA with dropout."""
        lora = LinearLoRA(in_features=768, out_features=768, dropout=0.5)
        lora.train()
        
        x = torch.randn(2, 4, 768)
        output1 = lora(x)
        output2 = lora(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2)


class TestAttentionLoRA:
    """Tests for AttentionLoRA adapter."""
    
    def test_attention_lora_initialization(self):
        """Test AttentionLoRA initialization."""
        lora = AttentionLoRA(hidden_dim=768, num_heads=8, rank=8)
        
        assert hasattr(lora, 'lora_q')
        assert hasattr(lora, 'lora_k')
        assert hasattr(lora, 'lora_v')
    
    def test_attention_lora_forward(self):
        """Test AttentionLoRA forward pass."""
        batch_size, seq_len, dim = 2, 4, 768
        lora = AttentionLoRA(hidden_dim=dim, num_heads=8)
        
        q = torch.randn(batch_size, seq_len, dim)
        k = torch.randn(batch_size, seq_len, dim)
        v = torch.randn(batch_size, seq_len, dim)
        
        q_out, k_out, v_out = lora(q, k, v)
        
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape
    
    def test_attention_lora_selective_adaptation(self):
        """Test selective adaptation of Q/K/V."""
        lora = AttentionLoRA(
            hidden_dim=768,
            adapt_q=True,
            adapt_k=False,
            adapt_v=True
        )
        
        q = torch.randn(2, 4, 768)
        k = torch.randn(2, 4, 768)
        v = torch.randn(2, 4, 768)
        
        q_out, k_out, v_out = lora(q, k, v)
        
        # K should not be adapted (should equal input)
        assert torch.allclose(k_out, k)


class TestLoRAAdapter:
    """Tests for complete LoRAAdapter."""
    
    def test_lora_adapter_initialization(self):
        """Test LoRAAdapter initialization."""
        adapter = LoRAAdapter(dim=768, rank=8, expansion_factor=4.0)
        
        assert hasattr(adapter, 'down')
        assert hasattr(adapter, 'up')
        assert hasattr(adapter, 'act')
    
    def test_lora_adapter_forward(self):
        """Test LoRAAdapter forward pass."""
        batch_size, seq_len, dim = 2, 4, 768
        adapter = LoRAAdapter(dim=dim, rank=8)
        
        x = torch.randn(batch_size, seq_len, dim)
        output = adapter(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_lora_adapter_residual(self):
        """Test that LoRAAdapter applies residual connection."""
        adapter = LoRAAdapter(dim=768, rank=8)
        adapter.down.lora_a.weight.data.zero_()
        adapter.down.lora_b.weight.data.zero_()
        adapter.up.lora_a.weight.data.zero_()
        adapter.up.lora_b.weight.data.zero_()
        
        x = torch.randn(2, 4, 768)
        output = adapter(x)
        
        # With zero LoRA weights, output should equal input (residual)
        assert torch.allclose(output, x, atol=1e-5)
    
    def test_lora_adapter_get_params(self):
        """Test getting LoRA parameters for training."""
        adapter = LoRAAdapter(dim=768, rank=8)
        params = adapter.get_lora_params()
        
        # Should have 4 parameters (A and B for down and up)
        assert len(params) >= 4


class TestActionDetectionHead:
    """Tests for ActionDetectionHead."""
    
    def test_action_detection_initialization(self):
        """Test ActionDetectionHead initialization."""
        head = ActionDetectionHead(
            input_dim=768,
            num_classes=150,
            num_frames=32
        )
        
        assert hasattr(head, 'temporal_conv')
        assert hasattr(head, 'class_head')
        assert hasattr(head, 'conf_head')
    
    def test_action_detection_forward(self):
        """Test ActionDetectionHead forward pass."""
        batch_size, num_frames, dim = 2, 32, 768
        head = ActionDetectionHead(
            input_dim=dim,
            num_classes=150,
            num_frames=num_frames
        )
        
        x = torch.randn(batch_size, num_frames, dim)
        output = head(x)
        
        assert 'class_logits' in output
        assert 'confidence' in output
        assert output['class_logits'].shape == (batch_size, num_frames, 150)
        assert output['confidence'].shape == (batch_size, num_frames)
    
    def test_action_detection_with_mask(self):
        """Test ActionDetectionHead with attention masks."""
        head = ActionDetectionHead(input_dim=768, num_classes=150, num_frames=32)
        
        x = torch.randn(2, 32, 768)
        masks = torch.ones(2, 32, dtype=torch.bool)
        masks[:, 20:] = False  # Mask out last 12 frames
        
        output = head(x, masks=masks)
        
        assert output['class_logits'].shape == (2, 32, 150)
    
    def test_action_detection_gradient_flow(self):
        """Test gradient flow through ActionDetectionHead."""
        head = ActionDetectionHead(input_dim=768, num_classes=150)
        
        x = torch.randn(2, 32, 768, requires_grad=True)
        output = head(x)
        loss = output['class_logits'].sum() + output['confidence'].sum()
        loss.backward()
        
        assert x.grad is not None


class TestVideoQAHead:
    """Tests for VideoQAHead."""
    
    def test_qa_head_initialization(self):
        """Test VideoQAHead initialization."""
        head = VideoQAHead(
            input_dim=768,
            question_dim=768,
            num_answers=1000
        )
        
        assert hasattr(head, 'attention')
        assert hasattr(head, 'answer_head')
    
    def test_qa_head_forward(self):
        """Test VideoQAHead forward pass."""
        batch_size, num_frames, dim = 2, 32, 768
        head = VideoQAHead(input_dim=dim, question_dim=dim, num_answers=1000)
        
        video_features = torch.randn(batch_size, num_frames, dim)
        question = torch.randn(batch_size, dim)
        
        output = head(video_features, question)
        
        assert 'answer_logits' in output
        assert 'confidence' in output
        assert output['answer_logits'].shape == (batch_size, 1000)
        assert output['confidence'].shape == (batch_size,)
    
    def test_qa_head_without_attention(self):
        """Test VideoQAHead without attention mechanism."""
        head = VideoQAHead(
            input_dim=768,
            question_dim=768,
            use_attention=False
        )
        
        video_features = torch.randn(2, 32, 768)
        question = torch.randn(2, 768)
        
        output = head(video_features, question)
        
        assert output['answer_logits'].shape[0] == 2


class TestVideoCaptioningHead:
    """Tests for VideoCaptioningHead."""
    
    def test_captioning_head_initialization(self):
        """Test VideoCaptioningHead initialization."""
        head = VideoCaptioningHead(
            input_dim=768,
            vocab_size=10000,
            caption_dim=512
        )
        
        assert hasattr(head, 'feature_proj')
        assert hasattr(head, 'boundary_fc')
        assert hasattr(head, 'decoder')
    
    def test_captioning_head_forward(self):
        """Test VideoCaptioningHead forward pass."""
        batch_size, num_frames, dim = 2, 32, 768
        head = VideoCaptioningHead(
            input_dim=dim,
            vocab_size=10000,
            caption_dim=512
        )
        
        video_features = torch.randn(batch_size, num_frames, dim)
        caption_tokens = torch.randint(0, 10000, (batch_size, 20))
        
        output = head(video_features, caption_tokens=caption_tokens)
        
        assert 'boundary_logits' in output
        assert 'caption_logits' in output
        assert output['boundary_logits'].shape == (batch_size, num_frames, 2)
    
    def test_captioning_head_boundary_prediction(self):
        """Test boundary prediction for dense captioning."""
        head = VideoCaptioningHead(input_dim=768, vocab_size=10000)
        
        video_features = torch.randn(2, 32, 768)
        output = head(video_features)
        
        # Boundary logits should sum to 1 after softmax
        boundary_probs = torch.softmax(output['boundary_logits'], dim=-1)
        assert boundary_probs.shape == (2, 32, 2)


class TestMultiTaskHead:
    """Tests for MultiTaskHead."""
    
    def test_multi_task_head_initialization(self):
        """Test MultiTaskHead initialization."""
        head = MultiTaskHead(
            input_dim=768,
            num_action_classes=150,
            num_qa_answers=1000,
            vocab_size=10000
        )
        
        assert hasattr(head, 'action_head')
        assert hasattr(head, 'qa_head')
        assert hasattr(head, 'captioning_head')
    
    def test_multi_task_head_all_tasks(self):
        """Test MultiTaskHead with all tasks."""
        batch_size, num_frames, dim = 2, 32, 768
        head = MultiTaskHead(input_dim=dim)
        
        video_features = torch.randn(batch_size, num_frames, dim)
        question = torch.randn(batch_size, dim)
        caption_tokens = torch.randint(0, 10000, (batch_size, 20))
        
        output = head(
            video_features,
            question_embedding=question,
            caption_tokens=caption_tokens
        )
        
        assert 'action' in output
        assert 'qa' in output
        assert 'captioning' in output
    
    def test_multi_task_head_selective_tasks(self):
        """Test MultiTaskHead with selected tasks only."""
        head = MultiTaskHead(input_dim=768)
        
        video_features = torch.randn(2, 32, 768)
        output = head(video_features, tasks=['action'])
        
        assert 'action' in output
        assert 'qa' not in output


class TestDomainExpert:
    """Tests for DomainExpert."""
    
    def test_domain_expert_initialization(self):
        """Test DomainExpert initialization."""
        expert = DomainExpert(
            input_dim=768,
            expert_dim=1024,
            domain_name='sports'
        )
        
        assert expert.domain_name == 'sports'
        assert hasattr(expert, 'expert_layers')
    
    def test_domain_expert_forward_2d(self):
        """Test DomainExpert forward pass with 2D input."""
        expert = DomainExpert(input_dim=768, expert_dim=1024)
        
        x = torch.randn(2, 768)
        output = expert(x)
        
        assert output.shape == (2, 768)
    
    def test_domain_expert_forward_3d(self):
        """Test DomainExpert forward pass with 3D input."""
        batch_size, seq_len, dim = 2, 32, 768
        expert = DomainExpert(input_dim=dim, expert_dim=1024)
        
        x = torch.randn(batch_size, seq_len, dim)
        output = expert(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_domain_expert_specialization_gate(self):
        """Test specialization gating mechanism."""
        expert = DomainExpert(
            input_dim=768,
            specialization_factor=0.9
        )
        
        x = torch.randn(2, 768)
        output = expert(x, apply_gate=True)
        
        assert output.shape == x.shape


class TestMultiDomainAdapter:
    """Tests for MultiDomainAdapter."""
    
    def test_multi_domain_adapter_initialization(self):
        """Test MultiDomainAdapter initialization."""
        adapter = MultiDomainAdapter(
            input_dim=768,
            domains=['sports', 'tutorials', 'news']
        )
        
        assert len(adapter.experts) == 3
        assert all(d in adapter.experts for d in ['sports', 'tutorials', 'news'])
    
    def test_multi_domain_adapter_forward(self):
        """Test MultiDomainAdapter forward pass."""
        batch_size, seq_len, dim = 2, 32, 768
        adapter = MultiDomainAdapter(input_dim=dim)
        
        x = torch.randn(batch_size, seq_len, dim)
        output, routing_info = adapter(x, return_routing=False)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_multi_domain_adapter_routing(self):
        """Test routing information from MultiDomainAdapter."""
        adapter = MultiDomainAdapter(input_dim=768)
        
        x = torch.randn(2, 32, 768)
        output, routing_info = adapter(x, return_routing=True)
        
        assert routing_info is not None
        assert all(d in routing_info for d in ['sports', 'tutorials', 'news', 'general'])
    
    def test_multi_domain_adapter_gradient_flow(self):
        """Test gradient flow through MultiDomainAdapter."""
        adapter = MultiDomainAdapter(input_dim=768)
        
        x = torch.randn(2, 32, 768, requires_grad=True)
        output, _ = adapter(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestDomainRouter:
    """Tests for DomainRouter."""
    
    def test_domain_router_initialization(self):
        """Test DomainRouter initialization."""
        router = DomainRouter(
            input_dim=768,
            domains=['sports', 'tutorials', 'news']
        )
        
        assert router.num_domains == 3
    
    def test_domain_router_forward(self):
        """Test DomainRouter forward pass."""
        router = DomainRouter(input_dim=768)
        
        x = torch.randn(2, 32, 768)
        routing_info = router(x)
        
        assert 'hard_routing' in routing_info
        assert 'soft_routing' in routing_info
        assert 'domain_logits' in routing_info
        
        # Check shapes
        assert routing_info['hard_routing'].shape == (2, 4)
        assert routing_info['soft_routing'].shape == (2, 4)
    
    def test_domain_router_with_confidence(self):
        """Test DomainRouter with confidence estimation."""
        router = DomainRouter(input_dim=768)
        
        x = torch.randn(2, 32, 768)
        routing_info = router(x, return_confidence=True)
        
        assert 'confidence' in routing_info
        assert routing_info['confidence'].shape == (2, 1)
    
    def test_domain_router_soft_routing_sums_to_one(self):
        """Test that soft routing weights sum to 1."""
        router = DomainRouter(input_dim=768)
        
        x = torch.randn(2, 32, 768)
        routing_info = router(x)
        
        soft_routing = routing_info['soft_routing']
        routing_sum = soft_routing.sum(dim=-1)
        
        # All rows should sum to 1
        assert torch.allclose(routing_sum, torch.ones(2), atol=1e-5)


class TestSpecializedTransformer:
    """Tests for SpecializedTransformer."""
    
    def test_specialized_transformer_initialization(self):
        """Test SpecializedTransformer initialization."""
        transformer = SpecializedTransformer(
            input_dim=768,
            num_heads=8,
            domains=['sports', 'tutorials']
        )
        
        assert hasattr(transformer, 'attention')
        assert hasattr(transformer, 'domain_adapter')
    
    def test_specialized_transformer_forward(self):
        """Test SpecializedTransformer forward pass."""
        batch_size, seq_len, dim = 2, 32, 768
        transformer = SpecializedTransformer(input_dim=dim)
        
        x = torch.randn(batch_size, seq_len, dim)
        output = transformer(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_specialized_transformer_gradient_flow(self):
        """Test gradient flow through SpecializedTransformer."""
        transformer = SpecializedTransformer(input_dim=768)
        
        x = torch.randn(2, 32, 768, requires_grad=True)
        output = transformer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_lora_with_action_head(self):
        """Test using LoRA with ActionDetectionHead."""
        adapter = LoRAAdapter(dim=768)
        head = ActionDetectionHead(input_dim=768, num_classes=150)
        
        # Feature extraction
        x = torch.randn(2, 32, 768)
        
        # Apply LoRA adaptation
        x_adapted = adapter(x)
        
        # Action detection
        output = head(x_adapted)
        
        assert output['class_logits'].shape == (2, 32, 150)
    
    def test_domain_adapter_with_multi_task(self):
        """Test using MultiDomainAdapter with MultiTaskHead."""
        domain_adapter = MultiDomainAdapter(input_dim=768)
        task_head = MultiTaskHead(input_dim=768)
        
        # Feature extraction
        x = torch.randn(2, 32, 768)
        question = torch.randn(2, 768)
        
        # Domain adaptation
        x_domain_adapted, _ = domain_adapter(x)
        
        # Multi-task prediction
        output = task_head(x_domain_adapted, question_embedding=question)
        
        assert 'action' in output
        assert 'qa' in output
    
    def test_full_pipeline_with_router(self):
        """Test full pipeline with DomainRouter, Adapter, and Heads."""
        router = DomainRouter(input_dim=768)
        adapter = MultiDomainAdapter(input_dim=768)
        head = MultiTaskHead(input_dim=768)
        
        # Input
        video_features = torch.randn(2, 32, 768)
        question = torch.randn(2, 768)
        
        # Routing
        routing_info = router(video_features)
        
        # Domain adaptation
        adapted_features, _ = adapter(video_features)
        
        # Task prediction
        output = head(adapted_features, question_embedding=question)
        
        assert 'action' in output
        assert routing_info['hard_routing'].shape == (2, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
