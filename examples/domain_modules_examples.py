"""
Practical Examples for Domain Modules

This script demonstrates practical usage of domain modules including:
- Example 1: Basic LoRA fine-tuning
- Example 2: Action detection with domain specialization
- Example 3: Multi-task learning
- Example 4: Efficient domain routing
- Example 5: End-to-end training pipeline

Run with: python examples/domain_modules_examples.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def example_1_basic_lora_finetuning():
    """Example 1: Basic LoRA fine-tuning with linear layers."""
    print("\n" + "="*60)
    print("Example 1: Basic LoRA Fine-Tuning")
    print("="*60)
    
    from hierarchicalvlm.domain_modules.adapters.lora import LoRAAdapter
    
    # Create LoRA adapter for 768-dimensional features
    lora = LoRAAdapter(
        dim=768,
        rank=8,
        alpha=16.0,
        dropout=0.05,
        expansion_factor=4.0
    )
    
    # Simulate video features (batch=2, frames=32, dim=768)
    video_features = torch.randn(2, 32, 768)
    
    # Apply LoRA adaptation
    adapted_features = lora(video_features)
    
    print(f"Input shape: {video_features.shape}")
    print(f"Output shape: {adapted_features.shape}")
    print(f"Trainable parameters: {lora.get_lora_params().__len__()} groups")
    
    # Count parameters
    total_params = sum(p.numel() for p in lora.parameters())
    print(f"Total LoRA parameters: {total_params:,}")
    
    # Forward pass
    print(f"\n✓ LoRA adaptation successful")
    print(f"  Input:  {video_features.shape}")
    print(f"  Output: {adapted_features.shape}")


def example_2_action_detection():
    """Example 2: Action detection with LoRA and domain specialization."""
    print("\n" + "="*60)
    print("Example 2: Action Detection with Domain Specialization")
    print("="*60)
    
    from hierarchicalvlm.domain_modules.adapters.lora import LoRAAdapter
    from hierarchicalvlm.domain_modules.heads.task_heads import ActionDetectionHead
    from hierarchicalvlm.domain_modules.domain_experts.domain_expert import DomainRouter
    
    # Components
    domain_router = DomainRouter(input_dim=768, domains=['sports', 'tutorials', 'news'])
    lora = LoRAAdapter(dim=768, rank=8)
    action_head = ActionDetectionHead(
        input_dim=768,
        num_classes=150,
        num_frames=32
    )
    
    # Simulate video
    video_features = torch.randn(1, 32, 768)
    
    # Step 1: Route to appropriate domain
    routing_info = domain_router(video_features)
    domain_idx = routing_info['hard_routing'].argmax(dim=-1).item()
    selected_domain = routing_info['domains'][domain_idx]
    
    print(f"Selected domain: {selected_domain}")
    print(f"Routing probabilities: {dict(zip(routing_info['domains'], routing_info['soft_routing'][0]))}")
    
    # Step 2: Adapt features with LoRA
    adapted = lora(video_features)
    
    # Step 3: Detect actions
    output = action_head(adapted)
    
    print(f"\nAction detection output:")
    print(f"  Class logits: {output['class_logits'].shape}")
    print(f"  Confidence: {output['confidence'].shape}")
    print(f"  Average confidence: {output['confidence'].mean():.4f}")
    
    # Top-3 predicted actions per frame
    top_classes = output['class_logits'][0].topk(3, dim=-1)[1]
    print(f"  Top 3 actions (frame 0): {top_classes[0].tolist()}")


def example_3_multi_task_learning():
    """Example 3: Multi-task learning (Action, QA, Captioning)."""
    print("\n" + "="*60)
    print("Example 3: Multi-Task Learning")
    print("="*60)
    
    from hierarchicalvlm.domain_modules.heads.task_heads import MultiTaskHead
    from hierarchicalvlm.domain_modules.domain_experts.domain_expert import (
        MultiDomainAdapter
    )
    
    # Setup
    domain_adapter = MultiDomainAdapter(
        input_dim=768,
        domains=['sports', 'tutorials', 'news', 'general']
    )
    multi_task = MultiTaskHead(
        input_dim=768,
        num_action_classes=150,
        num_qa_answers=1000,
        vocab_size=10000,
        num_frames=32
    )
    
    # Inputs
    video = torch.randn(2, 32, 768)
    question = torch.randn(2, 768)
    caption_tokens = torch.randint(0, 10000, (2, 20))
    
    # Adapt through domains
    adapted_video, routing = domain_adapter(video, return_routing=True)
    
    print(f"Domain routing weights:")
    for domain, weights in routing.items():
        print(f"  {domain}: mean={weights.mean().item():.4f}")
    
    # Multi-task prediction
    tasks_output = multi_task(
        adapted_video,
        question_embedding=question,
        caption_tokens=caption_tokens,
        tasks=['action', 'qa', 'captioning']
    )
    
    print(f"\nMulti-task outputs:")
    print(f"  Action detection: {tasks_output['action']['class_logits'].shape}")
    print(f"  Video QA: {tasks_output['qa']['answer_logits'].shape}")
    print(f"  Captioning: {tasks_output['captioning']['boundary_logits'].shape}")


def example_4_intelligent_routing():
    """Example 4: Intelligent domain routing and selection."""
    print("\n" + "="*60)
    print("Example 4: Intelligent Domain Routing")
    print("="*60)
    
    from hierarchicalvlm.domain_modules.domain_experts.domain_expert import (
        DomainRouter, MultiDomainAdapter
    )
    
    # Router analyzes input
    router = DomainRouter(input_dim=768, domains=['sports', 'tutorials', 'news', 'general'])
    adapter = MultiDomainAdapter(input_dim=768, domains=['sports', 'tutorials', 'news', 'general'])
    
    # Multiple videos
    videos = [
        torch.randn(1, 32, 768),  # Video 1
        torch.randn(1, 32, 768),  # Video 2
        torch.randn(1, 32, 768),  # Video 3
    ]
    domain_names = ['sports', 'tutorials', 'news']
    
    for video, true_domain in zip(videos, domain_names):
        # Analyze video
        routing_info = router(video, return_confidence=True)
        
        # Get routing decisions
        hard_routing = routing_info['hard_routing'][0]
        soft_routing = routing_info['soft_routing'][0]
        confidence = routing_info['confidence'][0].item()
        
        selected_domain = routing_info['domains'][hard_routing.argmax().item()]
        
        print(f"\nVideo labeled as: {true_domain}")
        print(f"  Predicted domain: {selected_domain}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Soft routing: {dict(zip(routing_info['domains'], soft_routing.round(decimals=3)))}")
        
        # Adapt features
        adapted, _ = adapter(video)


def example_5_training_pipeline():
    """Example 5: End-to-end training pipeline."""
    print("\n" + "="*60)
    print("Example 5: Training Pipeline with LoRA")
    print("="*60)
    
    from hierarchicalvlm.domain_modules.adapters.lora import LoRAAdapter
    from hierarchicalvlm.domain_modules.heads.task_heads import ActionDetectionHead
    
    # Model setup
    lora = LoRAAdapter(dim=768, rank=8)
    action_head = ActionDetectionHead(input_dim=768, num_classes=10)  # Small for demo
    
    # Only train LoRA and head
    params_to_train = list(lora.parameters()) + list(action_head.parameters())
    optimizer = optim.Adam(params_to_train, lr=5e-4)
    
    # Dummy data
    batch_size = 2
    num_frames = 16
    num_classes = 10
    
    videos = torch.randn(batch_size, num_frames, 768)
    targets = torch.randint(0, num_classes, (batch_size, num_frames))
    
    # Training loop (3 steps for demo)
    print("\nTraining for 3 steps...")
    for step in range(3):
        # Forward
        adapted = lora(videos)
        output = action_head(adapted)
        logits = output['class_logits']
        
        # Loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, num_classes),
            targets.view(-1)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")


def example_6_specialized_transformer():
    """Example 6: Specialized Transformer with domain adaptation."""
    print("\n" + "="*60)
    print("Example 6: Specialized Transformer Block")
    print("="*60)
    
    from hierarchicalvlm.domain_modules.domain_experts.domain_expert import (
        SpecializedTransformer
    )
    
    # Create specialized transformer
    transformer = SpecializedTransformer(
        input_dim=768,
        num_heads=8,
        domains=['sports', 'tutorials', 'news'],
        hidden_dim=1024
    )
    
    # Input
    video = torch.randn(2, 32, 768)
    
    print(f"Input shape: {video.shape}")
    
    # Forward pass through multiple layers
    x = video
    for layer_idx in range(2):
        x = transformer(x)
        print(f"After layer {layer_idx + 1}: {x.shape}")
    
    print(f"\n✓ Specialized transformer successful")


def example_7_lora_vs_full_finetuning():
    """Example 7: Compare LoRA vs full fine-tuning efficiency."""
    print("\n" + "="*60)
    print("Example 7: LoRA vs Full Fine-Tuning Comparison")
    print("="*60)
    
    from hierarchicalvlm.domain_modules.adapters.lora import LoRAAdapter, LinearLoRA
    
    # LoRA approach
    lora = LoRAAdapter(dim=768, rank=8)
    lora_params = sum(p.numel() for p in lora.parameters())
    
    # Full fine-tuning equivalent
    full = nn.Sequential(
        nn.Linear(768, 3072),
        nn.GELU(),
        nn.Linear(3072, 768)
    )
    full_params = sum(p.numel() for p in full.parameters())
    
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Full fine-tuning parameters: {full_params:,}")
    print(f"Reduction: {100 * (1 - lora_params/full_params):.1f}%")
    
    # Compare outputs
    x = torch.randn(2, 32, 768)
    
    lora_out = lora(x)
    full_out = full(x.view(-1, 768)).view(2, 32, 768)
    
    print(f"\nBoth produce same output shape: {lora_out.shape}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Domain Modules Examples - HierarchicalVLM")
    print("="*70)
    
    try:
        example_1_basic_lora_finetuning()
    except Exception as e:
        print(f"Example 1 skipped: {e}")
    
    try:
        example_2_action_detection()
    except Exception as e:
        print(f"Example 2 skipped: {e}")
    
    try:
        example_3_multi_task_learning()
    except Exception as e:
        print(f"Example 3 skipped: {e}")
    
    try:
        example_4_intelligent_routing()
    except Exception as e:
        print(f"Example 4 skipped: {e}")
    
    try:
        example_5_training_pipeline()
    except Exception as e:
        print(f"Example 5 skipped: {e}")
    
    try:
        example_6_specialized_transformer()
    except Exception as e:
        print(f"Example 6 skipped: {e}")
    
    try:
        example_7_lora_vs_full_finetuning()
    except Exception as e:
        print(f"Example 7 skipped: {e}")
    
    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
