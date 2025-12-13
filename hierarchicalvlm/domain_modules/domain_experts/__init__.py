"""
Domain Expert Modules - Multi-Domain Specialization

This module provides domain specialization components:
- DomainExpert: Individual domain specialist module
- MultiDomainAdapter: Routes through multiple experts
- DomainRouter: Intelligent routing to domain experts
- SpecializedTransformer: Transformer with domain specialization
"""

from .domain_expert import (
    DomainExpert,
    MultiDomainAdapter,
    DomainRouter,
    SpecializedTransformer
)

__all__ = [
    'DomainExpert',
    'MultiDomainAdapter',
    'DomainRouter',
    'SpecializedTransformer'
]
