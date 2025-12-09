"""
Context Augmented Generation (CAG) Framework for Chicago Urban Analytics

This module implements the CAG methodology adapted from bronze__acs_eda,
providing contextual interpretation layers that bridge statistical metrics
with lived community reality.

Architecture Principles:
- Modular plugins, not system replacement
- Exploration-first pathways with lightweight contracts
- Build extension points, not predetermined experiences
- Preserve reversibility until value is proven

Components:
- ContextBuilder: Constructs enriched context for LLM analysis
- RealityInterpreter: Bridges quantitative metrics with qualitative reality
- PatternDiscovery: LLM-guided anomaly and pattern detection
- BlueprintGenerator: Suggests next-step analyses based on findings
- CAGPipeline: Optional orchestration layer wrapping existing pipeline
"""

from .base import CAGComponent, CAGContext, CAGResult
from .context_builder import ContextBuilder, ChicagoContextProfile
from .reality_interpreter import RealityInterpreter
from .pattern_discovery import (
    PatternDiscovery,
    PatternType,
    ConfidenceLevel,
    DiscoveredPattern,
    create_pattern,
)
from .blueprint_generator import BlueprintGenerator
from .pipeline import CAGPipeline

__all__ = [
    'CAGComponent',
    'CAGContext',
    'CAGResult',
    'ContextBuilder',
    'ChicagoContextProfile',
    'RealityInterpreter',
    'PatternDiscovery',
    'PatternType',
    'ConfidenceLevel',
    'DiscoveredPattern',
    'create_pattern',
    'BlueprintGenerator',
    'CAGPipeline',
]

__version__ = '0.1.0'
