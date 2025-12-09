"""
Base classes for the Context Augmented Generation framework.

Defines the core abstractions and contracts for CAG components,
following lightweight contract patterns for extensibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context that can be built for analysis."""
    DEMOGRAPHIC = "demographic"
    ECONOMIC = "economic"
    HOUSING = "housing"
    COMMUNITY = "community"
    HISTORICAL = "historical"
    POLICY = "policy"
    SENTIMENT = "sentiment"
    GEOGRAPHIC = "geographic"


class InterpretationMode(Enum):
    """Modes for reality interpretation."""
    STATISTICAL = "statistical"      # Pure statistical summary
    CONTEXTUAL = "contextual"        # Stats + domain context
    NARRATIVE = "narrative"          # Human-readable story
    POLICY = "policy"                # Policy implications
    COMMUNITY = "community"          # Community impact focus


@dataclass
class CAGContext:
    """
    Rich context container for LLM-guided analysis.

    Dynamically constructed at runtime with domain knowledge,
    historical patterns, and community-specific information.
    """
    # Core identification
    analysis_type: str
    geographic_scope: str  # ZIP code, community area, or region
    time_period: str

    # Domain knowledge layers
    domain_context: Dict[str, Any] = field(default_factory=dict)
    historical_patterns: Dict[str, Any] = field(default_factory=dict)
    community_profiles: Dict[str, Any] = field(default_factory=dict)

    # Statistical foundation
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    model_outputs: Dict[str, Any] = field(default_factory=dict)

    # Interpretation guidance
    known_anomalies: List[str] = field(default_factory=list)
    policy_considerations: List[str] = field(default_factory=list)

    # Metadata
    context_types: List[ContextType] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def to_prompt_context(self) -> str:
        """
        Serialize context to a format suitable for LLM prompts.

        Returns enriched context string that bridges raw data
        with interpretive framework.
        """
        sections = []

        # Geographic and temporal framing
        sections.append(f"## Analysis Context")
        sections.append(f"- **Scope**: {self.geographic_scope}")
        sections.append(f"- **Period**: {self.time_period}")
        sections.append(f"- **Type**: {self.analysis_type}")
        sections.append("")

        # Domain knowledge
        if self.domain_context:
            sections.append("## Domain Knowledge")
            for key, value in self.domain_context.items():
                if isinstance(value, list):
                    sections.append(f"### {key.replace('_', ' ').title()}")
                    for item in value:
                        sections.append(f"- {item}")
                else:
                    sections.append(f"- **{key}**: {value}")
            sections.append("")

        # Historical patterns
        if self.historical_patterns:
            sections.append("## Historical Patterns")
            for pattern, details in self.historical_patterns.items():
                sections.append(f"### {pattern}")
                if isinstance(details, dict):
                    for k, v in details.items():
                        sections.append(f"- {k}: {v}")
                else:
                    sections.append(f"- {details}")
            sections.append("")

        # Community context
        if self.community_profiles:
            sections.append("## Community Context")
            for community, profile in self.community_profiles.items():
                sections.append(f"### {community}")
                if isinstance(profile, dict):
                    for k, v in profile.items():
                        sections.append(f"- {k}: {v}")
            sections.append("")

        # Known anomalies to watch for
        if self.known_anomalies:
            sections.append("## Known Anomalies and Considerations")
            for anomaly in self.known_anomalies:
                sections.append(f"- {anomaly}")
            sections.append("")

        # Policy context
        if self.policy_considerations:
            sections.append("## Policy Considerations")
            for consideration in self.policy_considerations:
                sections.append(f"- {consideration}")
            sections.append("")

        return "\n".join(sections)

    def merge_with(self, other: 'CAGContext') -> 'CAGContext':
        """Merge two contexts, combining their knowledge."""
        merged = CAGContext(
            analysis_type=f"{self.analysis_type}+{other.analysis_type}",
            geographic_scope=self.geographic_scope,
            time_period=self.time_period,
        )

        # Merge dictionaries
        merged.domain_context = {**self.domain_context, **other.domain_context}
        merged.historical_patterns = {**self.historical_patterns, **other.historical_patterns}
        merged.community_profiles = {**self.community_profiles, **other.community_profiles}
        merged.statistical_summary = {**self.statistical_summary, **other.statistical_summary}
        merged.model_outputs = {**self.model_outputs, **other.model_outputs}

        # Merge lists
        merged.known_anomalies = list(set(self.known_anomalies + other.known_anomalies))
        merged.policy_considerations = list(set(self.policy_considerations + other.policy_considerations))
        merged.context_types = list(set(self.context_types + other.context_types))

        return merged

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            'analysis_type': self.analysis_type,
            'geographic_scope': self.geographic_scope,
            'time_period': self.time_period,
            'domain_context': self.domain_context,
            'historical_patterns': self.historical_patterns,
            'community_profiles': self.community_profiles,
            'statistical_summary': self.statistical_summary,
            'model_outputs': self.model_outputs,
            'known_anomalies': self.known_anomalies,
            'policy_considerations': self.policy_considerations,
            'context_types': [ct.value for ct in self.context_types],
            'created_at': self.created_at.isoformat(),
            'version': self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CAGContext':
        """Deserialize from dictionary."""
        context = cls(
            analysis_type=data['analysis_type'],
            geographic_scope=data['geographic_scope'],
            time_period=data['time_period'],
        )
        context.domain_context = data.get('domain_context', {})
        context.historical_patterns = data.get('historical_patterns', {})
        context.community_profiles = data.get('community_profiles', {})
        context.statistical_summary = data.get('statistical_summary', {})
        context.model_outputs = data.get('model_outputs', {})
        context.known_anomalies = data.get('known_anomalies', [])
        context.policy_considerations = data.get('policy_considerations', [])
        context.context_types = [ContextType(ct) for ct in data.get('context_types', [])]
        if 'created_at' in data:
            context.created_at = datetime.fromisoformat(data['created_at'])
        context.version = data.get('version', '1.0')
        return context


@dataclass
class CAGResult:
    """
    Result container for CAG-enhanced analysis.

    Combines statistical outputs with contextual interpretations
    and suggested next steps.
    """
    # Core results
    analysis_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Statistical foundation (from existing models)
    statistical_results: Dict[str, Any] = field(default_factory=dict)

    # CAG enhancements
    contextual_interpretation: str = ""
    lived_reality_bridge: str = ""
    community_impact_narrative: str = ""

    # Pattern discovery
    discovered_patterns: List[Dict[str, Any]] = field(default_factory=list)
    anomalies_identified: List[Dict[str, Any]] = field(default_factory=list)

    # Forward-looking
    suggested_investigations: List[str] = field(default_factory=list)
    policy_implications: List[str] = field(default_factory=list)

    # Quality indicators
    confidence_assessment: Dict[str, float] = field(default_factory=dict)
    data_gaps_identified: List[str] = field(default_factory=list)

    # Metadata
    context_used: Optional[CAGContext] = None
    interpretation_mode: InterpretationMode = InterpretationMode.CONTEXTUAL

    def to_report_section(self) -> str:
        """Generate a report-ready markdown section."""
        sections = []

        sections.append("## Contextual Analysis")
        sections.append("")

        if self.contextual_interpretation:
            sections.append("### Statistical Interpretation")
            sections.append(self.contextual_interpretation)
            sections.append("")

        if self.lived_reality_bridge:
            sections.append("### Bridging to Lived Reality")
            sections.append(self.lived_reality_bridge)
            sections.append("")

        if self.community_impact_narrative:
            sections.append("### Community Impact")
            sections.append(self.community_impact_narrative)
            sections.append("")

        if self.discovered_patterns:
            sections.append("### Discovered Patterns")
            for pattern in self.discovered_patterns:
                sections.append(f"- **{pattern.get('name', 'Pattern')}**: {pattern.get('description', '')}")
            sections.append("")

        if self.anomalies_identified:
            sections.append("### Anomalies Requiring Attention")
            for anomaly in self.anomalies_identified:
                sections.append(f"- **{anomaly.get('type', 'Anomaly')}** in {anomaly.get('location', 'multiple areas')}: {anomaly.get('description', '')}")
            sections.append("")

        if self.policy_implications:
            sections.append("### Policy Implications")
            for implication in self.policy_implications:
                sections.append(f"- {implication}")
            sections.append("")

        if self.suggested_investigations:
            sections.append("### Suggested Next Investigations")
            for suggestion in self.suggested_investigations:
                sections.append(f"- {suggestion}")
            sections.append("")

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'analysis_id': self.analysis_id,
            'timestamp': self.timestamp.isoformat(),
            'statistical_results': self.statistical_results,
            'contextual_interpretation': self.contextual_interpretation,
            'lived_reality_bridge': self.lived_reality_bridge,
            'community_impact_narrative': self.community_impact_narrative,
            'discovered_patterns': self.discovered_patterns,
            'anomalies_identified': self.anomalies_identified,
            'suggested_investigations': self.suggested_investigations,
            'policy_implications': self.policy_implications,
            'confidence_assessment': self.confidence_assessment,
            'data_gaps_identified': self.data_gaps_identified,
            'interpretation_mode': self.interpretation_mode.value,
            'context_used': self.context_used.to_dict() if self.context_used else None,
        }


class CAGComponent(ABC):
    """
    Abstract base class for all CAG components.

    Defines the lightweight contract that all CAG components must follow,
    while allowing flexibility in implementation.
    """

    def __init__(self, name: str, output_dir: Optional[Path] = None):
        """
        Initialize CAG component.

        Args:
            name: Component identifier
            output_dir: Directory for component outputs
        """
        self.name = name
        self.output_dir = output_dir or Path("output/cag")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"cag.{name}")

        # Component state
        self._initialized = False
        self._last_run = None
        self._run_count = 0

    @abstractmethod
    def process(self, data: Any, context: Optional[CAGContext] = None) -> CAGResult:
        """
        Process data with optional context enhancement.

        Args:
            data: Input data (varies by component)
            context: Optional CAG context for enrichment

        Returns:
            CAGResult with enhanced analysis
        """
        pass

    def initialize(self) -> bool:
        """
        Initialize component resources.

        Override in subclasses for component-specific initialization.
        Returns True if initialization successful.
        """
        self._initialized = True
        self.logger.info(f"Initialized {self.name}")
        return True

    def cleanup(self) -> None:
        """Clean up component resources."""
        self._initialized = False
        self.logger.info(f"Cleaned up {self.name}")

    def save_result(self, result: CAGResult, filename: Optional[str] = None) -> Path:
        """Save a CAG result to disk."""
        if filename is None:
            filename = f"{self.name}_{result.analysis_id}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        self.logger.info(f"Saved result to {output_path}")
        return output_path

    def _update_run_stats(self) -> None:
        """Update internal run statistics."""
        self._last_run = datetime.now()
        self._run_count += 1


class CAGPlugin:
    """
    Plugin interface for integrating CAG with existing pipeline.

    Allows CAG components to be attached to the pipeline without
    modifying the core pipeline code. Follows the principle of
    "build extension points, not predetermined experiences."
    """

    def __init__(self, name: str, components: List[CAGComponent]):
        """
        Initialize plugin with CAG components.

        Args:
            name: Plugin identifier
            components: List of CAG components to include
        """
        self.name = name
        self.components = {c.name: c for c in components}
        self.enabled = True
        self.logger = logging.getLogger(f"cag.plugin.{name}")

    def pre_pipeline(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called before pipeline execution.

        Can modify pipeline state or prepare CAG components.
        """
        return pipeline_state

    def post_data_collection(
        self,
        collected_data: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook called after data collection phase.

        Opportunity to build context from raw data.
        """
        return collected_data

    def post_model_execution(
        self,
        model_results: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook called after model execution.

        Primary integration point for reality interpretation.
        """
        return model_results

    def post_report_generation(
        self,
        reports: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook called after report generation.

        Can enhance reports with contextual sections.
        """
        return reports

    def post_pipeline(
        self,
        pipeline_results: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook called after pipeline completion.

        Final opportunity for blueprint generation and suggestions.
        """
        return pipeline_results
