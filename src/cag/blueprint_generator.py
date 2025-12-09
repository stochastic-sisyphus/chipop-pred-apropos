"""
Blueprint Generator - Suggests Next-Step Analyses.

Implements the CAG principle of self-directed discovery by generating
"blueprints" for subsequent analyses based on findings from current runs.

This creates a research progression pathway that:
- Identifies gaps in current analysis
- Suggests feature engineering opportunities
- Proposes validation approaches
- Recommends data collection priorities
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import json
import logging

from .base import (
    CAGComponent,
    CAGContext,
    CAGResult,
)
from .pattern_discovery import DiscoveredPattern, FeatureSuggestion, ConfidenceLevel, PatternType

logger = logging.getLogger(__name__)


@dataclass
class AnalysisBlueprint:
    """
    Blueprint for a suggested follow-up analysis.

    Provides structured guidance for next-step investigations
    without prescribing specific implementations.
    """
    blueprint_id: str
    name: str
    description: str

    # Analysis type and scope
    analysis_type: str  # e.g., "deep_dive", "validation", "expansion", "new_dimension"
    geographic_scope: List[str] = field(default_factory=list)
    time_scope: str = ""

    # What prompted this suggestion
    source_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    source_findings: List[str] = field(default_factory=list)

    # Objectives
    primary_objective: str = ""
    secondary_objectives: List[str] = field(default_factory=list)

    # Data requirements
    required_data: List[str] = field(default_factory=list)
    optional_data: List[str] = field(default_factory=list)
    data_gaps: List[str] = field(default_factory=list)

    # Expected outcomes
    expected_insights: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    # Priority and effort
    priority: str = "medium"  # "high", "medium", "low"
    estimated_complexity: str = "medium"  # "low", "medium", "high"

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None  # Some blueprints become stale

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'blueprint_id': self.blueprint_id,
            'name': self.name,
            'description': self.description,
            'analysis_type': self.analysis_type,
            'geographic_scope': self.geographic_scope,
            'time_scope': self.time_scope,
            'source_patterns': self.source_patterns,
            'source_findings': self.source_findings,
            'primary_objective': self.primary_objective,
            'secondary_objectives': self.secondary_objectives,
            'required_data': self.required_data,
            'optional_data': self.optional_data,
            'data_gaps': self.data_gaps,
            'expected_insights': self.expected_insights,
            'success_criteria': self.success_criteria,
            'priority': self.priority,
            'estimated_complexity': self.estimated_complexity,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }

    def to_report_section(self) -> str:
        """Generate report section for this blueprint."""
        sections = []

        priority_indicator = {
            'high': 'HIGH PRIORITY',
            'medium': 'MEDIUM PRIORITY',
            'low': 'LOW PRIORITY',
        }

        sections.append(f"### {self.name}")
        sections.append(f"**{priority_indicator.get(self.priority, 'PRIORITY')}** | Complexity: {self.estimated_complexity}")
        sections.append("")
        sections.append(self.description)
        sections.append("")

        sections.append(f"**Primary Objective**: {self.primary_objective}")

        if self.secondary_objectives:
            sections.append("**Secondary Objectives**:")
            for obj in self.secondary_objectives:
                sections.append(f"- {obj}")

        sections.append("")

        if self.geographic_scope:
            sections.append(f"**Geographic Scope**: {', '.join(self.geographic_scope)}")

        if self.time_scope:
            sections.append(f"**Time Scope**: {self.time_scope}")

        sections.append("")

        if self.required_data:
            sections.append("**Required Data**:")
            for data in self.required_data:
                sections.append(f"- {data}")

        if self.data_gaps:
            sections.append("**Data Gaps to Address**:")
            for gap in self.data_gaps:
                sections.append(f"- {gap}")

        sections.append("")

        if self.expected_insights:
            sections.append("**Expected Insights**:")
            for insight in self.expected_insights:
                sections.append(f"- {insight}")

        if self.success_criteria:
            sections.append("**Success Criteria**:")
            for criterion in self.success_criteria:
                sections.append(f"- {criterion}")

        sections.append("")
        return "\n".join(sections)


class BlueprintGenerator(CAGComponent):
    """
    Generates blueprints for follow-up analyses.

    Analyzes current findings and suggests next-step investigations,
    implementing the CAG principle of self-directed discovery.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize Blueprint Generator.

        Args:
            output_dir: Directory for blueprint outputs
        """
        super().__init__("blueprint_generator", output_dir)

        # Blueprint storage
        self.blueprints: List[AnalysisBlueprint] = []

        # Configuration
        self.max_blueprints_per_run = 5
        self.generate_validation_blueprints = True
        self.generate_expansion_blueprints = True

    def process(
        self,
        data: Dict[str, Any],
        context: Optional[CAGContext] = None
    ) -> CAGResult:
        """
        Generate analysis blueprints based on findings.

        Args:
            data: Analysis results including patterns and interpretations
            context: CAG context

        Returns:
            CAGResult with generated blueprints
        """
        self._update_run_stats()

        if context is None:
            context = CAGContext(
                analysis_type="blueprint_generation",
                geographic_scope="Chicago",
                time_period="current",
            )

        # Generate blueprints
        blueprints = self.generate_blueprints(data, context)

        # Create result
        result = CAGResult(
            analysis_id=f"blueprints_{self._run_count}",
            context_used=context,
        )

        result.suggested_investigations = [b.primary_objective for b in blueprints]
        result.contextual_interpretation = self._generate_summary(blueprints)

        # Add blueprints as discovered patterns (for report integration)
        for blueprint in blueprints:
            result.discovered_patterns.append({
                'name': blueprint.name,
                'description': blueprint.description,
                'type': 'suggested_analysis',
                'priority': blueprint.priority,
            })

        # Store blueprints
        self.blueprints.extend(blueprints)

        self.logger.info(f"Generated {len(blueprints)} analysis blueprints")
        return result

    def generate_blueprints(
        self,
        data: Dict[str, Any],
        context: CAGContext
    ) -> List[AnalysisBlueprint]:
        """
        Generate analysis blueprints from current findings.

        Args:
            data: Analysis results
            context: CAG context

        Returns:
            List of suggested analysis blueprints
        """
        blueprints = []

        # Extract patterns and findings
        patterns = self._extract_patterns(data)
        data_gaps = self._identify_data_gaps(data, context)

        # Generate validation blueprints for high-confidence patterns
        if self.generate_validation_blueprints:
            for pattern in patterns:
                if pattern.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MODERATE):
                    blueprint = self._create_validation_blueprint(pattern, context)
                    if blueprint:
                        blueprints.append(blueprint)

        # Generate expansion blueprints for interesting findings
        if self.generate_expansion_blueprints:
            expansion = self._create_expansion_blueprints(patterns, context)
            blueprints.extend(expansion)

        # Generate gap-filling blueprints
        gap_blueprints = self._create_gap_blueprints(data_gaps, context)
        blueprints.extend(gap_blueprints)

        # Generate cross-dimensional analysis blueprints
        cross_dim = self._create_cross_dimensional_blueprints(data, context)
        blueprints.extend(cross_dim)

        # Prioritize and limit
        blueprints = self._prioritize_blueprints(blueprints)[:self.max_blueprints_per_run]

        return blueprints

    def _extract_patterns(self, data: Dict[str, Any]) -> List[DiscoveredPattern]:
        """Extract discovered patterns from data."""
        patterns = []

        # From pattern discovery results
        if 'discovered_patterns' in data:
            for p_dict in data['discovered_patterns']:
                if isinstance(p_dict, DiscoveredPattern):
                    patterns.append(p_dict)
                elif isinstance(p_dict, dict):
                    # Reconstruct pattern with proper enum handling
                    pattern_type_raw = p_dict.get('pattern_type', '')
                    try:
                        pattern_type = PatternType(pattern_type_raw) if pattern_type_raw else PatternType.COMMUNITY_INDICATOR
                    except ValueError:
                        pattern_type = PatternType.COMMUNITY_INDICATOR

                    confidence_raw = p_dict.get('confidence', 'exploratory')
                    try:
                        confidence = ConfidenceLevel(confidence_raw) if isinstance(confidence_raw, str) else confidence_raw
                    except ValueError:
                        confidence = ConfidenceLevel.EXPLORATORY

                    pattern = DiscoveredPattern(
                        pattern_id=p_dict.get('pattern_id', ''),
                        pattern_type=pattern_type,
                        name=p_dict.get('name', ''),
                        description=p_dict.get('description', ''),
                        confidence=confidence,
                        statistical_evidence=p_dict.get('statistical_evidence', {}),
                        contextual_evidence=p_dict.get('contextual_evidence', []),
                        affected_zip_codes=p_dict.get('affected_zip_codes', []),
                        spatial_relationship=p_dict.get('spatial_relationship', ''),
                        time_horizon=p_dict.get('time_horizon', ''),
                        suggested_investigations=p_dict.get('suggested_investigations', []),
                        potential_actions=p_dict.get('potential_actions', []),
                        data_requirements=p_dict.get('data_requirements', []),
                        validated=p_dict.get('validated', False),
                        validation_notes=p_dict.get('validation_notes', ''),
                    )
                    patterns.append(pattern)

        # From CAG results
        if 'cag_results' in data:
            for result in data['cag_results']:
                if hasattr(result, 'discovered_patterns'):
                    for p in result.discovered_patterns:
                        if isinstance(p, DiscoveredPattern):
                            patterns.append(p)
                        elif isinstance(p, dict):
                            # Recursively handle dict patterns from results
                            patterns.extend(self._extract_patterns({'discovered_patterns': [p]}))

        return patterns

    def _identify_data_gaps(
        self,
        data: Dict[str, Any],
        context: CAGContext
    ) -> List[str]:
        """Identify gaps in current data coverage."""
        gaps = []

        # Check for missing data types
        model_results = data.get('model_results', {})

        # Expected analyses vs actual
        expected_analyses = [
            'multifamily_growth',
            'retail_gap',
            'population_prediction',
            'income_distribution',
            'zoning_impact',
        ]

        for analysis in expected_analyses:
            if analysis not in model_results or not model_results[analysis]:
                gaps.append(f"Missing or incomplete {analysis.replace('_', ' ')} analysis")

        # Check geographic coverage
        all_zips = set()
        for model_name, results in model_results.items():
            if isinstance(results, dict):
                zips = results.get('zip_codes', [])
                all_zips.update(zips)

        known_zips = set(context.community_profiles.keys())
        uncovered = known_zips - all_zips
        if uncovered:
            gaps.append(f"Limited coverage for {len(uncovered)} ZIP codes with known profiles")

        # Check temporal coverage
        if 'time_series' not in str(data).lower():
            gaps.append("Time series analysis may enhance trend understanding")

        return gaps

    def _create_validation_blueprint(
        self,
        pattern: DiscoveredPattern,
        context: CAGContext
    ) -> Optional[AnalysisBlueprint]:
        """Create a blueprint to validate a discovered pattern."""
        return AnalysisBlueprint(
            blueprint_id=f"validate_{pattern.pattern_id}",
            name=f"Validate: {pattern.name}",
            description=(
                f"Validation analysis for '{pattern.name}'. "
                f"Current confidence: {pattern.confidence.value}. "
                f"This blueprint outlines steps to confirm or refute the pattern."
            ),
            analysis_type="validation",
            geographic_scope=pattern.affected_zip_codes,
            source_patterns=[pattern.pattern_id],
            primary_objective=f"Validate the {pattern.name} pattern with additional evidence",
            secondary_objectives=[
                "Quantify pattern strength",
                "Identify boundary conditions",
                "Assess reproducibility",
            ],
            required_data=pattern.data_requirements or [
                "Historical time series for affected areas",
                "Comparison data from similar neighborhoods",
            ],
            expected_insights=[
                f"Confirmed or refuted: {pattern.description[:100]}...",
                "Quantified effect size and significance",
            ],
            success_criteria=[
                "Statistical significance at p < 0.05",
                "Pattern holds across multiple time periods",
                "Community validation aligns with statistical findings",
            ],
            priority='high' if pattern.confidence == ConfidenceLevel.HIGH else 'medium',
            estimated_complexity='medium',
        )

    def _create_expansion_blueprints(
        self,
        patterns: List[DiscoveredPattern],
        context: CAGContext
    ) -> List[AnalysisBlueprint]:
        """Create blueprints to expand on interesting patterns."""
        blueprints = []

        # Group patterns by type
        by_type: Dict[str, List[DiscoveredPattern]] = {}
        for pattern in patterns:
            ptype = str(pattern.pattern_type)
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(pattern)

        # Create expansion blueprints for pattern clusters
        for ptype, type_patterns in by_type.items():
            if len(type_patterns) >= 2:
                affected_zips = []
                for p in type_patterns:
                    affected_zips.extend(p.affected_zip_codes)

                blueprints.append(AnalysisBlueprint(
                    blueprint_id=f"expand_{ptype}_{datetime.now().strftime('%Y%m%d')}",
                    name=f"Expand {ptype.replace('_', ' ').title()} Analysis",
                    description=(
                        f"Multiple {ptype.replace('_', ' ')} patterns detected across "
                        f"{len(set(affected_zips))} ZIP codes. Deeper analysis may reveal "
                        f"common drivers or interconnections."
                    ),
                    analysis_type="expansion",
                    geographic_scope=list(set(affected_zips)),
                    source_patterns=[p.pattern_id for p in type_patterns],
                    primary_objective=f"Identify common drivers of {ptype.replace('_', ' ')} patterns",
                    secondary_objectives=[
                        "Map causal relationships between patterns",
                        "Identify leading indicators",
                        "Assess intervention opportunities",
                    ],
                    required_data=[
                        "Detailed time series for affected areas",
                        f"{ptype.replace('_', ' ')} specific metrics",
                    ],
                    expected_insights=[
                        "Common drivers identified",
                        "Intervention points mapped",
                        "Predictive indicators established",
                    ],
                    priority='medium',
                    estimated_complexity='high',
                ))

        return blueprints

    def _create_gap_blueprints(
        self,
        gaps: List[str],
        context: CAGContext
    ) -> List[AnalysisBlueprint]:
        """Create blueprints to address data gaps."""
        blueprints = []

        for gap in gaps[:2]:  # Limit to top 2 gaps
            blueprints.append(AnalysisBlueprint(
                blueprint_id=f"gap_{hash(gap) % 10000}_{datetime.now().strftime('%Y%m%d')}",
                name=f"Address: {gap[:50]}...",
                description=f"Analysis to address identified gap: {gap}",
                analysis_type="gap_filling",
                source_findings=[gap],
                primary_objective=f"Fill data or analysis gap: {gap}",
                required_data=["Depends on specific gap"],
                data_gaps=[gap],
                priority='medium',
                estimated_complexity='medium',
            ))

        return blueprints

    def _create_cross_dimensional_blueprints(
        self,
        data: Dict[str, Any],
        context: CAGContext
    ) -> List[AnalysisBlueprint]:
        """Create blueprints for cross-dimensional analysis."""
        blueprints = []

        model_results = data.get('model_results', {})

        # If we have both demographic and economic data, suggest intersection analysis
        if 'income_distribution' in model_results and 'retail_gap' in model_results:
            blueprints.append(AnalysisBlueprint(
                blueprint_id=f"cross_income_retail_{datetime.now().strftime('%Y%m%d')}",
                name="Income-Retail Intersection Analysis",
                description=(
                    "Cross-dimensional analysis examining the relationship between "
                    "income distribution patterns and retail gap dynamics. May reveal "
                    "how economic changes manifest in service accessibility."
                ),
                analysis_type="cross_dimensional",
                primary_objective="Map relationship between income changes and retail accessibility",
                secondary_objectives=[
                    "Identify income thresholds for retail viability",
                    "Predict retail changes from income shifts",
                    "Assess service equity implications",
                ],
                required_data=[
                    "Income distribution by ZIP",
                    "Retail per capita by ZIP",
                    "Historical trends for both dimensions",
                ],
                expected_insights=[
                    "Income-retail relationship quantified",
                    "Threshold effects identified",
                    "Prediction model feasibility assessed",
                ],
                priority='medium',
                estimated_complexity='high',
            ))

        # If we have housing and demographic data, suggest displacement analysis
        if 'multifamily_growth' in model_results and 'population_prediction' in model_results:
            blueprints.append(AnalysisBlueprint(
                blueprint_id=f"cross_housing_demo_{datetime.now().strftime('%Y%m%d')}",
                name="Development-Displacement Intersection",
                description=(
                    "Cross-dimensional analysis examining the relationship between "
                    "housing development patterns and population changes. Critical for "
                    "understanding displacement dynamics."
                ),
                analysis_type="cross_dimensional",
                primary_objective="Quantify relationship between development and population change",
                secondary_objectives=[
                    "Identify displacement leading indicators",
                    "Map development-to-displacement timeline",
                    "Assess community stability thresholds",
                ],
                required_data=[
                    "Building permit data with types",
                    "Population change by ZIP",
                    "Demographic composition over time",
                ],
                expected_insights=[
                    "Development types most associated with displacement",
                    "Timeline from development to demographic shift",
                    "Community stability indicators",
                ],
                priority='high',
                estimated_complexity='high',
            ))

        return blueprints

    def _prioritize_blueprints(
        self,
        blueprints: List[AnalysisBlueprint]
    ) -> List[AnalysisBlueprint]:
        """Prioritize blueprints by importance."""
        priority_order = {'high': 0, 'medium': 1, 'low': 2}

        # Sort by priority, then by complexity (prefer simpler)
        return sorted(
            blueprints,
            key=lambda b: (
                priority_order.get(b.priority, 1),
                {'low': 0, 'medium': 1, 'high': 2}.get(b.estimated_complexity, 1)
            )
        )

    def _generate_summary(self, blueprints: List[AnalysisBlueprint]) -> str:
        """Generate summary of generated blueprints."""
        if not blueprints:
            return "No follow-up analyses suggested based on current findings."

        high_priority = [b for b in blueprints if b.priority == 'high']
        summary = f"Generated {len(blueprints)} analysis blueprints."

        if high_priority:
            summary += f" {len(high_priority)} high-priority: "
            summary += ", ".join(b.name for b in high_priority[:3])

        return summary

    def get_blueprints_by_type(
        self,
        analysis_type: str
    ) -> List[AnalysisBlueprint]:
        """Get blueprints filtered by analysis type."""
        return [b for b in self.blueprints if b.analysis_type == analysis_type]

    def generate_blueprint_report(self) -> str:
        """Generate comprehensive blueprint report."""
        sections = []

        sections.append("# Analysis Blueprint Report")
        sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        sections.append(f"Total Blueprints: {len(self.blueprints)}")
        sections.append("")

        # Group by type
        by_type: Dict[str, List[AnalysisBlueprint]] = {}
        for bp in self.blueprints:
            if bp.analysis_type not in by_type:
                by_type[bp.analysis_type] = []
            by_type[bp.analysis_type].append(bp)

        for analysis_type, bps in by_type.items():
            sections.append(f"## {analysis_type.replace('_', ' ').title()} Analyses")
            sections.append("")
            for bp in bps:
                sections.append(bp.to_report_section())

        return "\n".join(sections)
