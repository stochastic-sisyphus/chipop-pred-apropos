"""
Pattern Discovery Engine - LLM-Guided Analysis for Urban Data.

Implements dynamic pattern discovery that finds non-obvious relationships
in Chicago urban data through LLM-guided analysis.

Discovery Targets:
- Demographic transition indicators predicting development patterns
- Economic stress signals contradicting statistical growth metrics
- Cross-ZIP spillover effects and neighborhood interactions
- Infrastructure capacity constraints limiting statistical predictions

Design Philosophy:
- Suggest patterns rather than hardcode assumptions
- Maintain exploration flexibility
- Provide confidence/validation metrics
- Generate actionable insights for urban planning
- Preserve reversibility until patterns prove valuable
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Union
import json
import logging

import pandas as pd
import numpy as np

from .base import (
    CAGComponent,
    CAGContext,
    CAGResult,
    ContextType,
)

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns that can be discovered."""
    DEMOGRAPHIC_TRANSITION = "demographic_transition"
    ECONOMIC_DIVERGENCE = "economic_divergence"
    SPATIAL_SPILLOVER = "spatial_spillover"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    INFRASTRUCTURE_CONSTRAINT = "infrastructure_constraint"
    MARKET_SIGNAL = "market_signal"
    COMMUNITY_INDICATOR = "community_indicator"


class ConfidenceLevel(Enum):
    """Confidence levels for discovered patterns."""
    HIGH = "high"           # Strong statistical evidence
    MODERATE = "moderate"   # Some statistical support
    EXPLORATORY = "exploratory"  # Hypothesis requiring validation
    SPECULATIVE = "speculative"  # Weak evidence, needs investigation


@dataclass
class DiscoveredPattern:
    """
    A pattern discovered through analysis.

    Represents a potentially actionable insight with associated
    confidence and validation requirements.
    """
    pattern_id: str
    pattern_type: PatternType
    name: str
    description: str

    # Evidence and confidence
    confidence: ConfidenceLevel
    statistical_evidence: Dict[str, Any] = field(default_factory=dict)
    contextual_evidence: List[str] = field(default_factory=list)

    # Geographic scope
    affected_zip_codes: List[str] = field(default_factory=list)
    spatial_relationship: str = ""  # e.g., "adjacent", "corridor", "cluster"

    # Temporal characteristics
    time_horizon: str = ""  # e.g., "emerging", "established", "declining"
    first_detected: Optional[datetime] = None

    # Actionability
    suggested_investigations: List[str] = field(default_factory=list)
    potential_actions: List[str] = field(default_factory=list)
    data_requirements: List[str] = field(default_factory=list)

    # Validation status
    validated: bool = False
    validation_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'name': self.name,
            'description': self.description,
            'confidence': self.confidence.value,
            'statistical_evidence': self.statistical_evidence,
            'contextual_evidence': self.contextual_evidence,
            'affected_zip_codes': self.affected_zip_codes,
            'spatial_relationship': self.spatial_relationship,
            'time_horizon': self.time_horizon,
            'first_detected': self.first_detected.isoformat() if self.first_detected else None,
            'suggested_investigations': self.suggested_investigations,
            'potential_actions': self.potential_actions,
            'data_requirements': self.data_requirements,
            'validated': self.validated,
            'validation_notes': self.validation_notes,
        }

    def to_report_section(self) -> str:
        """Generate report section for this pattern."""
        sections = []

        confidence_emoji = {
            ConfidenceLevel.HIGH: "HIGH",
            ConfidenceLevel.MODERATE: "MODERATE",
            ConfidenceLevel.EXPLORATORY: "EXPLORATORY",
            ConfidenceLevel.SPECULATIVE: "SPECULATIVE",
        }

        sections.append(f"### {self.name}")
        sections.append(f"**Type**: {self.pattern_type.value.replace('_', ' ').title()}")
        sections.append(f"**Confidence**: {confidence_emoji.get(self.confidence, 'UNKNOWN')}")
        sections.append("")
        sections.append(self.description)
        sections.append("")

        if self.affected_zip_codes:
            sections.append(f"**Affected Areas**: {', '.join(self.affected_zip_codes)}")

        if self.spatial_relationship:
            sections.append(f"**Spatial Pattern**: {self.spatial_relationship}")

        if self.time_horizon:
            sections.append(f"**Time Horizon**: {self.time_horizon}")

        sections.append("")

        if self.suggested_investigations:
            sections.append("**Suggested Investigations**:")
            for inv in self.suggested_investigations:
                sections.append(f"- {inv}")
            sections.append("")

        if self.potential_actions:
            sections.append("**Potential Actions**:")
            for action in self.potential_actions:
                sections.append(f"- {action}")
            sections.append("")

        return "\n".join(sections)


@dataclass
class FeatureSuggestion:
    """
    A suggested feature for model enhancement.

    LLM-guided feature discovery suggests new features based on
    pattern analysis, allowing the model to evolve without
    hardcoding assumptions.
    """
    feature_name: str
    feature_type: str  # e.g., "ratio", "lag", "interaction", "spatial"
    description: str

    # Source data
    source_variables: List[str] = field(default_factory=list)
    calculation_logic: str = ""

    # Expected value
    hypothesis: str = ""
    expected_impact: str = ""

    # Validation requirements
    validation_approach: str = ""
    minimum_data_points: int = 30

    # Priority
    priority: str = "medium"  # "high", "medium", "low"
    rationale: str = ""


class PatternDetector(ABC):
    """
    Abstract detector for specific pattern types.

    Allows pluggable pattern detection strategies without
    modifying core discovery logic.
    """

    @property
    @abstractmethod
    def pattern_type(self) -> PatternType:
        """The type of pattern this detector finds."""
        pass

    @abstractmethod
    def detect(
        self,
        data: Dict[str, Any],
        context: CAGContext
    ) -> List[DiscoveredPattern]:
        """
        Detect patterns of this type in the data.

        Args:
            data: Pipeline data to analyze
            context: CAG context for interpretation

        Returns:
            List of discovered patterns
        """
        pass


class DemographicTransitionDetector(PatternDetector):
    """Detects demographic transition patterns."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.DEMOGRAPHIC_TRANSITION

    def detect(
        self,
        data: Dict[str, Any],
        context: CAGContext
    ) -> List[DiscoveredPattern]:
        """Detect demographic transition patterns."""
        patterns = []

        # Look for income shift patterns
        model_results = data.get('model_results', {})
        income_results = model_results.get('income_distribution', {})

        if income_results:
            # Check for income polarization
            for zip_code, metrics in income_results.get('zip_metrics', {}).items():
                if isinstance(metrics, dict):
                    income_change = metrics.get('median_income_change', 0)
                    income_diversity = metrics.get('income_diversity', 1.0)

                    # Rising incomes with decreasing diversity suggests gentrification
                    if income_change > 0.1 and income_diversity < 0.5:
                        profile = context.community_profiles.get(zip_code, {})

                        patterns.append(DiscoveredPattern(
                            pattern_id=f"demo_trans_{zip_code}_{datetime.now().strftime('%Y%m%d')}",
                            pattern_type=self.pattern_type,
                            name=f"Income Transition in {zip_code}",
                            description=(
                                f"ZIP code {zip_code} shows rising median income (+{income_change:.1%}) "
                                f"combined with decreasing income diversity. This pattern typically "
                                f"indicates demographic transition that may not be captured by "
                                f"aggregate population statistics."
                            ),
                            confidence=ConfidenceLevel.MODERATE,
                            statistical_evidence={
                                'income_change': income_change,
                                'income_diversity': income_diversity,
                            },
                            contextual_evidence=[
                                profile.get('historical', {}).get('gentrification', 'No historical context'),
                            ],
                            affected_zip_codes=[zip_code],
                            time_horizon='emerging',
                            suggested_investigations=[
                                "Track demographic composition changes over 5-year periods",
                                "Monitor small business turnover as early indicator",
                                "Analyze housing permit types (renovation vs new construction)",
                            ],
                            potential_actions=[
                                "Engage community organizations for ground-truth validation",
                                "Consider displacement monitoring program",
                            ],
                        ))

        # Look for population composition shifts
        population_results = model_results.get('population_prediction', {})
        if population_results:
            forecasts = population_results.get('forecasts', {})
            for zip_code, forecast in forecasts.items():
                # Look for areas with stable population but changing composition
                pop_change = forecast.get('change', 0) if isinstance(forecast, dict) else 0
                if abs(pop_change) < 0.02:  # Stable overall population
                    # Check context for transition indicators
                    profile = context.community_profiles.get(zip_code, {})
                    if profile:
                        housing = profile.get('housing', {})
                        if 'transition' in str(housing).lower() or 'changing' in str(housing).lower():
                            patterns.append(DiscoveredPattern(
                                pattern_id=f"demo_hidden_{zip_code}_{datetime.now().strftime('%Y%m%d')}",
                                pattern_type=self.pattern_type,
                                name=f"Hidden Demographic Shift in {zip_code}",
                                description=(
                                    f"ZIP code {zip_code} shows stable total population but contextual "
                                    f"indicators suggest underlying demographic composition change. "
                                    f"This 'hidden transition' may mask significant community change."
                                ),
                                confidence=ConfidenceLevel.EXPLORATORY,
                                affected_zip_codes=[zip_code],
                                time_horizon='uncertain',
                                suggested_investigations=[
                                    "Disaggregate population data by age, income, and ethnicity",
                                    "Analyze household size changes",
                                    "Review school enrollment patterns",
                                ],
                            ))

        return patterns


class EconomicDivergenceDetector(PatternDetector):
    """Detects economic reality gaps where stats diverge from experience."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.ECONOMIC_DIVERGENCE

    def detect(
        self,
        data: Dict[str, Any],
        context: CAGContext
    ) -> List[DiscoveredPattern]:
        """Detect economic divergence patterns."""
        patterns = []

        model_results = data.get('model_results', {})

        # Look for areas where economic indicators diverge from reality signals
        retail_results = model_results.get('retail_gap', {})
        income_results = model_results.get('income_distribution', {})

        if retail_results and income_results:
            retail_gaps = retail_results.get('gap_scores', {})
            income_metrics = income_results.get('zip_metrics', {})

            for zip_code in set(retail_gaps.keys()) & set(income_metrics.keys()):
                retail_gap = retail_gaps.get(zip_code, 0)
                income_data = income_metrics.get(zip_code, {})
                income_growth = income_data.get('income_growth', 0) if isinstance(income_data, dict) else 0

                # Paradox: Income growth but high retail gap
                if income_growth > 0.05 and retail_gap > 0.5:
                    patterns.append(DiscoveredPattern(
                        pattern_id=f"econ_div_{zip_code}_{datetime.now().strftime('%Y%m%d')}",
                        pattern_type=self.pattern_type,
                        name=f"Income-Service Paradox in {zip_code}",
                        description=(
                            f"ZIP code {zip_code} shows income growth ({income_growth:.1%}) "
                            f"but persistent retail/service gaps. This divergence suggests "
                            f"economic growth may not be translating to improved daily "
                            f"quality of life. The 'GDP growth vs felt recession' dynamic "
                            f"may be at play."
                        ),
                        confidence=ConfidenceLevel.MODERATE,
                        statistical_evidence={
                            'income_growth': income_growth,
                            'retail_gap': retail_gap,
                        },
                        affected_zip_codes=[zip_code],
                        suggested_investigations=[
                            "Analyze retail category composition (luxury vs essential)",
                            "Survey resident satisfaction with service accessibility",
                            "Map commute patterns for service access",
                        ],
                        potential_actions=[
                            "Prioritize essential retail recruitment",
                            "Investigate barriers to retail investment",
                        ],
                    ))

        return patterns


class SpatialSpilloverDetector(PatternDetector):
    """Detects cross-ZIP spillover effects and neighborhood interactions."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.SPATIAL_SPILLOVER

    def detect(
        self,
        data: Dict[str, Any],
        context: CAGContext
    ) -> List[DiscoveredPattern]:
        """Detect spatial spillover patterns."""
        patterns = []

        # Define adjacent ZIP code relationships for Chicago
        # This is a simplified adjacency - in production would use GIS data
        adjacency = {
            '60615': ['60637', '60619', '60649'],  # Hyde Park adjacent
            '60617': ['60619', '60633', '60649'],  # South Chicago adjacent
            '60619': ['60615', '60617', '60620', '60621'],  # Chatham adjacent
            '60620': ['60619', '60621', '60636'],  # Auburn Gresham
            '60621': ['60619', '60620', '60636', '60609'],  # Englewood
            '60622': ['60647', '60612', '60614'],  # Wicker Park
            '60647': ['60622', '60618', '60651', '60639'],  # Logan Square
            '60614': ['60622', '60610', '60657'],  # Lincoln Park
            '60607': ['60601', '60606', '60608', '60612'],  # West Loop
        }

        model_results = data.get('model_results', {})
        multifamily_results = model_results.get('multifamily_growth', {})

        if multifamily_results:
            growth_scores = multifamily_results.get('growth_scores', {})

            # Look for high-growth areas near low-growth areas (spillover potential)
            for zip_code, score in growth_scores.items():
                if score > 0.6:  # High growth
                    adjacent_zips = adjacency.get(zip_code, [])
                    low_growth_adjacent = [
                        adj for adj in adjacent_zips
                        if adj in growth_scores and growth_scores[adj] < 0.3
                    ]

                    if low_growth_adjacent:
                        patterns.append(DiscoveredPattern(
                            pattern_id=f"spillover_{zip_code}_{datetime.now().strftime('%Y%m%d')}",
                            pattern_type=self.pattern_type,
                            name=f"Development Spillover from {zip_code}",
                            description=(
                                f"High development activity in {zip_code} may create "
                                f"spillover effects into adjacent lower-activity areas "
                                f"({', '.join(low_growth_adjacent)}). Development pressure "
                                f"often spreads along transit corridors and displaces "
                                f"population to adjacent neighborhoods."
                            ),
                            confidence=ConfidenceLevel.EXPLORATORY,
                            statistical_evidence={
                                'source_growth': score,
                                'adjacent_growth': {adj: growth_scores.get(adj, 0) for adj in low_growth_adjacent},
                            },
                            affected_zip_codes=[zip_code] + low_growth_adjacent,
                            spatial_relationship='adjacent_spillover',
                            time_horizon='emerging',
                            suggested_investigations=[
                                "Track permit activity trends in adjacent areas",
                                "Monitor rent changes along the boundary",
                                "Analyze transit connectivity between areas",
                            ],
                            potential_actions=[
                                "Proactive affordability planning in adjacent areas",
                                "Community engagement before spillover occurs",
                            ],
                        ))

        return patterns


class TemporalAnomalyDetector(PatternDetector):
    """Detects unusual temporal patterns that may indicate shifts."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.TEMPORAL_ANOMALY

    def detect(
        self,
        data: Dict[str, Any],
        context: CAGContext
    ) -> List[DiscoveredPattern]:
        """Detect temporal anomaly patterns."""
        patterns = []

        model_results = data.get('model_results', {})
        population_results = model_results.get('population_prediction', {})

        if population_results:
            # Look for confidence interval anomalies
            forecasts = population_results.get('forecasts', {})

            for zip_code, forecast in forecasts.items():
                if isinstance(forecast, dict):
                    confidence_interval = forecast.get('confidence_interval', 0)
                    if confidence_interval > 0.3:  # Wide confidence interval
                        patterns.append(DiscoveredPattern(
                            pattern_id=f"temporal_{zip_code}_{datetime.now().strftime('%Y%m%d')}",
                            pattern_type=self.pattern_type,
                            name=f"High Forecast Uncertainty in {zip_code}",
                            description=(
                                f"Population forecasts for {zip_code} have unusually wide "
                                f"confidence intervals ({confidence_interval:.1%}), indicating "
                                f"high uncertainty. This may signal a neighborhood in transition "
                                f"where historical patterns are poor predictors of future trends."
                            ),
                            confidence=ConfidenceLevel.EXPLORATORY,
                            statistical_evidence={
                                'confidence_interval': confidence_interval,
                            },
                            affected_zip_codes=[zip_code],
                            time_horizon='uncertain',
                            suggested_investigations=[
                                "Identify factors causing forecast instability",
                                "Compare to similar neighborhoods' historical transitions",
                                "Look for structural breaks in historical data",
                            ],
                        ))

        return patterns


class PatternDiscovery(CAGComponent):
    """
    LLM-Guided Pattern Discovery Engine.

    Finds non-obvious relationships in Chicago urban data through
    dynamic pattern detection that suggests insights rather than
    hardcoding assumptions.

    Design Principles:
    - Suggest patterns rather than hardcode assumptions
    - Maintain exploration flexibility
    - Provide confidence/validation metrics
    - Generate actionable insights for urban planning
    - Preserve reversibility until patterns prove valuable
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize Pattern Discovery Engine.

        Args:
            output_dir: Directory for discovery outputs
        """
        super().__init__("pattern_discovery", output_dir)

        # Initialize detectors
        self.detectors: Dict[PatternType, PatternDetector] = {
            PatternType.DEMOGRAPHIC_TRANSITION: DemographicTransitionDetector(),
            PatternType.ECONOMIC_DIVERGENCE: EconomicDivergenceDetector(),
            PatternType.SPATIAL_SPILLOVER: SpatialSpilloverDetector(),
            PatternType.TEMPORAL_ANOMALY: TemporalAnomalyDetector(),
        }

        # Pattern storage
        self.discovered_patterns: List[DiscoveredPattern] = []
        self.feature_suggestions: List[FeatureSuggestion] = []

        # Discovery configuration
        self.min_confidence = ConfidenceLevel.EXPLORATORY
        self.active_pattern_types: Set[PatternType] = set(PatternType)

    def process(
        self,
        data: Dict[str, Any],
        context: Optional[CAGContext] = None
    ) -> CAGResult:
        """
        Run pattern discovery on pipeline data.

        Args:
            data: Pipeline data including model results
            context: CAG context for interpretation

        Returns:
            CAGResult with discovered patterns
        """
        self._update_run_stats()

        # Use provided context or create minimal
        if context is None:
            context = CAGContext(
                analysis_type="pattern_discovery",
                geographic_scope="Chicago",
                time_period="current",
            )

        # Run all active detectors
        patterns = self.discover_patterns(data, context)

        # Generate feature suggestions from patterns
        suggestions = self.suggest_features(patterns, data)

        # Create result
        result = CAGResult(
            analysis_id=f"discovery_{self._run_count}",
            context_used=context,
        )

        # Populate result
        result.discovered_patterns = [p.to_dict() for p in patterns]
        result.suggested_investigations = self._compile_investigations(patterns)
        result.contextual_interpretation = self._generate_discovery_summary(patterns)

        # Add high-confidence patterns as anomalies
        for pattern in patterns:
            if pattern.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MODERATE):
                result.anomalies_identified.append({
                    'type': pattern.pattern_type.value,
                    'location': ', '.join(pattern.affected_zip_codes),
                    'description': pattern.description,
                    'confidence': pattern.confidence.value,
                })

        # Store for later access
        self.discovered_patterns.extend(patterns)
        self.feature_suggestions.extend(suggestions)

        self.logger.info(f"Discovered {len(patterns)} patterns, {len(suggestions)} feature suggestions")
        return result

    def discover_patterns(
        self,
        data: Dict[str, Any],
        context: CAGContext
    ) -> List[DiscoveredPattern]:
        """
        Run pattern detection across all active detectors.

        Args:
            data: Pipeline data
            context: CAG context

        Returns:
            List of discovered patterns
        """
        all_patterns = []

        for pattern_type, detector in self.detectors.items():
            if pattern_type in self.active_pattern_types:
                try:
                    patterns = detector.detect(data, context)
                    # Filter by confidence threshold
                    patterns = [
                        p for p in patterns
                        if self._meets_confidence_threshold(p.confidence)
                    ]
                    all_patterns.extend(patterns)
                    self.logger.debug(f"Detector {pattern_type.value} found {len(patterns)} patterns")
                except Exception as e:
                    self.logger.warning(f"Detector {pattern_type.value} failed: {e}")

        return all_patterns

    def suggest_features(
        self,
        patterns: List[DiscoveredPattern],
        data: Dict[str, Any]
    ) -> List[FeatureSuggestion]:
        """
        Generate feature suggestions based on discovered patterns.

        This implements the CAG principle of suggesting features rather
        than hardcoding them, allowing the model to evolve based on
        discovered relationships.

        Args:
            patterns: Discovered patterns
            data: Pipeline data

        Returns:
            List of feature suggestions
        """
        suggestions = []

        for pattern in patterns:
            # Generate suggestions based on pattern type
            if pattern.pattern_type == PatternType.DEMOGRAPHIC_TRANSITION:
                suggestions.append(FeatureSuggestion(
                    feature_name='income_diversity_index',
                    feature_type='ratio',
                    description="Income diversity index measuring income distribution breadth",
                    source_variables=['median_income', 'income_percentiles'],
                    calculation_logic="Gini coefficient or entropy of income distribution",
                    hypothesis="Lower diversity predicts demographic transition",
                    expected_impact="Improve early detection of gentrification",
                    priority='high' if pattern.confidence == ConfidenceLevel.HIGH else 'medium',
                    rationale=f"Pattern '{pattern.name}' suggests income diversity is predictive",
                ))

            elif pattern.pattern_type == PatternType.SPATIAL_SPILLOVER:
                suggestions.append(FeatureSuggestion(
                    feature_name='adjacent_development_pressure',
                    feature_type='spatial',
                    description="Development pressure from adjacent ZIP codes",
                    source_variables=['permit_counts', 'adjacency_matrix'],
                    calculation_logic="Weighted sum of adjacent ZIP permit activity",
                    hypothesis="Development spills over from high-activity neighbors",
                    expected_impact="Predict development timing in adjacent areas",
                    priority='medium',
                    rationale=f"Pattern '{pattern.name}' shows spillover effects",
                ))

            elif pattern.pattern_type == PatternType.ECONOMIC_DIVERGENCE:
                suggestions.append(FeatureSuggestion(
                    feature_name='service_accessibility_index',
                    feature_type='composite',
                    description="Combined measure of essential service accessibility",
                    source_variables=['retail_per_capita', 'transit_access', 'service_categories'],
                    calculation_logic="Weighted index of essential service availability",
                    hypothesis="Low accessibility despite income growth indicates reality gap",
                    expected_impact="Bridge statistical growth to lived experience",
                    priority='high' if pattern.confidence == ConfidenceLevel.HIGH else 'medium',
                    rationale=f"Pattern '{pattern.name}' reveals economic-reality divergence",
                ))

            elif pattern.pattern_type == PatternType.TEMPORAL_ANOMALY:
                suggestions.append(FeatureSuggestion(
                    feature_name='forecast_stability_score',
                    feature_type='derived',
                    description="Measure of forecast stability over time windows",
                    source_variables=['historical_forecasts', 'actual_outcomes'],
                    calculation_logic="Rolling comparison of forecast accuracy",
                    hypothesis="Unstable forecasts indicate neighborhoods in transition",
                    expected_impact="Flag areas requiring alternative modeling approaches",
                    priority='medium',
                    rationale=f"Pattern '{pattern.name}' shows forecast instability",
                ))

        return suggestions

    def add_detector(
        self,
        pattern_type: PatternType,
        detector: PatternDetector
    ) -> None:
        """Add a custom pattern detector."""
        self.detectors[pattern_type] = detector
        self.active_pattern_types.add(pattern_type)
        self.logger.info(f"Added detector for {pattern_type.value}")

    def set_confidence_threshold(self, level: ConfidenceLevel) -> None:
        """Set minimum confidence level for reported patterns."""
        self.min_confidence = level
        self.logger.info(f"Confidence threshold set to {level.value}")

    def activate_pattern_types(self, types: List[PatternType]) -> None:
        """Set which pattern types to actively detect."""
        self.active_pattern_types = set(types)
        self.logger.info(f"Active pattern types: {[t.value for t in types]}")

    def get_high_priority_patterns(self) -> List[DiscoveredPattern]:
        """Get patterns with high or moderate confidence."""
        return [
            p for p in self.discovered_patterns
            if p.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MODERATE)
        ]

    def get_feature_suggestions_by_priority(
        self,
        priority: str = 'high'
    ) -> List[FeatureSuggestion]:
        """Get feature suggestions filtered by priority."""
        return [
            s for s in self.feature_suggestions
            if s.priority == priority
        ]

    def validate_pattern(
        self,
        pattern_id: str,
        validated: bool,
        notes: str = ""
    ) -> None:
        """Mark a pattern as validated or invalidated."""
        for pattern in self.discovered_patterns:
            if pattern.pattern_id == pattern_id:
                pattern.validated = validated
                pattern.validation_notes = notes
                self.logger.info(f"Pattern {pattern_id} validation: {validated}")
                return
        self.logger.warning(f"Pattern {pattern_id} not found")

    def generate_discovery_report(self) -> str:
        """Generate a comprehensive discovery report."""
        sections = []

        sections.append("# Pattern Discovery Report")
        sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        sections.append(f"Patterns Discovered: {len(self.discovered_patterns)}")
        sections.append(f"Feature Suggestions: {len(self.feature_suggestions)}")
        sections.append("")

        # Group patterns by type
        by_type: Dict[PatternType, List[DiscoveredPattern]] = {}
        for pattern in self.discovered_patterns:
            if pattern.pattern_type not in by_type:
                by_type[pattern.pattern_type] = []
            by_type[pattern.pattern_type].append(pattern)

        for pattern_type, patterns in by_type.items():
            sections.append(f"## {pattern_type.value.replace('_', ' ').title()}")
            sections.append("")
            for pattern in patterns:
                sections.append(pattern.to_report_section())
                sections.append("")

        # Feature suggestions
        if self.feature_suggestions:
            sections.append("## Suggested Features")
            sections.append("")
            for suggestion in self.feature_suggestions:
                sections.append(f"### {suggestion.feature_name}")
                sections.append(f"**Type**: {suggestion.feature_type}")
                sections.append(f"**Priority**: {suggestion.priority}")
                sections.append(f"**Description**: {suggestion.description}")
                sections.append(f"**Hypothesis**: {suggestion.hypothesis}")
                sections.append(f"**Rationale**: {suggestion.rationale}")
                sections.append("")

        return "\n".join(sections)

    def _meets_confidence_threshold(self, confidence: ConfidenceLevel) -> bool:
        """Check if a confidence level meets the threshold."""
        confidence_order = [
            ConfidenceLevel.SPECULATIVE,
            ConfidenceLevel.EXPLORATORY,
            ConfidenceLevel.MODERATE,
            ConfidenceLevel.HIGH,
        ]
        return confidence_order.index(confidence) >= confidence_order.index(self.min_confidence)

    def _compile_investigations(
        self,
        patterns: List[DiscoveredPattern]
    ) -> List[str]:
        """Compile suggested investigations from all patterns."""
        investigations = []
        for pattern in patterns:
            for inv in pattern.suggested_investigations:
                if inv not in investigations:
                    investigations.append(inv)
        return investigations

    def _generate_discovery_summary(
        self,
        patterns: List[DiscoveredPattern]
    ) -> str:
        """Generate summary of discovery results."""
        if not patterns:
            return "No significant patterns discovered in current analysis."

        high_confidence = [p for p in patterns if p.confidence == ConfidenceLevel.HIGH]
        moderate_confidence = [p for p in patterns if p.confidence == ConfidenceLevel.MODERATE]

        summary_parts = []

        if high_confidence:
            summary_parts.append(
                f"Discovered {len(high_confidence)} high-confidence pattern(s) "
                f"requiring attention: {', '.join(p.name for p in high_confidence[:3])}"
            )

        if moderate_confidence:
            summary_parts.append(
                f"Found {len(moderate_confidence)} moderate-confidence pattern(s) "
                f"warranting investigation"
            )

        total_exploratory = len(patterns) - len(high_confidence) - len(moderate_confidence)
        if total_exploratory > 0:
            summary_parts.append(
                f"Additionally identified {total_exploratory} exploratory patterns "
                f"for potential future investigation"
            )

        return ". ".join(summary_parts) + "."
