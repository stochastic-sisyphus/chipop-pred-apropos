"""
Reality Interpreter - Bridging Statistics with Lived Community Reality.

Addresses the core CAG insight: "GDP says growth while people feel recession."
This module synthesizes statistical outputs with contextual understanding to
produce interpretations that reflect actual community experience.

Key Capabilities:
- Economic Reality Bridge: Interpret how stats manifest in lived experience
- Demographic Transition Analysis: What population changes mean for communities
- Development Pressure Synthesis: How market forces affect neighborhoods
- Community Stability Assessment: Beyond statistics to resilience indicators
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import logging

import pandas as pd
import numpy as np

from .base import (
    CAGComponent,
    CAGContext,
    CAGResult,
    ContextType,
    InterpretationMode,
)

logger = logging.getLogger(__name__)


class RealityDimension(Enum):
    """Dimensions of lived reality to interpret."""
    ECONOMIC_WELLBEING = "economic_wellbeing"
    HOUSING_SECURITY = "housing_security"
    COMMUNITY_STABILITY = "community_stability"
    SERVICE_ACCESSIBILITY = "service_accessibility"
    DISPLACEMENT_RISK = "displacement_risk"
    OPPORTUNITY_ACCESS = "opportunity_access"


@dataclass
class RealityIndicator:
    """
    An indicator that bridges a statistical measure to lived experience.

    Maps quantitative metrics to qualitative community impact assessments.
    """
    name: str
    statistical_source: str  # The underlying statistical measure
    reality_dimension: RealityDimension
    interpretation_template: str  # How to interpret this indicator

    # Threshold configurations
    concern_threshold: Optional[float] = None
    alert_threshold: Optional[float] = None

    # Contextual modifiers
    amplifying_factors: List[str] = field(default_factory=list)
    mitigating_factors: List[str] = field(default_factory=list)

    def interpret_value(
        self,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interpret a statistical value in terms of lived reality.

        Returns interpretation with concern level and narrative.
        """
        interpretation = {
            'indicator': self.name,
            'statistical_value': value,
            'dimension': self.reality_dimension.value,
        }

        # Determine concern level
        if self.alert_threshold and value >= self.alert_threshold:
            interpretation['concern_level'] = 'high'
            interpretation['narrative'] = f"High alert: {self.interpretation_template}"
        elif self.concern_threshold and value >= self.concern_threshold:
            interpretation['concern_level'] = 'moderate'
            interpretation['narrative'] = f"Concern: {self.interpretation_template}"
        else:
            interpretation['concern_level'] = 'low'
            interpretation['narrative'] = f"Within expected range: {self.interpretation_template}"

        # Apply contextual modifiers if context provided
        if context:
            amplifiers_present = [f for f in self.amplifying_factors if f in str(context)]
            mitigators_present = [f for f in self.mitigating_factors if f in str(context)]

            if amplifiers_present:
                interpretation['concern_amplifiers'] = amplifiers_present
            if mitigators_present:
                interpretation['concern_mitigators'] = mitigators_present

        return interpretation


# Pre-configured reality indicators for Chicago analysis
CHICAGO_REALITY_INDICATORS = {
    # Economic wellbeing indicators
    'income_growth_vs_cost': RealityIndicator(
        name='Income-Cost Divergence',
        statistical_source='median_income_growth',
        reality_dimension=RealityDimension.ECONOMIC_WELLBEING,
        interpretation_template="Income growth may not match rising cost of living, creating financial stress",
        concern_threshold=0.5,  # 50% divergence
        alert_threshold=1.0,    # 100% divergence
        amplifying_factors=['rising_property_taxes', 'healthcare_costs', 'childcare_costs'],
        mitigating_factors=['strong_employment', 'community_resources'],
    ),

    'employment_quality': RealityIndicator(
        name='Employment Quality Gap',
        statistical_source='unemployment_rate',
        reality_dimension=RealityDimension.ECONOMIC_WELLBEING,
        interpretation_template="Low unemployment may mask underemployment, gig work instability, or wage stagnation",
        concern_threshold=0.0,  # Even low unemployment can hide issues
        alert_threshold=0.05,   # 5% unemployment
        amplifying_factors=['gig_economy_presence', 'service_sector_dominance'],
        mitigating_factors=['union_presence', 'major_employers'],
    ),

    # Housing security indicators
    'rent_burden': RealityIndicator(
        name='Housing Cost Burden',
        statistical_source='rent_to_income_ratio',
        reality_dimension=RealityDimension.HOUSING_SECURITY,
        interpretation_template="High housing costs relative to income create financial precarity and displacement risk",
        concern_threshold=0.30,  # 30% of income on housing
        alert_threshold=0.50,    # 50% severely burdened
        amplifying_factors=['rising_property_taxes', 'limited_affordable_units'],
        mitigating_factors=['rent_stabilization', 'affordable_housing_development'],
    ),

    'development_pressure': RealityIndicator(
        name='Development Displacement Pressure',
        statistical_source='permit_growth_rate',
        reality_dimension=RealityDimension.DISPLACEMENT_RISK,
        interpretation_template="Rapid development can signal neighborhood change and potential displacement",
        concern_threshold=0.10,  # 10% permit growth
        alert_threshold=0.25,   # 25% rapid development
        amplifying_factors=['luxury_development', 'teardowns', 'condo_conversions'],
        mitigating_factors=['affordable_requirements', 'community_land_trusts'],
    ),

    # Community stability indicators
    'population_turnover': RealityIndicator(
        name='Community Stability Index',
        statistical_source='population_change_rate',
        reality_dimension=RealityDimension.COMMUNITY_STABILITY,
        interpretation_template="Population changes reflect community disruption or attraction patterns",
        concern_threshold=-0.05,  # 5% decline
        alert_threshold=-0.10,   # 10% decline
        amplifying_factors=['school_closures', 'business_closures', 'crime_increases'],
        mitigating_factors=['community_organizations', 'institutional_anchors'],
    ),

    'retail_accessibility': RealityIndicator(
        name='Service Desert Risk',
        statistical_source='retail_per_capita',
        reality_dimension=RealityDimension.SERVICE_ACCESSIBILITY,
        interpretation_template="Low retail presence indicates potential food/service deserts affecting daily life",
        concern_threshold=0.5,  # 50% below average
        alert_threshold=0.25,  # 75% below average
        amplifying_factors=['transit_limited', 'car_dependency', 'elderly_population'],
        mitigating_factors=['online_delivery', 'mobile_services'],
    ),
}


class InterpretationStrategy(ABC):
    """
    Abstract strategy for interpreting statistical results.

    Allows pluggable interpretation approaches without modifying core logic.
    """

    @abstractmethod
    def interpret(
        self,
        statistical_data: Dict[str, Any],
        context: CAGContext,
        indicators: Dict[str, RealityIndicator]
    ) -> Dict[str, Any]:
        """
        Interpret statistical data using this strategy.

        Args:
            statistical_data: Raw statistical outputs
            context: Contextual knowledge
            indicators: Reality indicators to apply

        Returns:
            Interpretation results
        """
        pass


class StatisticalInterpretation(InterpretationStrategy):
    """Pure statistical interpretation without contextual enhancement."""

    def interpret(
        self,
        statistical_data: Dict[str, Any],
        context: CAGContext,
        indicators: Dict[str, RealityIndicator]
    ) -> Dict[str, Any]:
        """Generate statistical summary without reality bridging."""
        return {
            'mode': 'statistical',
            'summary': statistical_data,
            'interpretations': [],
        }


class ContextualInterpretation(InterpretationStrategy):
    """Contextually-enhanced interpretation using domain knowledge."""

    def interpret(
        self,
        statistical_data: Dict[str, Any],
        context: CAGContext,
        indicators: Dict[str, RealityIndicator]
    ) -> Dict[str, Any]:
        """Generate interpretation with contextual enhancement."""
        interpretations = []

        for indicator_name, indicator in indicators.items():
            # Find matching statistical value
            value = self._find_statistical_value(
                statistical_data,
                indicator.statistical_source
            )

            if value is not None:
                interp = indicator.interpret_value(value, context.domain_context)
                interpretations.append(interp)

        return {
            'mode': 'contextual',
            'statistical_summary': statistical_data,
            'interpretations': interpretations,
            'context_used': list(context.context_types),
        }

    def _find_statistical_value(
        self,
        data: Dict[str, Any],
        source_name: str
    ) -> Optional[float]:
        """Find a statistical value by source name."""
        # Direct match
        if source_name in data:
            val = data[source_name]
            if isinstance(val, (int, float)):
                return float(val)

        # Search nested
        for key, value in data.items():
            if isinstance(value, dict):
                result = self._find_statistical_value(value, source_name)
                if result is not None:
                    return result

        return None


# --- Pure functions for narrative generation ---
# These can be tested independently and keep the strategy class thin


def generate_overview(
    interpretations: List[Dict[str, Any]],
    context: CAGContext,
) -> str:
    """
    Generate overview narrative from interpretations.

    Args:
        interpretations: List of interpretation results
        context: CAG context for geographic/temporal scope

    Returns:
        Overview narrative string
    """
    high_concerns = [i for i in interpretations if i.get('concern_level') == 'high']
    moderate_concerns = [i for i in interpretations if i.get('concern_level') == 'moderate']

    if high_concerns:
        return (
            f"Analysis of {context.geographic_scope} for {context.time_period} "
            f"reveals {len(high_concerns)} high-concern indicators requiring attention. "
            "These statistical patterns suggest potential community stress that may not "
            "be visible in aggregate economic metrics."
        )
    if moderate_concerns:
        return (
            f"Analysis of {context.geographic_scope} for {context.time_period} "
            f"shows {len(moderate_concerns)} areas of moderate concern. "
            "While aggregate statistics may appear stable, underlying patterns "
            "suggest monitoring is warranted."
        )
    return (
        f"Analysis of {context.geographic_scope} for {context.time_period} "
        "shows indicators within expected ranges. Community conditions appear "
        "stable based on available metrics."
    )


def generate_dimension_narratives(
    interpretations: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Generate narratives organized by reality dimension.

    Args:
        interpretations: List of interpretation results

    Returns:
        Dictionary mapping dimensions to narrative strings
    """
    narratives = {}

    # Group by dimension
    by_dimension: Dict[str, List[Dict[str, Any]]] = {}
    for interp in interpretations:
        dim = interp.get('dimension', 'other')
        if dim not in by_dimension:
            by_dimension[dim] = []
        by_dimension[dim].append(interp)

    # Generate narrative for each dimension
    for dimension, interps in by_dimension.items():
        concern_levels = [i.get('concern_level', 'low') for i in interps]
        if 'high' in concern_levels:
            narratives[dimension] = (
                f"Significant concerns in {dimension.replace('_', ' ')}. "
                "Statistical indicators suggest community stress that warrants "
                "deeper investigation and potential intervention."
            )
        elif 'moderate' in concern_levels:
            narratives[dimension] = (
                f"Moderate concerns in {dimension.replace('_', ' ')}. "
                "While not critical, trends should be monitored for "
                "potential escalation."
            )
        else:
            narratives[dimension] = (
                f"{dimension.replace('_', ' ').title()} indicators appear stable."
            )

    return narratives


def generate_community_impact(
    interpretations: List[Dict[str, Any]],
    context: CAGContext,
) -> str:
    """
    Generate community impact narrative bridging stats to lived reality.

    Args:
        interpretations: List of interpretation results
        context: CAG context with community information

    Returns:
        Community impact narrative string
    """
    community_context = context.domain_context.get('community', {})

    # Focus on high and moderate concerns
    concerns = [
        i for i in interpretations
        if i.get('concern_level') in ('high', 'moderate')
    ]

    if not concerns:
        return (
            "Based on available indicators, community conditions appear stable. "
            "However, aggregate statistics may not capture all aspects of lived experience. "
            "Continued monitoring and community engagement are recommended."
        )

    impact_points = []
    for concern in concerns:
        narrative = concern.get('narrative', '')
        if concern.get('concern_amplifiers'):
            narrative += f" (amplified by: {', '.join(concern['concern_amplifiers'])})"
        impact_points.append(narrative)

    community_orgs = community_context.get('active_organizations', [])
    if community_orgs:
        resources = f" Community organizations like {', '.join(community_orgs[:2])} may be valuable partners in addressing these concerns."
    else:
        resources = ""

    impact_text = "\n".join("• " + p for p in impact_points)
    return (
        "Community impact assessment:\n\n"
        f"{impact_text}\n\n"
        "These patterns suggest that statistical growth metrics may not fully "
        f"reflect the lived experience of community residents.{resources}"
    )


def generate_recommendations(
    interpretations: List[Dict[str, Any]],
) -> List[str]:
    """
    Generate actionable recommendations from interpretations.

    Args:
        interpretations: List of interpretation results

    Returns:
        List of recommendation strings
    """
    recommendations = []

    for interp in interpretations:
        if interp.get('concern_level') == 'high':
            dim = interp.get('dimension', 'unknown')
            if dim == 'economic_wellbeing':
                recommendations.append(
                    "Investigate income-cost divergence with detailed household analysis"
                )
            elif dim == 'housing_security':
                recommendations.append(
                    "Assess affordable housing inventory and preservation strategies"
                )
            elif dim == 'displacement_risk':
                recommendations.append(
                    "Implement early warning monitoring for displacement indicators"
                )
            elif dim == 'community_stability':
                recommendations.append(
                    "Engage community organizations to understand local dynamics"
                )

    # Add general recommendations
    if not recommendations:
        recommendations.append(
            "Continue monitoring key indicators for emerging patterns"
        )

    recommendations.append(
        "Validate statistical findings with community engagement and qualitative research"
    )

    return recommendations


class NarrativeInterpretation(InterpretationStrategy):
    """
    Full narrative interpretation for human-readable insights.

    This strategy produces the "lived reality bridge" output that
    translates statistics into community impact narratives.

    Uses pure functions for narrative generation to keep the class thin
    and make the narrative logic independently testable.
    """

    def interpret(
        self,
        statistical_data: Dict[str, Any],
        context: CAGContext,
        indicators: Dict[str, RealityIndicator]
    ) -> Dict[str, Any]:
        """Generate narrative interpretation with reality bridging."""
        # First get contextual interpretations
        contextual = ContextualInterpretation().interpret(
            statistical_data, context, indicators
        )
        interpretations = contextual.get('interpretations', [])

        # Generate narrative sections using pure functions
        narratives = {
            'overview': generate_overview(interpretations, context),
            'by_dimension': generate_dimension_narratives(interpretations),
            'community_impact': generate_community_impact(interpretations, context),
            'recommendations': generate_recommendations(interpretations),
        }

        return {
            'mode': 'narrative',
            'statistical_summary': statistical_data,
            'interpretations': interpretations,
            'narratives': narratives,
            'context_used': list(context.context_types),
        }


class RealityInterpreter(CAGComponent):
    """
    Bridges statistical analysis with lived community reality.

    Core implementation of the CAG principle that "policymakers see what
    people feel, not just what models measure."

    This component takes statistical outputs from existing models and
    enhances them with contextual interpretation that reflects actual
    community experience and impact.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        indicators: Optional[Dict[str, RealityIndicator]] = None
    ):
        """
        Initialize Reality Interpreter.

        Args:
            output_dir: Directory for interpretation outputs
            indicators: Custom reality indicators (uses Chicago defaults if None)
        """
        super().__init__("reality_interpreter", output_dir)

        # Load indicators
        self.indicators = indicators or CHICAGO_REALITY_INDICATORS.copy()

        # Interpretation strategies
        self.strategies = {
            InterpretationMode.STATISTICAL: StatisticalInterpretation(),
            InterpretationMode.CONTEXTUAL: ContextualInterpretation(),
            InterpretationMode.NARRATIVE: NarrativeInterpretation(),
        }

        # Default mode
        self.default_mode = InterpretationMode.NARRATIVE

    def process(
        self,
        data: Dict[str, Any],
        context: Optional[CAGContext] = None
    ) -> CAGResult:
        """
        Process statistical data with reality interpretation.

        Args:
            data: Statistical results from pipeline models
            context: CAG context for enhancement

        Returns:
            CAGResult with reality-bridged interpretations
        """
        self._update_run_stats()

        # Use provided context or create minimal
        if context is None:
            context = CAGContext(
                analysis_type="statistical",
                geographic_scope="Chicago",
                time_period="current",
            )

        # Run interpretation
        interpretation = self.interpret(
            data,
            context,
            mode=self.default_mode
        )

        # Create result
        result = CAGResult(
            analysis_id=f"interpretation_{self._run_count}",
            statistical_results=data,
            context_used=context,
            interpretation_mode=self.default_mode,
        )

        # Extract key outputs
        if 'narratives' in interpretation:
            result.contextual_interpretation = interpretation['narratives'].get('overview', '')
            result.lived_reality_bridge = interpretation['narratives'].get('community_impact', '')
            result.community_impact_narrative = interpretation['narratives'].get('community_impact', '')
            result.policy_implications = interpretation['narratives'].get('recommendations', [])

        # Add interpretations as discovered patterns
        for interp in interpretation.get('interpretations', []):
            if interp.get('concern_level') in ('high', 'moderate'):
                result.anomalies_identified.append({
                    'type': interp['indicator'],
                    'location': context.geographic_scope,
                    'description': interp.get('narrative', ''),
                    'concern_level': interp.get('concern_level'),
                })

        self.logger.info(f"Generated reality interpretation with {len(result.anomalies_identified)} concerns")
        return result

    def interpret(
        self,
        statistical_data: Dict[str, Any],
        context: CAGContext,
        mode: InterpretationMode = InterpretationMode.NARRATIVE
    ) -> Dict[str, Any]:
        """
        Interpret statistical data with specified mode.

        Args:
            statistical_data: Raw statistical outputs
            context: Contextual knowledge
            mode: Interpretation mode to use

        Returns:
            Interpretation results
        """
        strategy = self.strategies.get(mode, self.strategies[InterpretationMode.STATISTICAL])
        return strategy.interpret(statistical_data, context, self.indicators)

    def interpret_model_results(
        self,
        model_name: str,
        model_results: Dict[str, Any],
        context: CAGContext
    ) -> CAGResult:
        """
        Interpret results from a specific model.

        Provides model-specific interpretation logic.

        Args:
            model_name: Name of the model (e.g., 'multifamily_growth')
            model_results: Results from that model
            context: CAG context

        Returns:
            CAGResult with model-specific interpretation
        """
        self._update_run_stats()

        # Model-specific interpretation
        interpreters = {
            'multifamily_growth': self._interpret_multifamily_growth,
            'retail_gap': self._interpret_retail_gap,
            'population_prediction': self._interpret_population_prediction,
            'income_distribution': self._interpret_income_distribution,
            'zoning_impact': self._interpret_zoning_impact,
        }

        interpreter = interpreters.get(model_name, self._interpret_generic)
        return interpreter(model_results, context)

    def _interpret_multifamily_growth(
        self,
        results: Dict[str, Any],
        context: CAGContext
    ) -> CAGResult:
        """Interpret multifamily growth model results."""
        result = CAGResult(
            analysis_id=f"multifamily_{self._run_count}",
            statistical_results=results,
            context_used=context,
        )

        # Extract key metrics
        top_zips = results.get('top_growth_zips', [])
        growth_scores = results.get('growth_scores', {})

        # Bridge to reality
        high_growth_zips = [z for z, s in growth_scores.items() if s > 0.7]

        if high_growth_zips:
            # Check for gentrification risk
            gentrifying = []
            for zip_code in high_growth_zips:
                profile = context.community_profiles.get(zip_code, {})
                if profile:
                    housing = profile.get('housing', {})
                    if 'displacement' in str(housing.get('affordability_concerns', [])).lower():
                        gentrifying.append(zip_code)

            if gentrifying:
                result.lived_reality_bridge = (
                    f"High multifamily development activity in {', '.join(high_growth_zips)} "
                    f"indicates strong market interest. However, {len(gentrifying)} area(s) "
                    f"({', '.join(gentrifying)}) show potential displacement risk based on "
                    f"community affordability concerns. Statistical growth metrics should be "
                    f"balanced against community stability considerations."
                )
                result.anomalies_identified.append({
                    'type': 'displacement_risk',
                    'location': ', '.join(gentrifying),
                    'description': 'High development activity in areas with existing affordability concerns',
                })
            else:
                result.lived_reality_bridge = (
                    f"Multifamily development growth in {', '.join(high_growth_zips)} "
                    f"suggests healthy market activity. Community profiles indicate "
                    f"development pressure is within manageable ranges."
                )
        else:
            result.lived_reality_bridge = (
                "Moderate multifamily development activity across analyzed ZIP codes. "
                "No significant displacement pressure signals detected."
            )

        return result

    def _interpret_retail_gap(
        self,
        results: Dict[str, Any],
        context: CAGContext
    ) -> CAGResult:
        """Interpret retail gap model results."""
        result = CAGResult(
            analysis_id=f"retail_gap_{self._run_count}",
            statistical_results=results,
            context_used=context,
        )

        # Extract gap analysis
        opportunity_zones = results.get('opportunity_zones', [])
        gap_scores = results.get('gap_scores', {})

        # Bridge to reality
        high_gap_areas = [z for z, s in gap_scores.items() if s > 0.6]

        if high_gap_areas:
            # Check for service desert implications
            service_concerns = []
            for zip_code in high_gap_areas:
                profile = context.community_profiles.get(zip_code, {})
                if profile:
                    challenges = profile.get('economic', {}).get('challenges', [])
                    if any('desert' in c.lower() or 'limited' in c.lower() for c in challenges):
                        service_concerns.append(zip_code)

            if service_concerns:
                result.lived_reality_bridge = (
                    f"Statistical retail gap analysis identifies opportunity in "
                    f"{len(opportunity_zones)} zones. However, {len(service_concerns)} area(s) "
                    f"({', '.join(service_concerns)}) have existing service accessibility "
                    f"concerns. High gap scores in these areas represent both investment "
                    f"opportunity AND community need - the 'gap' reflects real service "
                    f"deserts affecting daily life, not just market opportunity."
                )
                result.policy_implications = [
                    "Prioritize essential retail (grocery, pharmacy) in high-gap areas",
                    "Consider community-serving retail requirements for new development",
                    "Investigate transportation barriers contributing to retail gaps",
                ]
            else:
                result.lived_reality_bridge = (
                    f"Retail gap analysis reveals {len(opportunity_zones)} areas with "
                    f"potential for retail development. Gap scores reflect market "
                    f"opportunity rather than acute service deserts."
                )

        return result

    def _interpret_population_prediction(
        self,
        results: Dict[str, Any],
        context: CAGContext
    ) -> CAGResult:
        """Interpret population prediction model results."""
        result = CAGResult(
            analysis_id=f"population_{self._run_count}",
            statistical_results=results,
            context_used=context,
        )

        forecasts = results.get('forecasts', {})
        declining_areas = []
        growing_areas = []

        for zip_code, forecast in forecasts.items():
            if isinstance(forecast, dict):
                trend = forecast.get('trend', 0)
            else:
                trend = forecast

            if trend < -0.05:
                declining_areas.append(zip_code)
            elif trend > 0.05:
                growing_areas.append(zip_code)

        narrative_parts = []

        if declining_areas:
            # Check community context for declining areas
            concerns = []
            for zip_code in declining_areas:
                profile = context.community_profiles.get(zip_code, {})
                if profile:
                    community = profile.get('community', {})
                    if community.get('concerns'):
                        concerns.extend(community['concerns'])

            narrative_parts.append(
                f"Population decline projected in {len(declining_areas)} area(s) "
                f"({', '.join(declining_areas[:3])}{'...' if len(declining_areas) > 3 else ''}). "
                f"Statistical forecasts should be contextualized with community factors: "
                f"{', '.join(set(concerns[:3]))}." if concerns else
                f"Population decline projected in {len(declining_areas)} areas."
            )

            result.anomalies_identified.append({
                'type': 'population_decline',
                'location': ', '.join(declining_areas),
                'description': 'Areas with projected population decline requiring attention',
            })

        if growing_areas:
            narrative_parts.append(
                f"Growth projected in {len(growing_areas)} area(s). "
                f"Rapid growth may strain infrastructure and affect affordability."
            )

        result.lived_reality_bridge = " ".join(narrative_parts) or "Population projections within normal ranges."

        return result

    def _interpret_income_distribution(
        self,
        results: Dict[str, Any],
        context: CAGContext
    ) -> CAGResult:
        """Interpret income distribution model results."""
        result = CAGResult(
            analysis_id=f"income_{self._run_count}",
            statistical_results=results,
            context_used=context,
        )

        gentrification_risk = results.get('gentrification_risk', {})
        displacement_risk = results.get('displacement_risk', {})

        high_risk_areas = [
            z for z, r in gentrification_risk.items()
            if r in ('high', 'very_high') or (isinstance(r, (int, float)) and r > 0.7)
        ]

        if high_risk_areas:
            result.lived_reality_bridge = (
                f"Income distribution analysis identifies {len(high_risk_areas)} area(s) "
                f"with elevated gentrification/displacement risk. While rising incomes "
                f"appear positive statistically, they often signal demographic transition "
                f"that displaces existing residents. The 'GDP growth vs felt recession' "
                f"dynamic applies here: aggregate income gains may mask loss of "
                f"community stability and displacement of lower-income households."
            )

            result.policy_implications = [
                "Implement displacement monitoring and early warning systems",
                "Consider right-to-return policies for displaced residents",
                "Prioritize affordable housing preservation in high-risk areas",
                "Engage existing community organizations before major investment decisions",
            ]

            for zip_code in high_risk_areas:
                result.anomalies_identified.append({
                    'type': 'gentrification_risk',
                    'location': zip_code,
                    'description': f"High gentrification/displacement risk in {zip_code}",
                })
        else:
            result.lived_reality_bridge = (
                "Income distribution analysis shows relatively stable patterns. "
                "No immediate gentrification pressure signals detected."
            )

        return result

    def _interpret_zoning_impact(
        self,
        results: Dict[str, Any],
        context: CAGContext
    ) -> CAGResult:
        """Interpret zoning impact model results."""
        result = CAGResult(
            analysis_id=f"zoning_{self._run_count}",
            statistical_results=results,
            context_used=context,
        )

        opportunity_zones = results.get('opportunity_zones', [])
        constraints = results.get('constraints', {})

        result.lived_reality_bridge = (
            f"Zoning analysis identifies {len(opportunity_zones)} potential "
            f"development opportunity zones. Zoning constraints can both "
            f"enable and restrict development - high constraint areas may "
            f"preserve community character but limit housing supply, while "
            f"low constraint areas may see rapid change affecting residents."
        )

        return result

    def _interpret_generic(
        self,
        results: Dict[str, Any],
        context: CAGContext
    ) -> CAGResult:
        """Generic interpretation for unknown models."""
        result = CAGResult(
            analysis_id=f"generic_{self._run_count}",
            statistical_results=results,
            context_used=context,
        )

        result.lived_reality_bridge = (
            "Statistical analysis complete. Consider how these metrics "
            "manifest in actual community experience and engage local "
            "stakeholders to validate findings."
        )

        return result

    def add_indicator(self, name: str, indicator: RealityIndicator) -> None:
        """Add a custom reality indicator."""
        self.indicators[name] = indicator
        self.logger.info(f"Added indicator: {name}")

    def set_interpretation_mode(self, mode: InterpretationMode) -> None:
        """Set the default interpretation mode."""
        self.default_mode = mode
        self.logger.info(f"Default mode set to: {mode.value}")

    def add_interpretation_strategy(
        self,
        mode: InterpretationMode,
        strategy: InterpretationStrategy
    ) -> None:
        """Add a custom interpretation strategy."""
        self.strategies[mode] = strategy
        self.logger.info(f"Added strategy for mode: {mode.value}")
