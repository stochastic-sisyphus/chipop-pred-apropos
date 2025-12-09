"""
Context Builder for Chicago Urban Analytics.

Constructs enriched context at runtime by combining:
- Domain knowledge about Chicago neighborhoods
- Historical development patterns
- Community-specific characteristics
- Policy and regulatory considerations

This implements the CAG principle of building "extended unique context
providing additional instructions, background knowledge, and contextual
information" for each analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
import logging
import json

from .base import (
    CAGComponent,
    CAGContext,
    CAGResult,
    ContextType,
    InterpretationMode,
)

logger = logging.getLogger(__name__)


@dataclass
class ChicagoContextProfile:
    """
    Chicago-specific context profile for a geographic area.

    Encapsulates the "lived reality" knowledge about Chicago
    neighborhoods that statistical models cannot capture.
    """
    zip_code: str
    community_area: str
    neighborhood_names: List[str] = field(default_factory=list)

    # Demographic character
    demographic_character: str = ""
    population_trends: str = ""
    migration_patterns: str = ""

    # Economic context
    economic_character: str = ""
    major_employers: List[str] = field(default_factory=list)
    commercial_corridors: List[str] = field(default_factory=list)
    economic_challenges: List[str] = field(default_factory=list)

    # Housing context
    housing_stock_character: str = ""
    development_pressures: str = ""
    affordability_concerns: List[str] = field(default_factory=list)

    # Community dynamics
    community_organizations: List[str] = field(default_factory=list)
    cultural_anchors: List[str] = field(default_factory=list)
    community_concerns: List[str] = field(default_factory=list)

    # Historical context
    historical_significance: str = ""
    past_development_waves: List[str] = field(default_factory=list)
    gentrification_history: str = ""

    # Policy context
    tif_districts: List[str] = field(default_factory=list)
    opportunity_zones: bool = False
    special_zoning: List[str] = field(default_factory=list)

    def to_context_dict(self) -> Dict[str, Any]:
        """Convert to context dictionary for CAGContext."""
        return {
            'geography': {
                'zip_code': self.zip_code,
                'community_area': self.community_area,
                'neighborhoods': self.neighborhood_names,
            },
            'demographic': {
                'character': self.demographic_character,
                'population_trends': self.population_trends,
                'migration_patterns': self.migration_patterns,
            },
            'economic': {
                'character': self.economic_character,
                'major_employers': self.major_employers,
                'commercial_corridors': self.commercial_corridors,
                'challenges': self.economic_challenges,
            },
            'housing': {
                'stock_character': self.housing_stock_character,
                'development_pressures': self.development_pressures,
                'affordability_concerns': self.affordability_concerns,
            },
            'community': {
                'organizations': self.community_organizations,
                'cultural_anchors': self.cultural_anchors,
                'concerns': self.community_concerns,
            },
            'historical': {
                'significance': self.historical_significance,
                'development_waves': self.past_development_waves,
                'gentrification_history': self.gentrification_history,
            },
            'policy': {
                'tif_districts': self.tif_districts,
                'opportunity_zone': self.opportunity_zones,
                'special_zoning': self.special_zoning,
            },
        }


def _load_chicago_profiles() -> Dict[str, ChicagoContextProfile]:
    """
    Load Chicago neighborhood profiles from JSON data file.

    Returns:
        Dictionary mapping ZIP codes to ChicagoContextProfile instances
    """
    data_path = Path(__file__).parent / "data" / "chicago_profiles.json"

    if not data_path.exists():
        logger.warning(f"Chicago profiles not found at {data_path}, using empty profiles")
        return {}

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        profiles = {}
        for zip_code, profile_dict in raw.items():
            profiles[zip_code] = ChicagoContextProfile(**profile_dict)
        return profiles

    except Exception as e:
        logger.error(f"Failed to load Chicago profiles: {e}")
        return {}


# Load profiles at module import time
CHICAGO_CONTEXT_PROFILES: Dict[str, ChicagoContextProfile] = _load_chicago_profiles()


class ContextBuilder(CAGComponent):
    """
    Builds rich contextual knowledge for LLM-guided analysis.

    Implements the CAG principle of constructing "extended unique context"
    at runtime that combines:
    - Domain knowledge about Chicago neighborhoods
    - Historical development patterns
    - Statistical summaries from collected data
    - Policy and regulatory considerations

    This context bridges raw statistical outputs with the interpretive
    framework needed for meaningful community impact analysis.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize ContextBuilder.

        Args:
            output_dir: Directory for context outputs
        """
        super().__init__("context_builder", output_dir)

        # Load Chicago context profiles
        self.chicago_profiles = CHICAGO_CONTEXT_PROFILES.copy()

        # Context type handlers
        self._context_builders = {
            ContextType.DEMOGRAPHIC: self._build_demographic_context,
            ContextType.ECONOMIC: self._build_economic_context,
            ContextType.HOUSING: self._build_housing_context,
            ContextType.COMMUNITY: self._build_community_context,
            ContextType.HISTORICAL: self._build_historical_context,
            ContextType.POLICY: self._build_policy_context,
        }

    # --- Helper methods for iteration and aggregation ---

    def _iter_profiles(
        self,
        zip_codes: List[str]
    ) -> Generator[Tuple[str, ChicagoContextProfile], None, None]:
        """
        Iterate over ZIP codes that have known profiles.

        Args:
            zip_codes: List of ZIP codes to iterate

        Yields:
            Tuples of (zip_code, profile) for ZIPs with known profiles
        """
        for z in zip_codes:
            profile = self.chicago_profiles.get(z)
            if profile:
                yield z, profile

    def _unique(self, items: List[Any]) -> List[Any]:
        """Return unique items preserving order where possible."""
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _ensure_context_type(self, context: CAGContext, ct: ContextType) -> None:
        """Ensure context type is in the context's type list."""
        if ct not in context.context_types:
            context.context_types.append(ct)

    # --- Main processing methods ---

    def process(
        self,
        data: Dict[str, Any],
        context: Optional[CAGContext] = None
    ) -> CAGResult:
        """
        Build enriched context from data and existing context.

        Args:
            data: Pipeline data including collected data and model results
            context: Optional existing context to enhance

        Returns:
            CAGResult containing the built context
        """
        self._update_run_stats()

        # Determine geographic scope
        zip_codes = self._extract_zip_codes(data)
        time_period = self._extract_time_period(data)
        analysis_type = data.get('analysis_type', 'comprehensive')

        # Build new context or enhance existing
        if context is None:
            context = CAGContext(
                analysis_type=analysis_type,
                geographic_scope=f"Chicago ZIP codes: {', '.join(zip_codes[:5])}{'...' if len(zip_codes) > 5 else ''}",
                time_period=time_period,
            )

        # Build each context type
        context = self._build_all_contexts(context, data, zip_codes)

        # Add statistical summary from data
        context.statistical_summary = self._build_statistical_summary(data)

        # Add known anomalies and policy considerations
        context.known_anomalies = self._identify_known_anomalies(zip_codes)
        context.policy_considerations = self._compile_policy_considerations(zip_codes)

        # Create result
        result = CAGResult(
            analysis_id=f"context_{self._run_count}",
            context_used=context,
        )

        result.contextual_interpretation = f"Built context for {len(zip_codes)} ZIP codes"
        result.discovered_patterns = self._identify_geographic_patterns(zip_codes)

        self.logger.info(f"Built context for {len(zip_codes)} ZIP codes")
        return result

    def build_for_zip_codes(
        self,
        zip_codes: List[str],
        analysis_type: str = "comprehensive",
        time_period: str = "2020-2024",
        context_types: Optional[List[ContextType]] = None
    ) -> CAGContext:
        """
        Build context specifically for a set of ZIP codes.

        Args:
            zip_codes: List of ZIP codes to build context for
            analysis_type: Type of analysis being performed
            time_period: Time period for analysis
            context_types: Specific context types to include (all if None)

        Returns:
            CAGContext with neighborhood knowledge
        """
        context = CAGContext(
            analysis_type=analysis_type,
            geographic_scope=f"Chicago: {', '.join(zip_codes[:3])}{'...' if len(zip_codes) > 3 else ''}",
            time_period=time_period,
        )

        # Use all context types if not specified
        if context_types is None:
            context_types = list(ContextType)

        # Build community profiles
        for zip_code, profile in self._iter_profiles(zip_codes):
            context.community_profiles[zip_code] = profile.to_context_dict()

        # Build specified context types
        for ct in context_types:
            if ct in self._context_builders:
                self._context_builders[ct](context, zip_codes)

        context.context_types = context_types

        return context

    def add_custom_profile(self, profile: ChicagoContextProfile) -> None:
        """
        Add a custom context profile for a ZIP code.

        Allows extending the knowledge base with new profiles.
        """
        self.chicago_profiles[profile.zip_code] = profile
        self.logger.info(f"Added custom profile for {profile.zip_code}")

    def get_profile(self, zip_code: str) -> Optional[ChicagoContextProfile]:
        """Get the context profile for a ZIP code."""
        return self.chicago_profiles.get(zip_code)

    def _build_all_contexts(
        self,
        context: CAGContext,
        data: Dict[str, Any],
        zip_codes: List[str]
    ) -> CAGContext:
        """Build all context types."""
        for context_type, builder in self._context_builders.items():
            try:
                builder(context, zip_codes, data)
            except Exception as e:
                self.logger.warning(f"Error building {context_type.value} context: {e}")
        return context

    # --- Context type builders (refactored to use helpers) ---

    def _build_demographic_context(
        self,
        context: CAGContext,
        zip_codes: List[str],
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build demographic context from profiles and data."""
        demographic_patterns = {}
        gentrifying = []
        declining = []

        for z, profile in self._iter_profiles(zip_codes):
            demographic_patterns[z] = {
                'character': profile.demographic_character,
                'trends': profile.population_trends,
                'migration': profile.migration_patterns,
            }

            if ('gentrification' in profile.gentrification_history.lower()
                    and 'not' not in profile.gentrification_history.lower()):
                gentrifying.append(z)

            if 'declin' in profile.population_trends.lower():
                declining.append(z)

        context.domain_context['demographic'] = {
            'zip_patterns': demographic_patterns,
            'gentrifying_areas': gentrifying,
            'declining_areas': declining,
            'total_zips_analyzed': len(zip_codes),
        }

        self._ensure_context_type(context, ContextType.DEMOGRAPHIC)

    def _build_economic_context(
        self,
        context: CAGContext,
        zip_codes: List[str],
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build economic context from profiles and data."""
        economic_patterns = {}
        all_employers = []
        all_corridors = []
        all_challenges = []

        for z, profile in self._iter_profiles(zip_codes):
            economic_patterns[z] = {
                'character': profile.economic_character,
                'employers': profile.major_employers,
                'corridors': profile.commercial_corridors,
                'challenges': profile.economic_challenges,
            }
            all_employers.extend(profile.major_employers)
            all_corridors.extend(profile.commercial_corridors)
            all_challenges.extend(profile.economic_challenges)

        context.domain_context['economic'] = {
            'zip_patterns': economic_patterns,
            'major_employers': self._unique(all_employers),
            'commercial_corridors': self._unique(all_corridors),
            'common_challenges': self._unique(all_challenges),
        }

        self._ensure_context_type(context, ContextType.ECONOMIC)

    def _build_housing_context(
        self,
        context: CAGContext,
        zip_codes: List[str],
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build housing context from profiles and data."""
        housing_patterns = {}
        high_pressure = []
        low_pressure = []
        affordability_issues = []

        for z, profile in self._iter_profiles(zip_codes):
            housing_patterns[z] = {
                'stock': profile.housing_stock_character,
                'pressures': profile.development_pressures,
                'affordability': profile.affordability_concerns,
            }

            if 'high' in profile.development_pressures.lower():
                high_pressure.append(z)
            elif 'low' in profile.development_pressures.lower():
                low_pressure.append(z)

            affordability_issues.extend(profile.affordability_concerns)

        context.domain_context['housing'] = {
            'zip_patterns': housing_patterns,
            'high_pressure_zips': high_pressure,
            'low_pressure_zips': low_pressure,
            'affordability_concerns': self._unique(affordability_issues),
        }

        self._ensure_context_type(context, ContextType.HOUSING)

    def _build_community_context(
        self,
        context: CAGContext,
        zip_codes: List[str],
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build community context from profiles."""
        community_patterns = {}
        all_organizations = []
        all_concerns = []

        for z, profile in self._iter_profiles(zip_codes):
            community_patterns[z] = {
                'organizations': profile.community_organizations,
                'anchors': profile.cultural_anchors,
                'concerns': profile.community_concerns,
            }
            all_organizations.extend(profile.community_organizations)
            all_concerns.extend(profile.community_concerns)

        context.domain_context['community'] = {
            'zip_patterns': community_patterns,
            'active_organizations': self._unique(all_organizations),
            'common_concerns': self._unique(all_concerns),
        }

        self._ensure_context_type(context, ContextType.COMMUNITY)

    def _build_historical_context(
        self,
        context: CAGContext,
        zip_codes: List[str],
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build historical context from profiles."""
        historical_patterns = {}
        former_industrial = []
        gentrification_cases = []

        for z, profile in self._iter_profiles(zip_codes):
            historical_patterns[z] = {
                'significance': profile.historical_significance,
                'development_waves': profile.past_development_waves,
                'gentrification': profile.gentrification_history,
            }

            if any('industrial' in w.lower() for w in profile.past_development_waves):
                former_industrial.append(z)

            if 'gentrification' in profile.gentrification_history.lower():
                gentrification_cases.append(z)

        context.historical_patterns = {
            'zip_patterns': historical_patterns,
            'former_industrial': former_industrial,
            'gentrification_case_studies': gentrification_cases,
        }

        self._ensure_context_type(context, ContextType.HISTORICAL)

    def _build_policy_context(
        self,
        context: CAGContext,
        zip_codes: List[str],
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build policy context from profiles."""
        policy_patterns = {}
        opportunity_zones = []
        tif_districts = []

        for z, profile in self._iter_profiles(zip_codes):
            policy_patterns[z] = {
                'tif': profile.tif_districts,
                'opportunity_zone': profile.opportunity_zones,
                'special_zoning': profile.special_zoning,
            }

            if profile.opportunity_zones:
                opportunity_zones.append(z)
            tif_districts.extend(profile.tif_districts)

        context.domain_context['policy'] = {
            'zip_patterns': policy_patterns,
            'opportunity_zones': opportunity_zones,
            'tif_districts': self._unique(tif_districts),
        }

        self._ensure_context_type(context, ContextType.POLICY)

    # --- Utility methods ---

    def _extract_zip_codes(self, data: Dict[str, Any]) -> List[str]:
        """Extract ZIP codes from pipeline data."""
        zip_codes = set()

        # Check various data locations
        if 'census_data' in data:
            census = data['census_data']
            if hasattr(census, 'columns') and 'zip_code' in census.columns:
                zip_codes.update(census['zip_code'].astype(str).unique())

        if 'zip_codes' in data:
            zip_codes.update(str(z) for z in data['zip_codes'])

        if 'model_results' in data:
            for model_name, results in data['model_results'].items():
                if isinstance(results, dict) and 'zip_codes' in results:
                    zip_codes.update(str(z) for z in results['zip_codes'])

        # Default to all known profiles if none extracted
        if not zip_codes:
            zip_codes = set(self.chicago_profiles.keys())

        return sorted(list(zip_codes))

    def _extract_time_period(self, data: Dict[str, Any]) -> str:
        """Extract time period from pipeline data."""
        if 'time_period' in data:
            return data['time_period']
        if 'start_year' in data and 'end_year' in data:
            return f"{data['start_year']}-{data['end_year']}"
        return "2020-2024"

    def _build_statistical_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build statistical summary from pipeline data."""
        summary = {}

        if 'model_results' in data:
            for model_name, results in data['model_results'].items():
                if isinstance(results, dict):
                    summary[model_name] = {
                        k: v for k, v in results.items()
                        if isinstance(v, (int, float, str, list)) and len(str(v)) < 500
                    }

        return summary

    def _identify_known_anomalies(self, zip_codes: List[str]) -> List[str]:
        """Identify known anomalies for the geographic scope."""
        anomalies = []

        for z, profile in self._iter_profiles(zip_codes):
            # University areas have unique demographics
            if 'university' in profile.demographic_character.lower():
                anomalies.append(f"{z}: University presence affects demographic patterns")

            # Industrial transition areas
            if 'industrial' in profile.economic_character.lower():
                anomalies.append(f"{z}: Industrial transition may affect economic indicators")

            # Active gentrification
            if 'active' in profile.gentrification_history.lower():
                anomalies.append(f"{z}: Active gentrification creating rapid change")

        return anomalies

    def _compile_policy_considerations(self, zip_codes: List[str]) -> List[str]:
        """Compile policy considerations for the geographic scope."""
        considerations = []

        oz_count = sum(1 for _, p in self._iter_profiles(zip_codes) if p.opportunity_zones)
        if oz_count > 0:
            considerations.append(f"{oz_count} ZIP codes are Opportunity Zones - investment incentives may accelerate development")

        # INVEST South/West
        isw_zips = [
            z for z, p in self._iter_profiles(zip_codes)
            if any('invest' in sz.lower() for sz in p.special_zoning)
        ]
        if isw_zips:
            considerations.append(f"INVEST South/West corridors in {', '.join(isw_zips)} - major public investment planned")

        return considerations

    def _identify_geographic_patterns(self, zip_codes: List[str]) -> List[Dict[str, Any]]:
        """Identify geographic patterns across ZIP codes."""
        patterns = []

        # Group by gentrification status
        gentrifying = [
            z for z, p in self._iter_profiles(zip_codes)
            if 'gentrification' in p.gentrification_history.lower()
            and 'not' not in p.gentrification_history.lower()
        ]
        if gentrifying:
            patterns.append({
                'name': 'Gentrification Pressure',
                'description': f'Active gentrification in {len(gentrifying)} areas: {", ".join(gentrifying)}',
                'zip_codes': gentrifying,
                'pattern_type': 'demographic_shift',
            })

        # Group by opportunity zone status
        opportunity = [z for z, p in self._iter_profiles(zip_codes) if p.opportunity_zones]
        if opportunity:
            patterns.append({
                'name': 'Opportunity Zone Concentration',
                'description': f'{len(opportunity)} Opportunity Zones may see accelerated investment',
                'zip_codes': opportunity,
                'pattern_type': 'policy_incentive',
            })

        return patterns
