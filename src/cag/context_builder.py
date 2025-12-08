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
from typing import Any, Dict, List, Optional, Set, Tuple
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


# Chicago neighborhood knowledge base
# This embodies the "lived reality" that bridges statistics to community experience
CHICAGO_CONTEXT_PROFILES: Dict[str, ChicagoContextProfile] = {
    # South Side - Investment Opportunity Areas
    '60615': ChicagoContextProfile(
        zip_code='60615',
        community_area='Hyde Park / Kenwood',
        neighborhood_names=['Hyde Park', 'Kenwood', 'North Kenwood'],
        demographic_character='Diverse, highly educated community anchored by University of Chicago',
        population_trends='Stable with recent growth due to university expansion',
        migration_patterns='Attracts academics, students, and young professionals',
        economic_character='Institution-driven economy with growing retail',
        major_employers=['University of Chicago', 'UChicago Medicine'],
        commercial_corridors=['53rd Street', 'Hyde Park Boulevard'],
        economic_challenges=['Retail gaps in some corridors', 'Income inequality within area'],
        housing_stock_character='Mix of historic mansions, mid-rises, and new construction',
        development_pressures='University-driven development, Obama Presidential Center impact',
        affordability_concerns=['Rising rents near campus', 'Displacement of long-term residents'],
        community_organizations=['Hyde Park Neighborhood Club', 'Kenwood Oakland Community Organization'],
        cultural_anchors=['Museum of Science and Industry', 'Smart Museum'],
        community_concerns=['Balancing development with affordability', 'Preserving historic character'],
        historical_significance='Historic African American wealth, intellectual heritage',
        past_development_waves=['1893 World\'s Fair', 'Urban renewal 1950s-60s', 'Recent university expansion'],
        gentrification_history='Ongoing university-adjacent gentrification',
        tif_districts=['Lake Park/Kenwood', 'Harper Court'],
        opportunity_zones=True,
        special_zoning=['Planned Development near university'],
    ),

    '60617': ChicagoContextProfile(
        zip_code='60617',
        community_area='South Chicago / East Side',
        neighborhood_names=['South Chicago', 'East Side', 'Calumet Heights', 'Avalon Park'],
        demographic_character='Working-class, historically industrial, predominantly Black and Hispanic',
        population_trends='Declining due to industrial job losses',
        migration_patterns='Outmigration of working families seeking employment',
        economic_character='Post-industrial transition, seeking new economic base',
        major_employers=['Remaining steel industry', 'Healthcare facilities'],
        commercial_corridors=['Commercial Avenue', '79th Street'],
        economic_challenges=['High unemployment', 'Commercial vacancy', 'Limited investment'],
        housing_stock_character='Single-family homes, some multifamily, aging stock',
        development_pressures='Low pressure, opportunity for strategic investment',
        affordability_concerns=['Disinvestment more than displacement', 'Property value decline'],
        community_organizations=['South Chicago Neighborhood Network'],
        cultural_anchors=['Southeast Chicago Historical Museum'],
        community_concerns=['Job creation', 'Crime reduction', 'Property maintenance'],
        historical_significance='Steel industry heritage, immigrant gateway',
        past_development_waves=['Steel mill expansion 1900s-1970s', 'Deindustrialization 1980s-present'],
        gentrification_history='Minimal gentrification pressure',
        tif_districts=['Commercial Avenue'],
        opportunity_zones=True,
        special_zoning=['Industrial corridors'],
    ),

    '60619': ChicagoContextProfile(
        zip_code='60619',
        community_area='Chatham / Greater Grand Crossing',
        neighborhood_names=['Chatham', 'Greater Grand Crossing', 'Avalon Park'],
        demographic_character='Historic Black middle-class community, strong institutions',
        population_trends='Population decline but community resilience',
        migration_patterns='Some outmigration of middle-class families',
        economic_character='Neighborhood commercial, professional services',
        major_employers=['Local businesses', 'Healthcare', 'Education'],
        commercial_corridors=['79th Street', '87th Street'],
        economic_challenges=['Commercial corridor revitalization needed', 'Business retention'],
        housing_stock_character='Well-maintained single-family homes, bungalows',
        development_pressures='Moderate, focused on retail and community development',
        affordability_concerns=['Maintaining homeownership', 'Property tax burden'],
        community_organizations=['Greater Chatham Initiative', 'RAGE (Residents Association of Greater Englewood)'],
        cultural_anchors=['DuSable Museum connection', 'Historic churches'],
        community_concerns=['Economic development', 'Youth opportunities', 'Safety'],
        historical_significance='Black middle-class achievement, institutional strength',
        past_development_waves=['Black migration destination 1950s-70s', 'Recent renewal efforts'],
        gentrification_history='Not currently experiencing gentrification',
        opportunity_zones=True,
        special_zoning=[],
    ),

    '60620': ChicagoContextProfile(
        zip_code='60620',
        community_area='Auburn Gresham',
        neighborhood_names=['Auburn Gresham', 'Washington Heights'],
        demographic_character='Predominantly Black, working and middle class, strong faith community',
        population_trends='Declining but stable core population',
        migration_patterns='Some outmigration, but strong community attachment',
        economic_character='Neighborhood commercial, healthcare presence',
        major_employers=['Little Company of Mary Hospital', 'Local businesses'],
        commercial_corridors=['79th Street', 'Halsted Street'],
        economic_challenges=['Commercial vacancy', 'Job access', 'Food desert concerns'],
        housing_stock_character='Chicago bungalows, well-maintained residential',
        development_pressures='Low, but strategic opportunities exist',
        affordability_concerns=['Property tax burden', 'Maintaining homeownership'],
        community_organizations=['Greater Auburn Gresham Development Corporation'],
        cultural_anchors=['Historic churches', 'St. Sabina Church'],
        community_concerns=['Youth violence', 'Economic opportunity', 'Property values'],
        historical_significance='Historic Black community, faith-based organizing',
        past_development_waves=['Post-WWII development', 'White flight era transition'],
        gentrification_history='Not experiencing gentrification',
        opportunity_zones=True,
        special_zoning=[],
    ),

    '60621': ChicagoContextProfile(
        zip_code='60621',
        community_area='Englewood',
        neighborhood_names=['Englewood', 'West Englewood'],
        demographic_character='Predominantly Black, significant poverty, strong community resilience',
        population_trends='Significant decline, but stabilization efforts underway',
        migration_patterns='Outmigration due to economic and safety concerns',
        economic_character='Underserved, major investment opportunity',
        major_employers=['Kennedy-King College', 'Community health centers'],
        commercial_corridors=['63rd Street (historic)', 'Halsted Street'],
        economic_challenges=['High unemployment', 'Commercial desert', 'Vacant lots'],
        housing_stock_character='Mix of maintained homes and vacant properties',
        development_pressures='Low market pressure, strategic public investment',
        affordability_concerns=['Need for quality affordable housing, not displacement'],
        community_organizations=['Teamwork Englewood', 'RAGE'],
        cultural_anchors=['Kennedy-King College', 'Historic 63rd Street memory'],
        community_concerns=['Safety', 'Jobs', 'Youth services', 'Vacant lot remediation'],
        historical_significance='Former thriving commercial hub, civil rights history',
        past_development_waves=['Commercial peak 1920s-60s', 'Decline 1970s-present', 'INVEST South/West'],
        gentrification_history='Not experiencing gentrification; needs investment not displacement',
        tif_districts=['Englewood Mall', '63rd/Ashland'],
        opportunity_zones=True,
        special_zoning=['Invest South/West corridor'],
    ),

    # West Side
    '60624': ChicagoContextProfile(
        zip_code='60624',
        community_area='West Garfield Park',
        neighborhood_names=['West Garfield Park', 'East Garfield Park'],
        demographic_character='Predominantly Black, high poverty, strong community bonds',
        population_trends='Significant historical decline, some stabilization',
        migration_patterns='Outmigration, but community leaders working on retention',
        economic_character='Severely underserved, needs strategic investment',
        major_employers=['Community organizations', 'Social services'],
        commercial_corridors=['Madison Street (historic)', 'Pulaski Road'],
        economic_challenges=['Extreme disinvestment', 'High vacancy', 'Limited services'],
        housing_stock_character='Significant vacancy, historic greystones, rehab potential',
        development_pressures='Low market, strategic rehab and community development',
        affordability_concerns=['Need quality affordable housing development'],
        community_organizations=['Garfield Park Community Council'],
        cultural_anchors=['Garfield Park Conservatory', 'Historic architecture'],
        community_concerns=['Safety', 'Jobs', 'Housing quality', 'Services access'],
        historical_significance='Former thriving community, architectural heritage',
        past_development_waves=['Pre-1960s prosperity', 'Post-riot decline', 'Recent targeted investment'],
        gentrification_history='Not experiencing gentrification',
        opportunity_zones=True,
        special_zoning=['Invest South/West corridor'],
    ),

    '60644': ChicagoContextProfile(
        zip_code='60644',
        community_area='Austin',
        neighborhood_names=['Austin', 'South Austin', 'North Austin'],
        demographic_character='Predominantly Black, working class, largest population on West Side',
        population_trends='Decline but still significant population base',
        migration_patterns='Some outmigration, but strong community identity',
        economic_character='Neighborhood commercial, some manufacturing',
        major_employers=['Local businesses', 'Social services', 'Manufacturing remnants'],
        commercial_corridors=['Chicago Avenue', 'Madison Street', 'Central Avenue'],
        economic_challenges=['Commercial vacancy', 'Job access', 'Transit connectivity'],
        housing_stock_character='Chicago bungalows, greystones, some multifamily',
        development_pressures='Moderate, potential Green Line TOD',
        affordability_concerns=['Maintaining affordability with any development'],
        community_organizations=['Austin Coming Together'],
        cultural_anchors=['Historic homes', 'Community gardens'],
        community_concerns=['Safety', 'Youth programming', 'Economic development'],
        historical_significance='Largest community area by geography, historic bungalows',
        past_development_waves=['Early 20th century development', 'Racial transition 1960s-70s'],
        gentrification_history='Not currently experiencing gentrification',
        opportunity_zones=True,
        special_zoning=[],
    ),

    # North Side - Gentrification Pressure Areas
    '60622': ChicagoContextProfile(
        zip_code='60622',
        community_area='Wicker Park / Bucktown / Ukrainian Village',
        neighborhood_names=['Wicker Park', 'Bucktown', 'Ukrainian Village'],
        demographic_character='Gentrified, young professionals, remnant ethnic communities',
        population_trends='Growing, attracting young professionals',
        migration_patterns='Influx of higher-income residents, displacement of long-term',
        economic_character='Thriving retail, dining, creative economy',
        major_employers=['Tech startups', 'Creative industries', 'Hospitality'],
        commercial_corridors=['Milwaukee Avenue', 'Division Street', 'Damen Avenue'],
        economic_challenges=['Small business affordability', 'Maintaining diversity'],
        housing_stock_character='Historic workers cottages, converted lofts, new luxury',
        development_pressures='High, luxury development, teardowns',
        affordability_concerns=['Severe displacement', 'Loss of affordable units', 'Cultural erasure'],
        community_organizations=['Wicker Park Committee'],
        cultural_anchors=['Double Door (closed)', 'Ukrainian cultural institutions'],
        community_concerns=['Affordability', 'Preserving character', 'Displacement'],
        historical_significance='Immigrant gateway, artist community, gentrification case study',
        past_development_waves=['Polish/Ukrainian settlement', 'Artist pioneer phase 1990s', 'Full gentrification 2000s'],
        gentrification_history='Advanced gentrification, often cited as cautionary example',
        tif_districts=['Kinzie Industrial', 'Division/Homan'],
        opportunity_zones=False,
        special_zoning=['Landmark district portions'],
    ),

    '60647': ChicagoContextProfile(
        zip_code='60647',
        community_area='Logan Square / Humboldt Park',
        neighborhood_names=['Logan Square', 'Humboldt Park', 'Palmer Square'],
        demographic_character='Transitioning, Latino heritage, increasing gentrification',
        population_trends='Growing with demographic shift',
        migration_patterns='Latino outmigration, professional influx',
        economic_character='Growing commercial, restaurant scene, tech presence',
        major_employers=['Local businesses', 'Tech companies', 'Healthcare'],
        commercial_corridors=['Milwaukee Avenue', 'Fullerton Avenue', 'California Avenue'],
        economic_challenges=['Displacement of Latino businesses', 'Affordability'],
        housing_stock_character='Greystones, courtyard buildings, increasing new construction',
        development_pressures='High, rapid change, community resistance',
        affordability_concerns=['Active displacement', 'Loss of Latino businesses', 'Rent increases'],
        community_organizations=['Logan Square Neighborhood Association', 'LUCHA'],
        cultural_anchors=['Logan Square monument', 'Puerto Rican community', 'Paseo Boricua (nearby)'],
        community_concerns=['Anti-displacement organizing', 'Cultural preservation', 'Affordability'],
        historical_significance='Latino cultural center, Puerto Rican heritage',
        past_development_waves=['Early greystone development', 'Latino settlement', 'Current gentrification'],
        gentrification_history='Active gentrification, significant community organizing',
        tif_districts=['Logan Square'],
        opportunity_zones=False,
        special_zoning=[],
    ),

    '60614': ChicagoContextProfile(
        zip_code='60614',
        community_area='Lincoln Park',
        neighborhood_names=['Lincoln Park', 'Old Town', 'Ranch Triangle'],
        demographic_character='Affluent, highly educated, young professionals and families',
        population_trends='Stable, high demand',
        migration_patterns='Attracts high-income residents',
        economic_character='Thriving retail, dining, professional services',
        major_employers=['DePaul University', 'Professional services', 'Hospitals'],
        commercial_corridors=['Armitage Avenue', 'Lincoln Avenue', 'Halsted Street'],
        economic_challenges=['Affordability for workers', 'Retail competition from online'],
        housing_stock_character='Historic homes, luxury condos, limited rental',
        development_pressures='Infill on remaining lots, historic preservation battles',
        affordability_concerns=['Essentially unaffordable for low/moderate income'],
        community_organizations=['Lincoln Park Community Association'],
        cultural_anchors=['Lincoln Park Zoo', 'Steppenwolf Theatre', 'DePaul campus'],
        community_concerns=['Density', 'Parking', 'Historic preservation'],
        historical_significance='Fully gentrified since 1970s, urban renewal history',
        past_development_waves=['Urban renewal displacement 1960s', 'Full gentrification 1970s-80s'],
        gentrification_history='Completed gentrification, cautionary case study',
        opportunity_zones=False,
        special_zoning=['Multiple landmark districts'],
    ),

    # Downtown/Near Loop
    '60601': ChicagoContextProfile(
        zip_code='60601',
        community_area='Loop / New East Side',
        neighborhood_names=['The Loop', 'New East Side', 'Lakeshore East'],
        demographic_character='High-income professionals, increasing residential',
        population_trends='Growing residential population',
        migration_patterns='Attracts downtown workers wanting urban lifestyle',
        economic_character='Commercial center, office market hub',
        major_employers=['Major corporations', 'Financial services', 'Professional services'],
        commercial_corridors=['State Street', 'Michigan Avenue', 'Wacker Drive'],
        economic_challenges=['Office vacancy post-pandemic', 'Retail adaptation'],
        housing_stock_character='High-rise luxury, converted office buildings',
        development_pressures='Office-to-residential conversion potential',
        affordability_concerns=['Not an affordability target area'],
        historical_significance='Heart of Chicago, architectural showcase',
        past_development_waves=['Chicago Fire rebuild', 'Skyscraper era', 'Modern tower development'],
        gentrification_history='Always commercial/high-end, not applicable',
        tif_districts=['LaSalle Central', 'Central Loop'],
        opportunity_zones=False,
        special_zoning=['Planned Development throughout'],
    ),

    '60607': ChicagoContextProfile(
        zip_code='60607',
        community_area='West Loop / Near West Side',
        neighborhood_names=['West Loop', 'Fulton Market', 'Greektown'],
        demographic_character='Young professionals, foodies, tech workers',
        population_trends='Explosive growth',
        migration_patterns='Major influx of high-income residents',
        economic_character='Restaurant/hospitality hub, tech presence, Google campus',
        major_employers=['Google', 'McDonald\'s HQ', 'Restaurant industry'],
        commercial_corridors=['Randolph Street', 'Fulton Market', 'Halsted Street'],
        economic_challenges=['Over-concentration in hospitality', 'Retail sustainability'],
        housing_stock_character='Converted warehouses, new luxury high-rises',
        development_pressures='Extreme, rapid transformation',
        affordability_concerns=['Complete displacement already occurred', 'Worker affordability'],
        community_organizations=['West Loop Community Organization'],
        cultural_anchors=['Restaurant Row', 'Greektown heritage'],
        community_concerns=['Traffic', 'Density', 'Neighborhood character'],
        historical_significance='Former meatpacking/industrial, now dining destination',
        past_development_waves=['Industrial era', 'Loft conversion 1990s', 'Luxury development 2010s-present'],
        gentrification_history='Completed super-gentrification',
        tif_districts=['Kinzie Industrial Corridor'],
        opportunity_zones=False,
        special_zoning=['Planned Manufacturing Districts (eroding)'],
    ),
}


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
        for zip_code in zip_codes:
            if zip_code in self.chicago_profiles:
                profile = self.chicago_profiles[zip_code]
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

    def _build_demographic_context(
        self,
        context: CAGContext,
        zip_codes: List[str],
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build demographic context from profiles and data."""
        demographic_patterns = {}

        for zip_code in zip_codes:
            if zip_code in self.chicago_profiles:
                profile = self.chicago_profiles[zip_code]
                demographic_patterns[zip_code] = {
                    'character': profile.demographic_character,
                    'trends': profile.population_trends,
                    'migration': profile.migration_patterns,
                }

        # Add aggregate patterns
        gentrifying = [z for z in zip_codes if z in self.chicago_profiles
                       and 'gentrification' in self.chicago_profiles[z].gentrification_history.lower()
                       and 'not' not in self.chicago_profiles[z].gentrification_history.lower()]
        declining = [z for z in zip_codes if z in self.chicago_profiles
                     and 'declin' in self.chicago_profiles[z].population_trends.lower()]

        context.domain_context['demographic'] = {
            'zip_patterns': demographic_patterns,
            'gentrifying_areas': gentrifying,
            'declining_areas': declining,
            'total_zips_analyzed': len(zip_codes),
        }

        if ContextType.DEMOGRAPHIC not in context.context_types:
            context.context_types.append(ContextType.DEMOGRAPHIC)

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

        for zip_code in zip_codes:
            if zip_code in self.chicago_profiles:
                profile = self.chicago_profiles[zip_code]
                economic_patterns[zip_code] = {
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
            'major_employers': list(set(all_employers)),
            'commercial_corridors': list(set(all_corridors)),
            'common_challenges': list(set(all_challenges)),
        }

        if ContextType.ECONOMIC not in context.context_types:
            context.context_types.append(ContextType.ECONOMIC)

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

        for zip_code in zip_codes:
            if zip_code in self.chicago_profiles:
                profile = self.chicago_profiles[zip_code]
                housing_patterns[zip_code] = {
                    'stock': profile.housing_stock_character,
                    'pressures': profile.development_pressures,
                    'affordability': profile.affordability_concerns,
                }

                if 'high' in profile.development_pressures.lower():
                    high_pressure.append(zip_code)
                elif 'low' in profile.development_pressures.lower():
                    low_pressure.append(zip_code)

                affordability_issues.extend(profile.affordability_concerns)

        context.domain_context['housing'] = {
            'zip_patterns': housing_patterns,
            'high_pressure_zips': high_pressure,
            'low_pressure_zips': low_pressure,
            'affordability_concerns': list(set(affordability_issues)),
        }

        if ContextType.HOUSING not in context.context_types:
            context.context_types.append(ContextType.HOUSING)

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

        for zip_code in zip_codes:
            if zip_code in self.chicago_profiles:
                profile = self.chicago_profiles[zip_code]
                community_patterns[zip_code] = {
                    'organizations': profile.community_organizations,
                    'anchors': profile.cultural_anchors,
                    'concerns': profile.community_concerns,
                }
                all_organizations.extend(profile.community_organizations)
                all_concerns.extend(profile.community_concerns)

        context.domain_context['community'] = {
            'zip_patterns': community_patterns,
            'active_organizations': list(set(all_organizations)),
            'common_concerns': list(set(all_concerns)),
        }

        if ContextType.COMMUNITY not in context.context_types:
            context.context_types.append(ContextType.COMMUNITY)

    def _build_historical_context(
        self,
        context: CAGContext,
        zip_codes: List[str],
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build historical context from profiles."""
        historical_patterns = {}

        for zip_code in zip_codes:
            if zip_code in self.chicago_profiles:
                profile = self.chicago_profiles[zip_code]
                historical_patterns[zip_code] = {
                    'significance': profile.historical_significance,
                    'development_waves': profile.past_development_waves,
                    'gentrification': profile.gentrification_history,
                }

        # Identify historical pattern categories
        former_industrial = [z for z in zip_codes if z in self.chicago_profiles
                            and any('industrial' in w.lower() for w in self.chicago_profiles[z].past_development_waves)]
        gentrification_cases = [z for z in zip_codes if z in self.chicago_profiles
                               and 'gentrification' in self.chicago_profiles[z].gentrification_history.lower()]

        context.historical_patterns = {
            'zip_patterns': historical_patterns,
            'former_industrial': former_industrial,
            'gentrification_case_studies': gentrification_cases,
        }

        if ContextType.HISTORICAL not in context.context_types:
            context.context_types.append(ContextType.HISTORICAL)

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

        for zip_code in zip_codes:
            if zip_code in self.chicago_profiles:
                profile = self.chicago_profiles[zip_code]
                policy_patterns[zip_code] = {
                    'tif': profile.tif_districts,
                    'opportunity_zone': profile.opportunity_zones,
                    'special_zoning': profile.special_zoning,
                }

                if profile.opportunity_zones:
                    opportunity_zones.append(zip_code)
                tif_districts.extend(profile.tif_districts)

        context.domain_context['policy'] = {
            'zip_patterns': policy_patterns,
            'opportunity_zones': opportunity_zones,
            'tif_districts': list(set(tif_districts)),
        }

        if ContextType.POLICY not in context.context_types:
            context.context_types.append(ContextType.POLICY)

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

        # Check for known patterns that might cause statistical anomalies
        for zip_code in zip_codes:
            if zip_code in self.chicago_profiles:
                profile = self.chicago_profiles[zip_code]

                # University areas have unique demographics
                if 'university' in profile.demographic_character.lower():
                    anomalies.append(f"{zip_code}: University presence affects demographic patterns")

                # Industrial transition areas
                if 'industrial' in profile.economic_character.lower():
                    anomalies.append(f"{zip_code}: Industrial transition may affect economic indicators")

                # Active gentrification
                if 'active' in profile.gentrification_history.lower():
                    anomalies.append(f"{zip_code}: Active gentrification creating rapid change")

        return anomalies

    def _compile_policy_considerations(self, zip_codes: List[str]) -> List[str]:
        """Compile policy considerations for the geographic scope."""
        considerations = []

        oz_count = sum(1 for z in zip_codes if z in self.chicago_profiles
                       and self.chicago_profiles[z].opportunity_zones)
        if oz_count > 0:
            considerations.append(f"{oz_count} ZIP codes are Opportunity Zones - investment incentives may accelerate development")

        # INVEST South/West
        isw_zips = [z for z in zip_codes if z in self.chicago_profiles
                    and any('invest' in sz.lower() for sz in self.chicago_profiles[z].special_zoning)]
        if isw_zips:
            considerations.append(f"INVEST South/West corridors in {', '.join(isw_zips)} - major public investment planned")

        return considerations

    def _identify_geographic_patterns(self, zip_codes: List[str]) -> List[Dict[str, Any]]:
        """Identify geographic patterns across ZIP codes."""
        patterns = []

        # Group by gentrification status
        gentrifying = [z for z in zip_codes if z in self.chicago_profiles
                       and 'gentrification' in self.chicago_profiles[z].gentrification_history.lower()
                       and 'not' not in self.chicago_profiles[z].gentrification_history.lower()]
        if gentrifying:
            patterns.append({
                'name': 'Gentrification Pressure',
                'description': f'Active gentrification in {len(gentrifying)} areas: {", ".join(gentrifying)}',
                'zip_codes': gentrifying,
                'pattern_type': 'demographic_shift',
            })

        # Group by opportunity zone status
        opportunity = [z for z in zip_codes if z in self.chicago_profiles
                       and self.chicago_profiles[z].opportunity_zones]
        if opportunity:
            patterns.append({
                'name': 'Opportunity Zone Concentration',
                'description': f'{len(opportunity)} Opportunity Zones may see accelerated investment',
                'zip_codes': opportunity,
                'pattern_type': 'policy_incentive',
            })

        return patterns
