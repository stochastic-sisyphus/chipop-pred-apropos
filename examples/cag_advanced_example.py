#!/usr/bin/env python3
"""
CAG Framework - Advanced Usage Examples

Comprehensive examples demonstrating all CAG framework capabilities:
1. Post-hoc enhancement of pipeline results
2. Standalone analysis on custom data
3. Individual component usage (ContextBuilder, RealityInterpreter, etc.)
4. Custom neighborhood profiles
5. Different interpretation modes

For a simpler starting point, see cag_minimal_example.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cag import (
    CAGPipeline,
    ContextBuilder,
    RealityInterpreter,
    PatternDiscovery,
    BlueprintGenerator,
    CAGContext,
    ChicagoContextProfile,
)
from src.cag.base import ContextType, InterpretationMode


def example_1_post_hoc_enhancement():
    """
    Example 1: Enhance existing pipeline results with CAG analysis.

    This is the most common use case - run your existing pipeline,
    then enhance the results with contextual interpretation.
    """
    print("\n" + "="*60)
    print("Example 1: Post-hoc Enhancement of Pipeline Results")
    print("="*60)

    # Simulate pipeline results (in practice, these come from your actual pipeline)
    pipeline_results = {
        'model_results': {
            'multifamily_growth': {
                'top_growth_zips': ['60615', '60607', '60647'],
                'growth_scores': {
                    '60615': 0.85,
                    '60607': 0.92,
                    '60647': 0.78,
                    '60619': 0.25,
                    '60621': 0.15,
                },
                'zip_codes': ['60615', '60607', '60647', '60619', '60621'],
            },
            'income_distribution': {
                'zip_metrics': {
                    '60615': {'median_income_change': 0.15, 'income_diversity': 0.4},
                    '60647': {'median_income_change': 0.12, 'income_diversity': 0.45},
                    '60619': {'median_income_change': -0.02, 'income_diversity': 0.7},
                },
                'gentrification_risk': {
                    '60615': 'moderate',
                    '60647': 'high',
                    '60619': 'low',
                },
            },
            'retail_gap': {
                'gap_scores': {
                    '60615': 0.3,
                    '60619': 0.7,
                    '60621': 0.85,
                },
                'opportunity_zones': ['60619', '60621'],
            },
        }
    }

    # Create CAG pipeline
    cag = CAGPipeline(
        output_dir=Path("output/cag_example"),
        enable_context_building=True,
        enable_reality_interpretation=True,
        enable_pattern_discovery=True,
        enable_blueprint_generation=True,
    )

    # Enhance pipeline results
    enhanced = cag.enhance_pipeline_results(
        pipeline_results,
        zip_codes=['60615', '60607', '60647', '60619', '60621'],
        time_period="2020-2024",
    )

    # Print results
    print("\n--- CAG Enhancement Results ---")

    if 'cag' in enhanced:
        cag_section = enhanced['cag']

        if 'interpretation' in cag_section:
            print("\nReality Bridge:")
            print(cag_section['interpretation'].get('reality_bridge', 'N/A'))

            if cag_section['interpretation'].get('anomalies'):
                print("\nIdentified Anomalies:")
                for anomaly in cag_section['interpretation']['anomalies']:
                    print(f"  - {anomaly.get('type')}: {anomaly.get('description', '')[:100]}...")

        if 'patterns' in cag_section:
            print(f"\nDiscovered Patterns: {len(cag_section['patterns'].get('discovered', []))}")

        if 'blueprints' in cag_section:
            print(f"Suggested Investigations: {len(cag_section['blueprints'].get('investigations', []))}")

    # Generate and print report
    print("\n--- Generated Report (excerpt) ---")
    report = cag.generate_cag_report()
    print(report[:2000] + "...\n[truncated]")

    return enhanced


def example_2_standalone_analysis():
    """
    Example 2: Run CAG analysis on custom data.

    Use this when you want to analyze data that didn't come from
    the standard pipeline, or when exploring new data sources.
    """
    print("\n" + "="*60)
    print("Example 2: Standalone CAG Analysis")
    print("="*60)

    # Custom data for analysis
    custom_data = {
        'permit_trends': {
            '60617': {'2023': 45, '2024': 120, 'change': 1.67},
            '60620': {'2023': 30, '2024': 35, 'change': 0.17},
        },
        'economic_indicators': {
            'unemployment_rate': 0.052,
            'median_income_growth': 0.03,
            'retail_per_capita': 0.4,
        },
    }

    # Create CAG pipeline
    cag = CAGPipeline(output_dir=Path("output/cag_standalone"))

    # Run standalone analysis
    result = cag.run_standalone(
        data=custom_data,
        analysis_type="custom_permit_analysis",
        zip_codes=['60617', '60620'],
    )

    print("\n--- Standalone Analysis Results ---")
    print(f"Analysis ID: {result.analysis_id}")
    print(f"Interpretation: {result.contextual_interpretation}")
    print(f"Patterns Found: {len(result.discovered_patterns)}")
    print(f"Anomalies: {len(result.anomalies_identified)}")

    return result


def example_3_individual_components():
    """
    Example 3: Use individual CAG components directly.

    For maximum flexibility, use components individually
    to build custom analysis workflows.
    """
    print("\n" + "="*60)
    print("Example 3: Individual Component Usage")
    print("="*60)

    # ----- Context Builder -----
    print("\n--- Context Builder ---")
    context_builder = ContextBuilder()

    # Build context for specific ZIP codes
    context = context_builder.build_for_zip_codes(
        zip_codes=['60615', '60622', '60647'],
        analysis_type="gentrification_study",
        time_period="2020-2024",
        context_types=[ContextType.DEMOGRAPHIC, ContextType.HOUSING, ContextType.HISTORICAL],
    )

    print(f"Built context for {context.geographic_scope}")
    print(f"Context types: {[ct.value for ct in context.context_types]}")
    print(f"Community profiles: {len(context.community_profiles)}")

    # Get a specific profile
    profile = context_builder.get_profile('60622')
    if profile:
        print(f"\n60622 (Wicker Park) Profile:")
        print(f"  - Character: {profile.demographic_character}")
        print(f"  - Gentrification: {profile.gentrification_history}")

    # ----- Reality Interpreter -----
    print("\n--- Reality Interpreter ---")
    interpreter = RealityInterpreter()

    # Sample statistical data
    stats = {
        'median_income_growth': 0.08,
        'unemployment_rate': 0.04,
        'rent_to_income_ratio': 0.35,
        'permit_growth_rate': 0.25,
    }

    # Interpret with context
    result = interpreter.process(stats, context)

    print(f"Interpretation: {result.contextual_interpretation}")
    print(f"Reality Bridge: {result.lived_reality_bridge[:200]}...")

    # ----- Pattern Discovery -----
    print("\n--- Pattern Discovery ---")
    discovery = PatternDiscovery()

    # Sample model results
    model_data = {
        'model_results': {
            'income_distribution': {
                'zip_metrics': {
                    '60622': {'median_income_change': 0.12, 'income_diversity': 0.35},
                    '60647': {'median_income_change': 0.10, 'income_diversity': 0.40},
                }
            },
            'multifamily_growth': {
                'growth_scores': {
                    '60622': 0.8,
                    '60647': 0.75,
                    '60619': 0.2,
                }
            }
        }
    }

    discovery_result = discovery.process(model_data, context)

    print(f"Patterns discovered: {len(discovery_result.discovered_patterns)}")
    for pattern in discovery_result.discovered_patterns[:3]:
        print(f"  - {pattern.get('name', 'Pattern')}: {pattern.get('confidence', 'unknown')} confidence")

    # ----- Blueprint Generator -----
    print("\n--- Blueprint Generator ---")
    blueprints = BlueprintGenerator()

    blueprint_result = blueprints.process({
        'model_results': model_data['model_results'],
        'discovered_patterns': discovery_result.discovered_patterns,
    }, context)

    print(f"Blueprints generated: {len(blueprint_result.suggested_investigations)}")
    for inv in blueprint_result.suggested_investigations[:3]:
        print(f"  - {inv}")

    return context, result, discovery_result, blueprint_result


def example_4_custom_profiles():
    """
    Example 4: Add custom neighborhood profiles.

    Extend the knowledge base with custom profiles for
    neighborhoods not yet covered or for special analysis.
    """
    print("\n" + "="*60)
    print("Example 4: Custom Neighborhood Profiles")
    print("="*60)

    context_builder = ContextBuilder()

    # Create custom profile for a neighborhood
    custom_profile = ChicagoContextProfile(
        zip_code='60612',
        community_area='Near West Side',
        neighborhood_names=['Near West Side', 'Illinois Medical District'],
        demographic_character='Mixed - medical workers, students, transitioning residential',
        population_trends='Growing due to medical district expansion',
        migration_patterns='Influx of medical professionals and students',
        economic_character='Anchor institutions (hospitals, university), growing commercial',
        major_employers=['Rush University Medical Center', 'UIC', 'Jesse Brown VA'],
        commercial_corridors=['Ashland Avenue', 'Madison Street'],
        economic_challenges=['Displacement pressure from institutional expansion'],
        housing_stock_character='Mix of historic homes, new high-rises, student housing',
        development_pressures='High - driven by medical district and university',
        affordability_concerns=['Hospital worker affordability', 'Student housing costs'],
        community_organizations=['Near West Side Community Development Corporation'],
        cultural_anchors=['United Center', 'Little Italy remnants'],
        community_concerns=['Institutional expansion impacts', 'Traffic and parking'],
        historical_significance='Historic immigrant neighborhoods, Jane Addams Hull House',
        past_development_waves=['Italian immigration', 'Urban renewal', 'Medical district growth'],
        gentrification_history='Institutional gentrification ongoing',
        tif_districts=['Roosevelt/Racine', 'Midwest'],
        opportunity_zones=False,
        special_zoning=['Planned Development for medical district'],
    )

    # Add to context builder
    context_builder.add_custom_profile(custom_profile)

    # Build context including new profile
    context = context_builder.build_for_zip_codes(
        zip_codes=['60612', '60607', '60622'],
        analysis_type="medical_district_impact",
    )

    print(f"Context now includes {len(context.community_profiles)} community profiles")
    print(f"60612 profile added: {'60612' in context.community_profiles}")

    # Verify custom profile is used
    profile_data = context.community_profiles.get('60612', {})
    print(f"\nCustom profile data:")
    print(f"  - Community area: {profile_data.get('geography', {}).get('community_area')}")
    print(f"  - Major employers: {profile_data.get('economic', {}).get('major_employers')}")

    return context


def example_5_interpretation_modes():
    """
    Example 5: Different interpretation modes.

    The Reality Interpreter supports multiple modes for
    different output needs.
    """
    print("\n" + "="*60)
    print("Example 5: Interpretation Modes")
    print("="*60)

    interpreter = RealityInterpreter()
    context_builder = ContextBuilder()

    context = context_builder.build_for_zip_codes(
        zip_codes=['60619'],
        analysis_type="mode_comparison",
    )

    stats = {
        'median_income_growth': 0.02,
        'unemployment_rate': 0.08,
        'retail_per_capita': 0.35,
    }

    # Statistical mode (basic)
    print("\n--- Statistical Mode ---")
    stat_result = interpreter.interpret(stats, context, InterpretationMode.STATISTICAL)
    print(f"Mode: {stat_result['mode']}")
    print(f"Output: Basic summary")

    # Contextual mode (enhanced)
    print("\n--- Contextual Mode ---")
    ctx_result = interpreter.interpret(stats, context, InterpretationMode.CONTEXTUAL)
    print(f"Mode: {ctx_result['mode']}")
    print(f"Interpretations: {len(ctx_result.get('interpretations', []))}")

    # Narrative mode (full)
    print("\n--- Narrative Mode ---")
    narr_result = interpreter.interpret(stats, context, InterpretationMode.NARRATIVE)
    print(f"Mode: {narr_result['mode']}")
    if 'narratives' in narr_result:
        print(f"Overview: {narr_result['narratives'].get('overview', 'N/A')[:200]}...")

    return stat_result, ctx_result, narr_result


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# CAG Framework Usage Examples")
    print("#"*60)

    # Run all examples
    example_1_post_hoc_enhancement()
    example_2_standalone_analysis()
    example_3_individual_components()
    example_4_custom_profiles()
    example_5_interpretation_modes()

    print("\n" + "#"*60)
    print("# Examples Complete")
    print("#"*60)
    print("\nFor more information, see the CAG module documentation.")
