#!/usr/bin/env python3
"""
CAG Framework - Minimal Usage Example

A simple, focused example showing the most common CAG use case:
enhancing pipeline results with contextual interpretation.

For advanced usage patterns, see cag_advanced_example.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cag import CAGPipeline


def main():
    """Minimal example: Enhance pipeline results with CAG."""

    # Example pipeline results (in practice, these come from your pipeline)
    pipeline_results = {
        'model_results': {
            'multifamily_growth': {
                'top_growth_zips': ['60615', '60607', '60647'],
                'growth_scores': {
                    '60615': 0.85,
                    '60607': 0.92,
                    '60647': 0.78,
                },
            },
            'income_distribution': {
                'zip_metrics': {
                    '60615': {'median_income_change': 0.15, 'income_diversity': 0.4},
                    '60647': {'median_income_change': 0.12, 'income_diversity': 0.45},
                },
            },
        }
    }

    # Create CAG pipeline and enhance results
    cag = CAGPipeline(output_dir=Path("output/cag_minimal"))
    enhanced = cag.enhance_pipeline_results(pipeline_results)

    # Print key findings
    print("CAG Enhancement Complete")
    print("=" * 40)

    if 'cag' in enhanced:
        cag_section = enhanced['cag']

        if 'interpretation' in cag_section:
            print("\nReality Bridge:")
            print(cag_section['interpretation'].get('reality_bridge', 'N/A'))

        if 'patterns' in cag_section:
            patterns = cag_section['patterns'].get('discovered', [])
            print(f"\nPatterns Discovered: {len(patterns)}")

        if 'blueprints' in cag_section:
            investigations = cag_section['blueprints'].get('investigations', [])
            print(f"Suggested Investigations: {len(investigations)}")

    print("\nFull report saved to output/cag_minimal/")


if __name__ == "__main__":
    main()
