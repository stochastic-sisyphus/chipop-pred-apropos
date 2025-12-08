"""
CAG Pipeline - Orchestration Layer for Context Augmented Generation.

Provides optional integration with the existing Chicago pipeline,
following the principle of "modular plugins, not system replacement."

The CAGPipeline wraps the existing pipeline and enhances its outputs
with contextual interpretation, pattern discovery, and blueprint generation
without modifying core pipeline logic.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging

from .base import (
    CAGComponent,
    CAGContext,
    CAGResult,
    CAGPlugin,
    ContextType,
    InterpretationMode,
)
from .context_builder import ContextBuilder, ChicagoContextProfile
from .reality_interpreter import RealityInterpreter
from .pattern_discovery import PatternDiscovery
from .blueprint_generator import BlueprintGenerator

logger = logging.getLogger(__name__)


class CAGPipeline:
    """
    Orchestrates CAG components for enhanced pipeline analysis.

    This is an optional wrapper that enhances existing pipeline outputs
    with contextual interpretation without requiring pipeline modifications.

    Usage Patterns:
    1. Post-hoc enhancement: Run after existing pipeline, enhance results
    2. Integrated mode: Hook into pipeline events via plugin interface
    3. Standalone analysis: Run CAG components on any data

    Architecture Principles:
    - Non-invasive: Works alongside existing pipeline
    - Reversible: CAG outputs are additive, not replacements
    - Modular: Each component can be used independently
    - Configurable: All behaviors can be customized
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        enable_context_building: bool = True,
        enable_reality_interpretation: bool = True,
        enable_pattern_discovery: bool = True,
        enable_blueprint_generation: bool = True,
    ):
        """
        Initialize CAG Pipeline.

        Args:
            output_dir: Directory for CAG outputs
            enable_context_building: Whether to build context
            enable_reality_interpretation: Whether to interpret reality
            enable_pattern_discovery: Whether to discover patterns
            enable_blueprint_generation: Whether to generate blueprints
        """
        self.output_dir = output_dir or Path("output/cag")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("cag.pipeline")

        # Initialize components
        self.components: Dict[str, CAGComponent] = {}

        if enable_context_building:
            self.context_builder = ContextBuilder(self.output_dir / "context")
            self.components['context_builder'] = self.context_builder
        else:
            self.context_builder = None

        if enable_reality_interpretation:
            self.reality_interpreter = RealityInterpreter(self.output_dir / "interpretation")
            self.components['reality_interpreter'] = self.reality_interpreter
        else:
            self.reality_interpreter = None

        if enable_pattern_discovery:
            self.pattern_discovery = PatternDiscovery(self.output_dir / "patterns")
            self.components['pattern_discovery'] = self.pattern_discovery
        else:
            self.pattern_discovery = None

        if enable_blueprint_generation:
            self.blueprint_generator = BlueprintGenerator(self.output_dir / "blueprints")
            self.components['blueprint_generator'] = self.blueprint_generator
        else:
            self.blueprint_generator = None

        # Pipeline state
        self._current_context: Optional[CAGContext] = None
        self._results: List[CAGResult] = []
        self._run_count = 0

    def enhance_pipeline_results(
        self,
        pipeline_results: Dict[str, Any],
        zip_codes: Optional[List[str]] = None,
        time_period: str = "2020-2024",
    ) -> Dict[str, Any]:
        """
        Enhance existing pipeline results with CAG analysis.

        This is the primary integration point for post-hoc enhancement.
        Takes complete pipeline results and adds contextual layers.

        Args:
            pipeline_results: Results from existing pipeline run
            zip_codes: ZIP codes to focus on (auto-detected if None)
            time_period: Time period for analysis

        Returns:
            Enhanced results with CAG additions
        """
        self._run_count += 1
        self.logger.info(f"CAG enhancement run {self._run_count}")

        # Prepare data structure
        data = {
            'model_results': pipeline_results.get('model_results', {}),
            'collected_data': pipeline_results.get('collected_data', {}),
            'reports': pipeline_results.get('reports', {}),
        }

        # Auto-detect ZIP codes if not provided
        if zip_codes is None:
            zip_codes = self._extract_zip_codes(data)

        # Phase 1: Build Context
        context = None
        if self.context_builder:
            self.logger.info("Building context...")
            context = self.context_builder.build_for_zip_codes(
                zip_codes=zip_codes,
                analysis_type="pipeline_enhancement",
                time_period=time_period,
            )

            # Add statistical summary from pipeline
            context.statistical_summary = self._summarize_model_results(
                data.get('model_results', {})
            )
            context.model_outputs = data.get('model_results', {})

            self._current_context = context
            self.logger.info(f"Context built for {len(zip_codes)} ZIP codes")

        # Phase 2: Reality Interpretation
        interpretation_result = None
        if self.reality_interpreter and context:
            self.logger.info("Interpreting reality bridge...")
            interpretation_result = self.reality_interpreter.process(
                data.get('model_results', {}),
                context
            )
            self._results.append(interpretation_result)
            self.logger.info("Reality interpretation complete")

        # Phase 3: Pattern Discovery
        discovery_result = None
        if self.pattern_discovery:
            self.logger.info("Discovering patterns...")
            discovery_result = self.pattern_discovery.process(
                data,
                context
            )
            self._results.append(discovery_result)
            self.logger.info(f"Discovered {len(discovery_result.discovered_patterns)} patterns")

        # Phase 4: Blueprint Generation
        blueprint_result = None
        if self.blueprint_generator:
            self.logger.info("Generating blueprints...")
            blueprint_data = {
                'model_results': data.get('model_results', {}),
                'discovered_patterns': discovery_result.discovered_patterns if discovery_result else [],
                'cag_results': self._results,
            }
            blueprint_result = self.blueprint_generator.process(
                blueprint_data,
                context
            )
            self._results.append(blueprint_result)
            self.logger.info(f"Generated {len(blueprint_result.suggested_investigations)} blueprints")

        # Compile enhanced results
        enhanced = self._compile_enhanced_results(
            pipeline_results,
            context,
            interpretation_result,
            discovery_result,
            blueprint_result,
        )

        # Save CAG outputs
        self._save_cag_outputs(enhanced)

        self.logger.info("CAG enhancement complete")
        return enhanced

    def create_plugin(self) -> CAGPlugin:
        """
        Create a plugin for integrated pipeline execution.

        Returns a plugin that can be registered with the existing
        pipeline for event-based CAG enhancement.

        Returns:
            CAGPlugin instance
        """
        return CAGPipelinePlugin(self)

    def run_standalone(
        self,
        data: Dict[str, Any],
        analysis_type: str = "standalone",
        zip_codes: Optional[List[str]] = None,
    ) -> CAGResult:
        """
        Run CAG analysis on arbitrary data (standalone mode).

        Args:
            data: Data to analyze
            analysis_type: Type of analysis
            zip_codes: ZIP codes to focus on

        Returns:
            Comprehensive CAG result
        """
        self._run_count += 1

        # Build context
        if zip_codes is None:
            zip_codes = self._extract_zip_codes(data)

        context = None
        if self.context_builder:
            context = self.context_builder.build_for_zip_codes(
                zip_codes=zip_codes,
                analysis_type=analysis_type,
            )

        # Create aggregate result
        aggregate = CAGResult(
            analysis_id=f"standalone_{self._run_count}",
            context_used=context,
        )

        # Run each component
        if self.reality_interpreter and context:
            interp = self.reality_interpreter.process(data, context)
            aggregate.contextual_interpretation = interp.contextual_interpretation
            aggregate.lived_reality_bridge = interp.lived_reality_bridge
            aggregate.policy_implications.extend(interp.policy_implications)

        if self.pattern_discovery:
            discovery = self.pattern_discovery.process(data, context)
            aggregate.discovered_patterns.extend(discovery.discovered_patterns)
            aggregate.anomalies_identified.extend(discovery.anomalies_identified)

        if self.blueprint_generator:
            blueprint_data = {
                'model_results': data,
                'discovered_patterns': aggregate.discovered_patterns,
            }
            blueprints = self.blueprint_generator.process(blueprint_data, context)
            aggregate.suggested_investigations.extend(blueprints.suggested_investigations)

        return aggregate

    def interpret_model(
        self,
        model_name: str,
        model_results: Dict[str, Any],
        zip_codes: Optional[List[str]] = None,
    ) -> CAGResult:
        """
        Interpret results from a specific model.

        Convenience method for targeted model interpretation.

        Args:
            model_name: Name of the model
            model_results: Results from that model
            zip_codes: ZIP codes to contextualize

        Returns:
            CAG interpretation result
        """
        if not self.reality_interpreter:
            raise RuntimeError("Reality interpreter not enabled")

        # Build context if available
        context = None
        if self.context_builder and zip_codes:
            context = self.context_builder.build_for_zip_codes(
                zip_codes=zip_codes,
                analysis_type=f"{model_name}_interpretation",
            )

        return self.reality_interpreter.interpret_model_results(
            model_name,
            model_results,
            context or CAGContext(
                analysis_type=model_name,
                geographic_scope="Chicago",
                time_period="current",
            )
        )

    def get_context(self) -> Optional[CAGContext]:
        """Get the current CAG context."""
        return self._current_context

    def get_results(self) -> List[CAGResult]:
        """Get all CAG results from current session."""
        return self._results

    def generate_cag_report(self) -> str:
        """Generate comprehensive CAG analysis report."""
        sections = []

        sections.append("# Context Augmented Generation Report")
        sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sections.append(f"Run Count: {self._run_count}")
        sections.append("")

        # Context summary
        if self._current_context:
            sections.append("## Analysis Context")
            sections.append(self._current_context.to_prompt_context())
            sections.append("")

        # Aggregate findings
        sections.append("## Key Findings")
        sections.append("")

        all_interpretations = []
        all_patterns = []
        all_investigations = []
        all_implications = []

        for result in self._results:
            if result.contextual_interpretation:
                all_interpretations.append(result.contextual_interpretation)
            if result.lived_reality_bridge:
                all_interpretations.append(f"**Reality Bridge**: {result.lived_reality_bridge}")
            all_patterns.extend(result.discovered_patterns)
            all_investigations.extend(result.suggested_investigations)
            all_implications.extend(result.policy_implications)

        if all_interpretations:
            sections.append("### Interpretations")
            for interp in all_interpretations:
                sections.append(interp)
                sections.append("")

        if all_patterns:
            sections.append("### Discovered Patterns")
            for pattern in all_patterns:
                if isinstance(pattern, dict):
                    sections.append(f"- **{pattern.get('name', 'Pattern')}**: {pattern.get('description', '')}")
                else:
                    sections.append(f"- {pattern}")
            sections.append("")

        if all_implications:
            sections.append("### Policy Implications")
            for impl in set(all_implications):
                sections.append(f"- {impl}")
            sections.append("")

        if all_investigations:
            sections.append("### Suggested Investigations")
            for inv in set(all_investigations):
                sections.append(f"- {inv}")
            sections.append("")

        # Component-specific reports
        if self.pattern_discovery and self.pattern_discovery.discovered_patterns:
            sections.append("## Pattern Discovery Details")
            sections.append(self.pattern_discovery.generate_discovery_report())

        if self.blueprint_generator and self.blueprint_generator.blueprints:
            sections.append("## Analysis Blueprints")
            sections.append(self.blueprint_generator.generate_blueprint_report())

        return "\n".join(sections)

    def reset(self) -> None:
        """Reset pipeline state for a new analysis session."""
        self._current_context = None
        self._results = []

        if self.pattern_discovery:
            self.pattern_discovery.discovered_patterns = []
            self.pattern_discovery.feature_suggestions = []

        if self.blueprint_generator:
            self.blueprint_generator.blueprints = []

        self.logger.info("CAG pipeline reset")

    def _extract_zip_codes(self, data: Dict[str, Any]) -> List[str]:
        """Extract ZIP codes from data."""
        zip_codes = set()

        # Search model results
        model_results = data.get('model_results', {})
        for model_name, results in model_results.items():
            if isinstance(results, dict):
                if 'zip_codes' in results:
                    zip_codes.update(str(z) for z in results['zip_codes'])
                if 'top_growth_zips' in results:
                    zip_codes.update(str(z) for z in results['top_growth_zips'])
                if 'opportunity_zones' in results:
                    zip_codes.update(str(z) for z in results['opportunity_zones'])

                # Check for nested ZIP data
                for key, value in results.items():
                    if isinstance(value, dict):
                        zip_codes.update(str(k) for k in value.keys() if len(str(k)) == 5 and str(k).isdigit())

        # Default to context builder's known ZIPs if none found
        if not zip_codes and self.context_builder:
            zip_codes = set(self.context_builder.chicago_profiles.keys())

        return sorted(list(zip_codes))

    def _summarize_model_results(
        self,
        model_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create summary of model results for context."""
        summary = {}

        for model_name, results in model_results.items():
            if isinstance(results, dict):
                model_summary = {}

                # Extract key metrics
                for key in ['top_growth_zips', 'opportunity_zones', 'growth_scores',
                           'gap_scores', 'risk_scores', 'forecasts']:
                    if key in results:
                        value = results[key]
                        if isinstance(value, dict):
                            model_summary[key] = f"{len(value)} entries"
                        elif isinstance(value, list):
                            model_summary[key] = f"{len(value)} items"
                        else:
                            model_summary[key] = value

                if model_summary:
                    summary[model_name] = model_summary

        return summary

    def _compile_enhanced_results(
        self,
        original: Dict[str, Any],
        context: Optional[CAGContext],
        interpretation: Optional[CAGResult],
        discovery: Optional[CAGResult],
        blueprints: Optional[CAGResult],
    ) -> Dict[str, Any]:
        """Compile enhanced results combining original and CAG outputs."""
        enhanced = original.copy()

        # Add CAG section
        enhanced['cag'] = {
            'enhanced': True,
            'timestamp': datetime.now().isoformat(),
            'run_id': self._run_count,
        }

        if context:
            enhanced['cag']['context'] = context.to_dict()

        if interpretation:
            enhanced['cag']['interpretation'] = {
                'contextual': interpretation.contextual_interpretation,
                'reality_bridge': interpretation.lived_reality_bridge,
                'community_impact': interpretation.community_impact_narrative,
                'anomalies': interpretation.anomalies_identified,
                'policy_implications': interpretation.policy_implications,
            }

        if discovery:
            enhanced['cag']['patterns'] = {
                'discovered': discovery.discovered_patterns,
                'investigations': discovery.suggested_investigations,
            }

        if blueprints:
            enhanced['cag']['blueprints'] = {
                'suggested': blueprints.discovered_patterns,
                'investigations': blueprints.suggested_investigations,
            }

        return enhanced

    def _save_cag_outputs(self, enhanced: Dict[str, Any]) -> None:
        """Save CAG outputs to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save full enhanced results
        results_path = self.output_dir / f"enhanced_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(enhanced, f, indent=2, default=str)

        # Save CAG report
        report_path = self.output_dir / f"cag_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(self.generate_cag_report())

        self.logger.info(f"CAG outputs saved to {self.output_dir}")


class CAGPipelinePlugin(CAGPlugin):
    """
    Plugin implementation for integrated pipeline execution.

    Hooks into existing pipeline events to provide real-time
    CAG enhancement during pipeline execution.
    """

    def __init__(self, cag_pipeline: CAGPipeline):
        """
        Initialize plugin with CAG pipeline.

        Args:
            cag_pipeline: CAG pipeline instance to use
        """
        super().__init__(
            name="cag_enhancement",
            components=list(cag_pipeline.components.values())
        )
        self.cag_pipeline = cag_pipeline
        self._collected_data: Dict[str, Any] = {}

    def pre_pipeline(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare CAG components before pipeline runs."""
        self.cag_pipeline.reset()
        self.logger.info("CAG plugin initialized for pipeline run")
        return pipeline_state

    def post_data_collection(
        self,
        collected_data: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build context after data collection."""
        self._collected_data = collected_data

        # Build initial context from collected data
        if self.cag_pipeline.context_builder:
            zip_codes = self.cag_pipeline._extract_zip_codes({
                'collected_data': collected_data
            })

            context = self.cag_pipeline.context_builder.build_for_zip_codes(
                zip_codes=zip_codes,
                analysis_type="pipeline_integrated",
            )
            self.cag_pipeline._current_context = context
            self.logger.info("CAG context built from collected data")

        return collected_data

    def post_model_execution(
        self,
        model_results: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance model results with interpretation."""
        context = self.cag_pipeline._current_context

        if self.cag_pipeline.reality_interpreter and context:
            # Interpret each model's results
            for model_name, results in model_results.items():
                if isinstance(results, dict) and results:
                    try:
                        interp = self.cag_pipeline.reality_interpreter.interpret_model_results(
                            model_name,
                            results,
                            context
                        )

                        # Add interpretation to results
                        results['cag_interpretation'] = {
                            'reality_bridge': interp.lived_reality_bridge,
                            'anomalies': interp.anomalies_identified,
                        }
                    except Exception as e:
                        self.logger.warning(f"Failed to interpret {model_name}: {e}")

        return model_results

    def post_report_generation(
        self,
        reports: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance reports with CAG sections."""
        # Add CAG report as additional output
        if self.cag_pipeline._results:
            reports['cag_report'] = self.cag_pipeline.generate_cag_report()

        return reports

    def post_pipeline(
        self,
        pipeline_results: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run pattern discovery and blueprint generation."""
        context = self.cag_pipeline._current_context

        # Pattern discovery
        if self.cag_pipeline.pattern_discovery:
            data = {
                'model_results': pipeline_results.get('model_results', {}),
                'collected_data': self._collected_data,
            }
            discovery = self.cag_pipeline.pattern_discovery.process(data, context)
            self.cag_pipeline._results.append(discovery)

            pipeline_results['cag_patterns'] = discovery.discovered_patterns

        # Blueprint generation
        if self.cag_pipeline.blueprint_generator:
            blueprint_data = {
                'model_results': pipeline_results.get('model_results', {}),
                'discovered_patterns': self.cag_pipeline.pattern_discovery.discovered_patterns if self.cag_pipeline.pattern_discovery else [],
            }
            blueprints = self.cag_pipeline.blueprint_generator.process(blueprint_data, context)
            self.cag_pipeline._results.append(blueprints)

            pipeline_results['cag_blueprints'] = blueprints.suggested_investigations

        self.logger.info("CAG post-pipeline processing complete")
        return pipeline_results
