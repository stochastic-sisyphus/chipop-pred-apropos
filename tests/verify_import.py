"""
Test script to verify the Pipeline class import path.
"""

import sys
import os
import importlib
import inspect

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Print sys.path
print("sys.path:")
for path in sys.path:
    print(f"  - {path}")

# Try to import Pipeline
print("\nTrying to import Pipeline...")
try:
    from src.pipeline import Pipeline
    print(f"Successfully imported Pipeline from {Pipeline.__module__}")
    print(f"Pipeline class defined in: {inspect.getfile(Pipeline)}")
    print(f"Pipeline.__init__ signature: {inspect.signature(Pipeline.__init__)}")
    print(f"Pipeline methods: {[method for method in dir(Pipeline) if not method.startswith('_')]}")
except Exception as e:
    print(f"Error importing Pipeline: {str(e)}")

# Try to import from specific module
print("\nTrying to import Pipeline from specific module...")
try:
    from src.pipeline.pipeline import Pipeline as PipelineSpecific
    print(f"Successfully imported Pipeline from src.pipeline.pipeline")
    print(f"Pipeline class defined in: {inspect.getfile(PipelineSpecific)}")
    print(f"Pipeline.__init__ signature: {inspect.signature(PipelineSpecific.__init__)}")
    print(f"Pipeline methods: {[method for method in dir(PipelineSpecific) if not method.startswith('_')]}")
except Exception as e:
    print(f"Error importing Pipeline from src.pipeline.pipeline: {str(e)}")

# Check if src/pipeline/__init__.py exists
init_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'pipeline', '__init__.py')
print(f"\nChecking if {init_path} exists: {os.path.exists(init_path)}")

# Check if src/pipeline.py exists
pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'pipeline.py')
print(f"Checking if {pipeline_path} exists: {os.path.exists(pipeline_path)}")

# Check if src/pipeline/pipeline.py exists
pipeline_module_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'pipeline', 'pipeline.py')
print(f"Checking if {pipeline_module_path} exists: {os.path.exists(pipeline_module_path)}")

# List all files in src/pipeline
pipeline_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'pipeline')
print(f"\nListing files in {pipeline_dir}:")
if os.path.exists(pipeline_dir) and os.path.isdir(pipeline_dir):
    for file in os.listdir(pipeline_dir):
        print(f"  - {file}")
else:
    print("  Directory does not exist")

# Check module cache
print("\nChecking sys.modules for pipeline modules:")
for module in sys.modules:
    if 'pipeline' in module:
        print(f"  - {module}")

# Try to directly load the module
print("\nTrying to directly load the pipeline module:")
try:
    spec = importlib.util.spec_from_file_location("pipeline", pipeline_module_path)
    if spec:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'Pipeline'):
            print(f"Successfully loaded Pipeline from {pipeline_module_path}")
            print(f"Pipeline.__init__ signature: {inspect.signature(module.Pipeline.__init__)}")
        else:
            print(f"Module loaded but does not contain Pipeline class")
    else:
        print(f"Could not create spec for {pipeline_module_path}")
except Exception as e:
    print(f"Error loading module: {str(e)}")
