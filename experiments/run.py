#!/usr/bin/env python3
"""
Artificial Life Neuroevolution Experiments - Headless Mode Entry Point
"""

import argparse
import json
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path):
    """Load configuration from JSON file with path traversal protection."""
    # Resolve the path to absolute
    config_path = os.path.abspath(config_path)
    # Get project root (two levels up from this file: /experiments/run.py -> /experiments -> /project_root)
    project_root = os.path.dirname(os.dirname(os.path.abspath(__file__)))
    # Ensure the config file is within the project directory
    if not config_path.startswith(project_root):
        print('Error: Configuration file must be within the project directory')
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)

def main():
    """Main entry point for headless experiment mode."""
    parser = argparse.ArgumentParser(description='Run Artificial Life Neuroevolution Experiments (Headless)')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--seeds', type=int, required=True, help='Number of random seeds to run')
    parser.add_argument('--mode', type=str, choices=['gui', 'headless'], default='headless', help='Execution mode')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Compute device')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print(f"Starting Artificial Life Neuroevolution Experiments")
    print(f"Configuration: {args.config}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Seeds: {args.seeds}")
    print(f"Experiment: {config.get('experiment_name', 'unnamed')}")

    # TODO: Implement actual experiment logic
    # This will be filled in during subsequent development phases
    print("Experiment framework loaded. Implementation pending...")

    # Placeholder for actual experiment execution
    try:
        # Import and run experiments (to be implemented)
        # from experiments.experiment_runner import ExperimentRunner
        # runner = ExperimentRunner(config, args.seeds, args.device)
        # runner.run_all_seeds()
        print("Experiments would run here...")
    except ImportError as e:
        print(f"Note: Experiment modules not yet implemented: {e}")
        print("This is expected during initial skeleton creation.")
    except Exception as e:
        print(f"Error running experiments: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()