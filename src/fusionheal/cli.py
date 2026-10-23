import argparse
import json
from .pipeline import run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FusionHeal prototype pipeline.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs")
    args = parser.parse_args()
    summary = run_pipeline(args.input, args.config, args.output_dir)
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
