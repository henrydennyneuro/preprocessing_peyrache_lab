import argparse
import sys
from pipeline import PreprocessingPipeline

def main():
    # Piped output (e.g. "| Tee-Object") uses the Windows ANSI code page, which has no
    # characters like the progress tick; print a replacement instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    pipeline = PreprocessingPipeline()
    pipeline.process_from_yaml(args.config)

if __name__ == "__main__":
    main()
