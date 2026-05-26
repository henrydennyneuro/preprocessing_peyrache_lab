import argparse
from pipeline import PreprocessingPipeline

def main():
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    pipeline = PreprocessingPipeline()
    pipeline.process_from_yaml(args.config)

if __name__ == "__main__":
    main()
