from preprocessing_pipeline import PreprocessingPipeline

def main():
    """Main function to run the preprocessing pipeline."""
    # Initialize the pipeline
    pipeline = PreprocessingPipeline(data_directory="data")

    # Example: Single-file processing
    file_to_process = "data/session1.nwb"
    steps = ["calculate_hd_tuning_parameters", "compute_waveform_parameters"]
    print(f"Processing single file: {file_to_process}")
    pipeline.process_file(file_path=file_to_process, steps=steps)

    # Example: Batch processing using a YAML configuration
    config_path = "configs/B2904.yaml"
    print(f"Processing batch files using configuration: {config_path}")
    pipeline.process_from_yaml(config_path=config_path)

if __name__ == "__main__":
    main()
