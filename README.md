# Preprocessing Pipeline for Peyrache Lab Data

This repository contains a Python-based preprocessing pipeline for extracting and analyzing features from neural recordings. It supports both single-file and batch processing workflows, with configuration managed via YAML files for flexibility and reusability.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Processing a Single File](#processing-a-single-file)
  - [Processing a Batch of Files](#processing-a-batch-of-files)
- [Directory Structure](#directory-structure)
- [Configuration Files](#configuration-files)
<!--
- [Contributing](#contributing)
-->
- [License](#license)

---

## Overview

The preprocessing pipeline is designed to:
- Compute various neural data parameters, such as head direction tuning and waveform properties.
- Detect oscillatory patterns in recordings.
- Support modular processing workflows by allowing users to select preprocessing steps.
- Handle both individual recordings and batch processing via YAML configuration files.

---

## Installation

### Prerequisites

- Python 3.8 or later
- Conda (recommended for environment management)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/henrydennyneuro/preprocessing_peyrache_lab.git
   cd preprocessing_peyrache_lab
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate preprocessing_env
   ```

3. Install additional dependencies (if needed):
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Processing a Single File

To process a single recording, specify the file path and desired preprocessing steps:

```python
from preprocessing_pipeline import PreprocessingPipeline

# Initialize the pipeline with the directory containing data files
pipeline = PreprocessingPipeline(data_directory="path_to_data_directory")

# Specify the file to process and the steps to apply
file_to_process = "data/session1.nwb"
steps = ["calculate_hd_tuning_parameters", "compute_waveform_parameters"]

# Process the single file
pipeline.process_file(file_path=file_to_process, steps=steps)
```

### Processing a Batch of Files

Batch processing is managed using YAML configuration files. Specify the files and steps in a YAML file (e.g., `configs/B2904.yaml`):

**Example YAML File:**
```yaml
files:
  - data/session1.nwb
  - data/session2.nwb
  - data/session3.nwb
steps:
  - calculate_hd_tuning_parameters
  - compute_waveform_parameters
```

**Run Batch Processing:**

```python
from preprocessing_pipeline import PreprocessingPipeline

# Initialize the pipeline with the directory containing data files
pipeline = PreprocessingPipeline(data_directory="path_to_data_directory")

# Process all files specified in the YAML configuration
pipeline.process_from_yaml(config_path="configs/B2904.yaml")
```

---

## Directory Structure

Here is the recommended directory structure for the project:

```
preprocessing_peyrache_lab/
│
├── configs/                      # Directory for configuration files
│   └── B2904.yaml
│
├── preprocessing_pipeline/       # Directory for Python source code
│   ├── __init__.py               # Makes this a Python package
│   └── pipeline.py               # Refactored class-based preprocessing pipeline
│
├── tests/                        # Directory for testing the pipeline
│   ├── __init__.py
│   └── test_pipeline.py          # Unit tests for pipeline functionality
│
├── environment.yml               # Conda environment configuration
├── .gitignore                    # Ignore unnecessary files in version control
├── README.md                     # Documentation for the project
└── LICENSE                       # Project license
```

---

## Configuration Files

Configuration files use the YAML format for specifying batch processing workflows. Example:

```yaml
files:
  - data/session1.nwb
  - data/session2.nwb
  - data/session3.nwb
steps:
  - calculate_hd_tuning_parameters
  - compute_waveform_parameters
```

Store these files in the `configs/` directory.

---
<!-- This section is hidden and will not appear in the rendered README
This text will not be shown in the rendered README.

## Contributing

We welcome contributions to improve this project! To contribute:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature_branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature_branch
   ```
5. Create a pull request.

-->
---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.