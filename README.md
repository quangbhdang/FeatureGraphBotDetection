# Graph Feature Bot Prediction In Social Media Network

A repo for group work on bot detection research and implementation

# Research question:

Since LLM bots have become much more challenging to identify within social media networks. Our research will focus on identifying LLM-driven bots within social media networks. The practical application of this is spam detection, scam prevention and reducing coordinated media manipulation using LLM agentic bots.

# For the dataset

You can obtain the dataset for InstaFake at the author Github repo [https://github.com/fcakyon/instafake-dataset]. Please ensure you place unzip version of both subdirectory of the dataset in the raw folder of InstaFake.

For the TwiBot-20 you can follow the guide from the dataset author [https://github.com/BunsenFeng/TwiBot-20] to obtain access to the data. Place the zip file in the raw folder as shown in the project folder of this repo.

# How to run

This project uses Make and DVC for reproducible data science workflows. Follow these steps to set up and run the bot detection pipeline.

## Prerequisites

- **Conda** (Anaconda or Miniconda)
- **Git**
- **DVC** (Data Version Control)

The dependencies and virtual environment is control by conda to ensure simple replication of the finding results.

## Quick Start

### 1. Clone and Setup Environment
```bash
# Clone the repository
git clone <repository-url>
cd sm_graph_features

# Create conda environment and initialize DVC
make recreate-env
```

### 2. Activate Environment
```bash
conda activate graph_sm
```

### 3. Set up DVC Remote (for data sharing)
```bash
# Check if remote is already configured
dvc remote list

# If no remote, set up DagsHub remote (already configured in .dvc/config)
dvc remote default origin
```

### 4. Run Data Preprocessing Pipeline
```bash
# Set up the preprocessing pipeline
make setup-dvc-preprocess-pipeline

# Run preprocessing (converts raw JSON to combined CSV)
make run-preprocess-instafake
```

### 5. Create Personal Notebook
```bash
# Creates a notebook with your username
# Use this notebook to experiments or rapidly prototype with the project
make create-user-notebook

# Your notebook will be at: Notebooks/<your-username>.ipynb
```

### 6. Train Model (when ready)
```bash
make train
```

## Available Make Commands

| Command | Description |
|---------|-------------|
| `make recreate-env` | Create conda environment from provided `environment.yml`|
| `make create-user-notebook` | Create a personal Jupyter notebook |
| `make setup-dvc-preprocess-pipeline` | Set up DVC pipeline for data preprocessing |
| `make run-preprocess-instafake` | Run the preprocessing pipeline |
| `make train` | Train the bot detection model |

## Project Structure

```
sm_graph_features/
├── Dataset/
│   └── InstaFake/
│       ├── raw/                     # Raw JSON data files
│       └── interim/                 # Processed CSV files
├── Src/
│   ├── preprocess_instafake.py     # Data preprocessing script
│   └── train_model.py              # Model training script
├── Notebooks/                      # Personal analysis notebooks
├── Makefile                        # Automation commands
├── environment.yml                 # Conda environment
├── dvc.yaml                        # DVC pipeline definition
└── README.md                       # This file
```

## Data Pipeline Workflow

1. **Raw Data**: JSON files with fake and real Instagram account data
2. **Preprocessing**: Combines and cleans data into a single CSV
3. **Training**: Trains ML models for bot detection
4. **Evaluation**: Tests model performance

## DVC Data Management

The project uses DVC to track data and ensure reproducibility:

```bash
# Check pipeline status
dvc status

# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull

# Reproduce entire pipeline
dvc repro
```

## Development Workflow

For contributors working on the project:

```bash
# 1. Set up environment
make recreate-env
conda activate graph_sm

# 2. Create your personal notebook
make create-user-notebook

# 3. Run preprocessing if data changed
make run-preprocess-instafake

# 4. Work in your notebook
jupyter lab Notebooks/<your-username>.ipynb

# 5. Commit changes
git add .
git commit -m "Your changes"
git push
```

## Troubleshooting

### Environment Issues
```bash
# Remove and recreate environment
conda env remove -n graph_sm
make recreate-env
```

### DVC Issues
```bash
# Check DVC status
dvc status
dvc remote list

# Reset DVC pipeline
rm dvc.yaml dvc.lock
make setup-dvc-preprocess-pipeline
```

### Data Access Issues
```bash
# Ensure you have access to the DagsHub remote
dvc pull

# Check if preprocessing generated output
ls -la Dataset/InstaFake/interim/
```

## Dependencies

All dependencies are managed through `environment.yml`:
- Python 3.11
- PyTorch + PyTorch Geometric
- Pandas, NumPy, Scikit-learn
- Jupyter, Matplotlib
- DVC, MLflow

## Contributing

1. Create your personal notebook: `make create-user-notebook`
2. Work on your analysis in `Notebooks/<your-username>.ipynb`
3. Update code in `Src/` directory
4. Test with: `make run-preprocess-instafake`
5. Commit and push changes

## Research Focus

This project investigates LLM-driven bot detection in social media networks, focusing on:
- Identifying sophisticated AI-generated content
- Graph-based feature extraction
- Behavioral pattern analysis
- Real-time detection capabilities

