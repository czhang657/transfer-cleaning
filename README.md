# Transfer Cleaning: Measuring Transferability of Automated Data Cleaning

This repository contains the code, data pipeline, and experiments for our project on evaluating the transferability of automated data cleaning pipelines. The experiments focus on financial time series data and explore how cleaning strategies generalize across datasets with different temporal, ontological, and quality characteristics.

## Setup

### 1. Install Dependencies

Ensure you are using Python 3.8+ and install required packages with:

```
pip install -r requirements.txt
```

## Data Download

Download the raw financial data using the following scripts from the repository root:

```
python data_downloader/download_stock_data.py
python data_downloader/download_currency_data.py
```

This will create the following directories:

- `default_setup_transfer_cleaning/stock/stock_data/`
- `default_setup_transfer_cleaning/currency/currency_data/`

## Running Experiments

### A. Default Cleaning Setup

Run the baseline experiments from the `default_setup_transfer_cleaning` directory:

```
cd default_setup_transfer_cleaning

# Run stock dataset experiment
python stock_experiment.py

# Run currency dataset experiment
python currency_experiment.py
```

### B. DiffPrep Cleaning Setup

To run experiments with DiffPrep-based automated pipeline selection:

1. From the repository root, move the stock data into the DiffPrep input folder:

```
python move_data.py
```

2. Then execute the experiment:

```
cd DiffPrep_transfer_cleaning
python run.py
```

## Project Status

This repository accompanies a research project currently under submission. We welcome feedback and suggestions as we continue to improve the framework and expand its applications.
