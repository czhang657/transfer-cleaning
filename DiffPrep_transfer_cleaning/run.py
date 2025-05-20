import os
import json
import pandas as pd
import numpy as np
import subprocess
import stock_experiment_functions as sef  # Make sure this module path is correct

# === Global Paths ===
src_folder = "data"  # Input folder containing {stock_code}_data/data.csv for each stock
result_root = "result/stock_runs/"
generate_dataset_dir="stock"
os.makedirs(generate_dataset_dir, exist_ok=True)
os.makedirs(result_root, exist_ok=True)

# Get all stock folders under the input directory
stock_folders = [f for f in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, f))]
print(stock_folders)

# Iterate over each stock
for stock_folder in stock_folders:
    stock_code = stock_folder.replace("_data", "")
    dataset_name = f"{stock_code}_full"
    default_output_dir = os.path.join(result_root, "diffprep_fix", dataset_name)

    # Skip if result already exists
    if os.path.exists(default_output_dir):
        print(f"⏭️ Skipping {stock_code}: result already exists.")
        continue

    # Load the stock data
    data_path = os.path.join(src_folder, stock_folder, "data.csv")
    df = pd.read_csv(data_path)

    # Skip if no 'Date' column
    if "Date" not in df.columns:
        print(f"⚠️ Skipping {stock_code}: 'Date' column not found.")
        continue

    print(f"\n========== Processing {stock_code} ==========")
    df['Date'] = pd.to_datetime(df['Date'])
    df = sef.create_stock_features(df, target="Close", num_days_pred=30)

    # Remove first 365 rows to avoid NaNs in day365 and std365
    df = df.iloc[365:].reset_index(drop=True)

    # Create dataset folder
    dataset_dir = os.path.join("stock", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Randomly inject 1% missing values (excluding target and Date columns)
    np.random.seed(42)
    target_col = "Close"
    feature_cols = [col for col in df.columns if col not in [target_col, "Date"]]
    mask = np.random.rand(*df[feature_cols].shape) < 0.01
    df = df.copy()
    df[feature_cols] = df[feature_cols].mask(mask)

    # Save corrupted data and info.json
    df.to_csv(os.path.join(dataset_dir, "data.csv"), index=False)
    info = {
        "label": target_col,
        "drop_variables": [],
        "categorical_variables": []
    }
    with open(os.path.join(dataset_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # Run DiffPrep
    cmd = (
        f"python main.py "
        f"--dataset {dataset_name} "
        f"--data_dir stock "
        f"--result_dir {result_root} "
        f"--model log "
        f"--method diffprep_fix "
        f"--train_seed 1 "
        f"--split_seed 1"
    )
    print(f"🛠️ Running: {cmd}")
    subprocess.run(cmd, shell=True)

    # Check if result exists
    if os.path.exists(default_output_dir):
        print(f"✅ Saved result to {default_output_dir}")
    else:
        print(f"⚠️  Warning: No result found in {default_output_dir}")

print("\n🎉 All stock experiments completed.")
