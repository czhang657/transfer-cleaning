import os
import random
import pandas as pd
import stock_experiment_functions as sef

# Directory where the results (processed files) are stored
already_run_path = 'stock/results_none_partitions_1_percent/'

# Directory containing the stock CSV data files
stock_path = "stock/stock_data/"

# Directory to store the full processed files (formerly temp_results)
full_partitions_dir = "stock/stock_data_partitions_full"
os.makedirs(already_run_path, exist_ok=True)
os.makedirs(full_partitions_dir, exist_ok=True)

# Get the full paths of all CSV files in the stock data folder
stock_files = [os.path.join(stock_path, f) for f in os.listdir(stock_path) if f.endswith('.csv')]

def process_stock_direct(item, target_column="Close", task_type="regression", metric="mean_squared_error"):
    """
    Directly process a single stock file:
      1. Read the CSV file and convert the 'Date' column to datetime.
      2. Generate features using the create_stock_features function.
      3. Save the complete processed data to the full partitions directory.
      4. Run the cleaning pipeline on the processed data.
    """
    # Extract the stock name from the file name (before the first underscore)
    name = os.path.basename(item).split("_")[0]
    
    # Read the CSV file and convert the 'Date' column to datetime format
    data_file = pd.read_csv(item)
    data_file['Date'] = pd.to_datetime(data_file['Date'])
    
    num_days_pred = 30  # Number of days for prediction
    
    # Generate features using the stock_experiment_functions
    data = sef.create_stock_features(data_file, target_column, num_days_pred)
    data = data.iloc[365:].reset_index(drop=True)
    # Save the processed complete data to the full partitions directory
    temp_file = os.path.join(full_partitions_dir, f"{name}_full.csv")
    data.to_csv(temp_file, index=False)
    
    # Construct the output path for the final results
    output_path = os.path.join(already_run_path, f"{name}_full.csv")
    
    # Check if the result already exists to avoid duplicate processing
    already_run = [os.path.join(already_run_path, f) for f in os.listdir(already_run_path) if f.endswith('.csv')]
    if output_path not in already_run:
        sef.run_cleaning_pipeline(temp_file, target_column, output_path, task_type, metric)
    
    return name

if __name__ == "__main__":
    # Retrieve the list of already processed result files
    already_run = [os.path.join(already_run_path, f) for f in os.listdir(already_run_path) if f.endswith('.csv')]
    
    # Reverse the order of the stock files list
    stock_files.reverse()
    random.seed(42)
    
    # If there are too many stock files, randomly sample 200 for processing
    sample_size = 1000
    if len(stock_files) > sample_size:
        stock_files = random.sample(stock_files, sample_size)
    # stock_files = ["stock/stock_data/ACRS_historical_data.csv"]
    count = 0
    for s in stock_files:
        count += 1
        print("Stock Number: ", count, s)
        process_stock_direct(s)
