from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
import stock_experiment_functions as sef
import os
import random

# Get a list of symbols already run
already_run_path = 'stock/results/'  # Replace with the actual path to your folder

# Average all of the dataframes in temp_results folder
stock_path = "stock/stock_data/"  # Replace with the actual path to your folder
part_path = "stock/stock_data_partitions/"

# Get a list of all files in the folder
stock_files = [stock_path + f for f in os.listdir(stock_path) if f.endswith('.csv')]


def generate_partitions(item, partition_width=1400, step=200):
    # Extract the desired substring, which is the company/stock name
    parts = item.split('/')
    name = parts[-1].split("_")[0]

    data_file = pd.read_csv(item)

    # Convert the 'Date' column to datetime format using pandas
    data_file['Date'] = pd.to_datetime(data_file['Date'])

    num_days_pred = 30

    # Duplicate and shift the 'close' column
    data = sef.create_stock_features(data_file, 'Close', num_days_pred)

    partitions = sef.generate_partitions(data, partition_width, step)

    # Number of partitions present
    print(name, len(partitions))

    sef.export_partitions_to_csv(partitions, part_path, name)
    return name, partitions

def stock_processing(partitions, name, target_column="Close", task_type="regression", metric="mean_squared_error"):
    already_run = [already_run_path + f for f in os.listdir(already_run_path) if f.endswith('.csv')]
    for index, partition in enumerate(partitions):
        file_name = f"{part_path}/{name}_partition_{index}.csv"
        output_path = f"stock/results/{name}_partition_{index}.csv"
        if not output_path in already_run:
            sef.run_cleaning_pipeline(file_name, target_column, output_path, task_type, metric)

if __name__ == "__main__":
    already_run = [already_run_path + f for f in os.listdir(already_run_path) if f.endswith('.csv')]
    # Extract all stock names
    stocks = set([a.split("/")[-1].split("_")[0] for a in already_run])
    # Reverse the order of the list
    stock_files.reverse()
    random.seed(42)

    # Randomly sample 1000 stocks
    sample_size = 200
    if len(stock_files) > sample_size:
        stock_files = random.sample(stock_files, sample_size)
    count = 0
    for s in stock_files:
        count += 1
        print("Stock Number: ", count)
        name = s.split('/')[-1].split("_")[0]
        name, parts = generate_partitions(s)
        stock_processing(parts, name)