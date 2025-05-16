import pandas as pd
import stock_experiment_functions as sef
from datetime import datetime, timedelta
import os
import numpy as np
import random
from scipy.stats import rankdata
# pd.set_option('future.no_silent_downcasting', True)
def clean_currency_columns(data):
    for column in data.columns:
        # if data[column].dtype == 'object' and data[column].str.contains('\$').any():
        #     data[column] = data[column].replace('[\$,]', '', regex=True).astype(float)
        if data[column].dtype == 'object' and data[column].str.contains('\\$').any():
            data[column] = data[column].replace('[\\$,]', '', regex=True).astype(float)



def drop_object_columns(data):
    object_columns = [col for col in data.columns if col != data.columns[0] and data[col].dtype == 'object']
    data.drop(columns=object_columns, inplace=True)

def create_stock_features(df, target="Close", num_days_pred=30):
    """
    Create time series features based on time series index.
    Source: https://www.kaggle.com/code/zeyadsayedadbullah/stock-price-prediction-using-xgboost-prophet-arima
    """
    date = pd.to_datetime(df["Date"])
    # df['dayofweek'] = date.dt.dayofweek
    # df['quarter'] = date.dt.quarter
    # df['month'] = date.dt.month
    # df['year'] = date.dt.year
    # df['dayofyear'] = date.dt.dayofyear
    # df['dayofmonth'] = date.dt.day
    # df['weekofyear'] = date.dt.isocalendar().week.astype("int32")

    # Calculate the mean for the past 5, 30, 365 days
    df['day_5'] = df['Close'].rolling(5).mean().shift(1)
    df['day_30'] = df['Close'].rolling(30).mean().shift(1)
    df['day_365'] = df['Close'].rolling(365).mean().shift(1)

    # Calculate the STD for the past 5, 365 days
    df['std_5'] = df['Close'].rolling(5).std().shift(1)
    df['std_365'] = df['Close'].rolling(365).std().shift(1)

    return df

# Function to duplicate and shift columns
def duplicate_and_shift(df, column, n):
    new_data = {}
    for i in range(1, n + 1):
        new_column_name = f'{column}_shifted_{i}'
        new_data[new_column_name] = df[column].shift(-i).copy()

    shifted = pd.DataFrame(new_data)
    combined = pd.concat([df, shifted], axis=1)
    # Drop the initial rows where duplicated values are not meaningful
    combined = combined.dropna()

    return combined

# Export the partitions to csv files
def export_partitions_to_csv(partitions, file_path, file_name):
    for index, partition in enumerate(partitions):
        tmp_name = f"{file_name}_partition_{index}.csv"
        partition.to_csv(file_path+tmp_name, index=False)


def generate_partitions(data, partition_width, step):
    partitions = []

    for i in range(0, len(data), step):
        partition = data.iloc[i:i+partition_width]
        if len(partition) == partition_width:
            partitions.append(partition)

    return partitions

def generate_partitions_all_widths(data, partition_width, step):
    partitions = []

    for i in range(0, len(data), step):
        partition = data.iloc[i:i+partition_width]
        partitions.append(partition)

    return partitions

def generate_partitions_size_based(data, num_partitions):
    partitions = []

    # Assuming data is your DataFrame
    num_rows = len(data)
    step_size = num_rows // ((num_partitions - 1) * 2)
    partition_size = num_rows // 2

    for i in range(0, num_rows, step_size):
        partition = data.iloc[i:i+partition_size]
        if len(partition) == partition_size:
            partitions.append(partition)
    
    return partitions
    

def convert_to_datetime(date_str):
    return datetime.strptime(date_str, '%m/%d/%Y')


def calculate_distance_between_partitions_by_name(root, partition_1, partition_2):
    p1 = pd.read_csv("%s/%s" % (root, partition_1))
    p2 = pd.read_csv("%s/%s" % (root, partition_2))
    return calculate_distance_between_partitions(p1, p2)

def calculate_distance_between_partitions(partition_1, partition_2):
    # Convert 'Date' column to datetime type
    partition_1['Date'] = pd.to_datetime(partition_1['Date'], format='%Y-%m-%d')
    partition_2['Date'] = pd.to_datetime(partition_2['Date'], format='%Y-%m-%d')

    # Find the midpoint date in each partition
    midpoint_date_partition_1 = partition_1['Date'].min() + (partition_1['Date'].max() - partition_1['Date'].min()) / 2
    midpoint_date_partition_2 = partition_2['Date'].min() + (partition_2['Date'].max() - partition_2['Date'].min()) / 2

    # Calculate the difference between midpoints
    time_difference = abs(midpoint_date_partition_2 - midpoint_date_partition_1).days

    # Round to the nearest multiple of 10
    rounded_days = round(time_difference / 10) * 10

    return rounded_days

def calculate_distance_between_temp_partitions(partition_1, partition_2, i, j):
    # Calculate the midpoint within each CSV file
    
    # Calculate the index of the middle row(s)
    middle_index1 = (len(partition_1) * (i+1)) // 2

    # Calculate the index of the middle row(s)
    middle_index2 = (len(partition_2) * (j+1)) // 2

    # Calculate the difference between midpoints
    difference = abs(middle_index1 - middle_index2)

    return difference #, int(months_difference), int(years_difference)

def reformat_temp_date_information(partitions):
    new_partitions = []

    # Assuming df is your DataFrame
    # Example DataFrame
    for partition in partitions:
        # Create a 'Date' column by combining 'Year', 'Month', and 'Day'
        partition['Date'] = pd.to_datetime(partition[['Month', 'Day', 'Year']], errors='coerce')

        # Filter out rows where 'Date' is NaT (Not a Time) which indicates invalid date
        partition = partition.dropna(subset=['Date'])

        # Drop the 'Year', 'Month', and 'Day' columns if needed
        partition = partition.drop(columns=['Month', 'Day', 'Year'])

        new_partitions.append(partition)

    return new_partitions


def calculate_parititon_midpoint(partition_1):
    partition_1.loc[:, 'Date'] = pd.to_datetime(partition_1['Date'], format='%Y-%m-%d')

    # Calculate the midpoint within each CSV file
    
    # Calculate the index of the middle row(s)
    middle_index1 = len(partition_1) // 2
    # Get the middle row value(s)
    middle_value1 = partition_1['Date'].iloc[middle_index1]

    return middle_value1 #, int(months_difference), int(years_difference)


def calculate_distance_between_every_partition_combination(partitions):
    days_differences = []
    for i in range(0, len(partitions)):
        for j in range(i+1, len(partitions)):
            days_difference = calculate_distance_between_partitions(partitions[i], partitions[j])
            print(days_difference)
            days_differences.append(days_difference)

    return days_differences

def calculate_distance_between_every_temp_partition_combination(partitions):
    days_differences = []
    for i in range(0, len(partitions)):
        for j in range(i+1, len(partitions)):
            days_difference = calculate_distance_between_temp_partitions(partitions[i], partitions[j], i, j)
            print(days_difference)
            days_differences.append(days_difference)

    return days_differences

def run_cleaning_pipeline(data_partition_csv, target_column, output_path, task_type="regression", metric="mean_squared_error"):
   cmd = (
    f'python3 pipeline_main_currency.py '
    f'--train_dataset {data_partition_csv} '
    f'--target_column {target_column} '
    f'--metric {metric} '
    f'--output_path {output_path} '
    f'--task_type {task_type}'
    )
   #print(f"Running: {cmd}")
   try:
      os.system(cmd)
   except FileNotFoundError:
      print(f"Error: The file {data_partition_csv} does not exist.")


def run_missing_value_solver(data_partition_csv, target_column, output_path, task_type="regression", metric="mean_squared_error"):
   cmd = (
    f'python /nethome/araymaker3/transfer-cleaning/missing_values/main.py '
    f'--train_dataset {data_partition_csv} '
    f'--target_column {target_column} '
    f'--metric {metric} '
    f'--output_path {output_path} '
    f'--task_type {task_type}'
    )
   print(f"Running: {cmd}")
   try:
      os.system(cmd)
   except FileNotFoundError:
      print(f"Error: The file {data_partition_csv} does not exist.")


def replace_with_missing_value(df):
    # Iterate through each column
    for column in df.columns:
        # Check if the column name contains "AvgTemperature"
        if 'AvgTemperature' in column:
            # Print column before replacement
            
            # idx = df[column] == -99.0
            # if len(df.loc[df[column].isin([-99, -99.0])]) > 0:
            #     print(f"Column '{column}' before replacement:")
            #     print(df.loc[df[column].isin([-99, -99.0])])
            
            # Replace values equal to -99 or -99.0 with NaN
            df[column] = df[column].replace([-99, -99.0], pd.NA)
            
            # Print column after replacement
            # print(f"\nColumn '{column}' after replacement:")
            # print(df.loc[idx])
            # print("\n")

            # # Identify rows where values are -99 or -99.0
            # rows_to_replace = df[df[column].isin([-99, -99.0])].index
            
            # # Print rows before replacement
            # print(f"Rows in column '{column}' before replacement:")
            # print(df.loc[rows_to_replace, column])
            
            # # Replace values equal to -99 or -99.0 with NaN
            # df.loc[rows_to_replace, column] = pd.NA
            
            # # Print rows after replacement
            # print(f"\nRows in column '{column}' after replacement:")
            # print(df.loc[rows_to_replace, column])
            # print("\n")
    
    return df


def calculate_abs_difference_dataframe(root, partition_1, partition_2):
    p1 = pd.read_csv("%s/%s" % (root, partition_1))
    p2 = pd.read_csv("%s/%s" % (root, partition_2))
    methods = list(p1.columns[1:len(p1.columns)])
    models = list(p1.iloc[:, 0])
    results = {}
    eps = 1e-4
    for i, mdl in enumerate(models):
        results[mdl] = {}
        for j, m in enumerate(methods):
            if j + 1 < p2.shape[1]:
                # v1 = (p1.iloc[i,j + 1] + eps) / (p1.iloc[i,-1] + eps)
                # v2 = (p2.iloc[i,j + 1] + eps) / (p2.iloc[i,-1] + eps)
                v1 = p1.iloc[i,j + 1]
                v2 = p2.iloc[i,j + 1]
                # results[mdl][m] = v1 - v2
                # if (m == "Noclean"):
                #     print("Noclean: ", v1, v2)
                # else:
                #     print("Else: ", v1, v2)
                results[mdl][m] = (abs(v1 - v2) + eps) / (abs(p1.iloc[i,-1] - p2.iloc[i,-1]) + eps)
            else:
                print(partition_2, "Error")
                continue
    return results


def calculate_rankings(df):
    rankings = {}
    for i, row in df.iterrows():
        model = row.iloc[0]
        methods_scores = row.iloc[1:]
        if methods_scores.isna().any():
            methods_scores = methods_scores.fillna(methods_scores.max() + 1 + random.random())  # Fill NA with a high value to give them the lowest rank
            methods_scores = methods_scores.infer_objects(copy=False)
        methods_scores = methods_scores + random.random()
        ranked_methods = methods_scores.rank(ascending=True).astype(float)
        rankings[model] = ranked_methods
    return rankings



def calculate_rank_difference_dataframe(partition_1, partition_2, distance):
    df_temp_1 = partition_1.copy()
    df_temp_2 = partition_2.copy()

    # Drop the first column and insert it back as 'Model'
    df_temp_1 = df_temp_1.drop(df_temp_1.columns[0], axis=1)
    df_temp_1.insert(0, 'Model', partition_1.iloc[:, 0])

    df_temp_1 = df_temp_1.melt(id_vars=['Model'], var_name='Metric', value_name='MSE')

    # Calculate rank within each model using 'min' method for tied ranks
    df_temp_1['Rank'] = df_temp_1.groupby(['Model'])['MSE'].rank(ascending=True, method='min')

    # Repeat the same steps for partition_2
    df_temp_2 = df_temp_2.drop(df_temp_2.columns[0], axis=1)
    df_temp_2.insert(0, 'Model', partition_2.iloc[:, 0])

    df_temp_2 = df_temp_2.melt(id_vars=['Model'], var_name='Metric', value_name='MSE')

    # Calculate rank within each model using 'min' method for tied ranks
    df_temp_2['Rank'] = df_temp_2.groupby(['Model'])['MSE'].rank(ascending=True, method='min')

    # Calculate the absolute difference in ranks between partition_1 and partition_2
    rank_diff = df_temp_1['Rank'].sub(df_temp_2['Rank']).abs()

    # Add the rank difference to df_temp_1
    df_temp_1['Rank_Difference'] = rank_diff

    # Remove MSE and Rank columns as they're no longer needed
    df_temp_1 = df_temp_1.drop(columns=['MSE', 'Rank'])

    # Add the distance column to indicate the window distance
    df_temp_1['Distance'] = distance

    # Prepare the final result as a dictionary
    result = {}
    for _, row in df_temp_1.iterrows():
        model = row['Model']
        metric = row['Metric']
        rank_diff = row['Rank_Difference']
        
        if model not in result:
            result[model] = {}
        
        result[model][metric] = rank_diff

    return result



def calculate_performance_ratio_dataframe(partition_1, partition_2, experiment, distance):
    df_temp_1 = partition_1.copy()
    df_temp_2 = partition_2.copy()

    df_temp_1 = df_temp_1.drop(df_temp_1.columns[0], axis=1)
    # Add the first column back in
    df_temp_1.insert(0, 'Model', partition_1.iloc[:, 0])

    # print(df_temp_1)
    # Reshape df_diff to have a model column, a metric column, and a MSE column
    df_temp_1 = df_temp_1.melt(id_vars=['Model'], var_name='Metric', value_name='MSE')
    # Add a rank of the MSE column by ascending order

    df_temp_2 = df_temp_2.drop(df_temp_2.columns[0], axis=1)
    # Add the first column back in
    df_temp_2.insert(0, 'Model', partition_2.iloc[:, 0])

    # print(df_temp_1)
    # Reshape df_diff to have a model column, a metric column, and a MSE column
    df_temp_2 = df_temp_2.melt(id_vars=['Model'], var_name='Metric', value_name='MSE')
    # Add a rank of the MSE column by ascending order

    min_value = min(df_temp_1.loc[df_temp_1['MSE'] > 0, 'MSE'].min(), df_temp_2.loc[df_temp_2['MSE'] > 0, 'MSE'].min())
    min_value = min(min_value, 1e-3)

    rank_diff = (df_temp_2['MSE']+min_value).div(df_temp_1['MSE']+min_value)
    # Assuming df is your DataFrame and 'MSE' is the column you want to check

    # Absolute value of the differences
    # rank_diff = rank_diff.abs()

    # Add the rank difference column to df_diff
    df_temp_1['Performance_Ratio'] = rank_diff

    # Remove MSE and Rank columns
    df_temp_1 = df_temp_1.drop(columns=['MSE'])

    # Add column to df_diff that is the window distance
    df_temp_1['Distance'] = distance

    return df_temp_1


def midpoint_column_addition(midpoints, partition_results, df_diffs):

    for x in range(0, len(midpoints)):
        df_temp_1 = partition_results[x].copy()

        print(df_temp_1)

        df_temp_1 = df_temp_1.drop(df_temp_1.columns[0], axis=1)
        # Add the first column back in
        df_temp_1.insert(0, 'Model', partition_results[x].iloc[:, 0])

        # print(df_temp_1)
        # Reshape df_diff to have a model column, a metric column, and a MSE column
        df_temp_1 = df_temp_1.melt(id_vars=['Model'], var_name='Metric', value_name='MSE')
        # Add a rank of the MSE column by ascending order

        # Remove MSE and Rank columns
        # df_temp_1 = df_temp_1.drop(columns=['MSE'])

        df_temp_1['Midpoint'] = midpoints[x]

        print(df_temp_1)

        df_diffs.append(df_temp_1)

    return df_diffs

def df_scatterplot_between_every_partition_combination(days_differences, partitions, partition_results, df_diffs):
    difference_index = 0
    for i in range(0, len(partitions)):
        for j in range(i+1, len(partitions)):
            df_diff = sef.calculate_rank_difference_dataframe(partition_results[i], partition_results[j], 'financial_analysis', days_differences[difference_index])
            df_diffs.append(df_diff)
            difference_index+=1
    
    return df_diffs

def df_ratio_scatterplot_between_every_partition_combination(days_differences, partitions, partition_results, df_diffs):
    difference_index = 0
    for i in range(0, len(partitions)):
        for j in range(i+1, len(partitions)):
            df_diff = sef.calculate_performance_ratio_dataframe(partition_results[i], partition_results[j], 'financial_analysis', days_differences[difference_index])
            df_diffs.append(df_diff)
            difference_index+=1
    
    return df_diffs


def randomly_drop_data(df, target_columns, drop_ratio):
    if df.empty:
        print("DataFrame is empty")
    # Calculate number of rows to drop
    n_rows_to_drop = int(len(df) * drop_ratio)
    
    # For each target column, randomly select rows and set them to NaN
    for column in target_columns:
        # Randomly choose indices to drop
        indices_to_drop = np.random.choice(df.index.tolist(), n_rows_to_drop, replace=False)
        df.loc[indices_to_drop, column] = np.nan
    
    return df

def remove_extreme_values(df, target_columns, drop_ratio=0.03):
    if df.empty:
        print("DataFrame is empty")
        return df
    
    # For each target column, drop the largest and smallest 3% of values
    for column in target_columns:
        if column in df.columns:
            # Calculate the 3rd and 97th percentiles
            lower_bound = df[column].quantile(drop_ratio)
            upper_bound = df[column].quantile(1 - drop_ratio)
            
            # Set values outside the bounds to NaN
            df.loc[df[column] < lower_bound, column] = np.nan
            df.loc[df[column] > upper_bound, column] = np.nan
    
    return df