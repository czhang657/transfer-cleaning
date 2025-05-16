from cleaning.OutlierSolver import OutlierSolver
from cleaning.MissingValueSolverCurrency import MissingValueSolverCurrency
import argparse
import pandas as pd
from train.Trainer import Trainer
import stock_experiment_functions as sef
from sklearn.preprocessing import StandardScaler
import os


def run_pipeline(ml_model):
    for impute_method in impute_method_numerical:
        # Fix Missing Values
        mv = MissingValueSolverCurrency(args.train_dataset, args.test_dataset, args.drop_threshold,
                                impute_method, "", args.target_column,
                                args.train_test_split, [])
        mv.detect_and_repair()
        scaler = StandardScaler()
        columns_to_scale = [col for col in mv.train_df.columns if col != "Date"]
        mv.train_df[columns_to_scale] = scaler.fit_transform(mv.train_df[columns_to_scale])
        mv.test_df[columns_to_scale] = scaler.transform(mv.test_df[columns_to_scale])
        # No clean with missing values fixed
        outlier_cleaning = OutlierSolver(mv.train_df, mv.test_df, "", "",
                        args.target_column, args.train_test_split)
        tr = Trainer(args.task_type, args.target_column, ml_model, args.metric, outlier_cleaning.train_df, outlier_cleaning.test_df)
        result = tr.ml_pipeline(drop_mv=False, std_scale=False)
        # Merge results
        key = impute_method + "_noclean"
        if not key in result_dict:
            result_dict[key] = {}
        for model in result:
            result_dict[key][model] = result[model]
        
    # Drop Missing Values
    mv = MissingValueSolverCurrency(args.train_dataset, args.test_dataset, args.drop_threshold,
                            "", "", args.target_column,
                            args.train_test_split, [])
    mv.detect_and_repair()
    # No clean with missing values fixed
    outlier_cleaning = OutlierSolver(mv.train_df, mv.test_df, "", "",
                        args.target_column, args.train_test_split)
    scaler = StandardScaler()
    columns_to_scale = [col for col in mv.train_df.columns if col != "Date"]
    outlier_cleaning.train_df[columns_to_scale] = scaler.fit_transform(outlier_cleaning.train_df[columns_to_scale])
    outlier_cleaning.test_df[columns_to_scale] = scaler.transform(outlier_cleaning.test_df[columns_to_scale])
    tr = Trainer(args.task_type, args.target_column, ml_model, args.metric, outlier_cleaning.train_df, outlier_cleaning.test_df)
    result = tr.ml_pipeline(drop_mv=False, std_scale=False)
    # Merge results
    key = "Noclean"
    if not key in result_dict:
        result_dict[key] = {}
    for model in result:
        result_dict[key][model] = result[model]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Functions to Detect Outliers on a Dataset and Amend Them')
    parser.add_argument('--train_dataset', required = True, type=str, help='Path to the train dataset')
    parser.add_argument('--test_dataset', required = False, default=None, type=str, help='Path to the test dataset')
    parser.add_argument('--drop_threshold', required = False, type=float, default=0.5, help='Percentage Threshold to drop columns with missing values')
    parser.add_argument('--outlier_detection_method', required = False, type=str, default='if', help='Method to detect outliers')
    parser.add_argument('--outlier_repair_method', required = False, type=str, default='mean', help='Method to repair outliers')
    parser.add_argument('--ml_model', required = False, type=str, default='logistic_regression', help='ML Model to use for classification')
    parser.add_argument('--target_column', required = True, type=str, help='Target Column for Classification')
    parser.add_argument('--metric', required = False, type=str, default='f1_score', help='Metric to assess the performance of the ML Model')
    parser.add_argument('--output_path', required = True, type=str, help='Path to store the output file with all results')
    parser.add_argument('--train_test_split', required = False, type=float, default=0.2, help='Train Test Split for the dataset')
    parser.add_argument('--feature_importance', required = False, type=str, default='logistic_regression', help='Whether to print feature importance or not')
    parser.add_argument('--columns_to_drop', required = False, type=str, default=None, help='Columns to drop from the dataset')
    parser.add_argument('--task_type', required = False, type=str, default='classification', help='Classification or Regression')

    args = parser.parse_args()
    # Solve for cleaning
    ol_detection_methods = ['iqr', 'if', 'zscore']
    ol_repair_methods = ['mean', 'median']
    # ol_repair_methods = ["mean"]
    impute_method_numerical = ["mean", "median", "mode", "knn", "mice"] #, "median", "mode", "knn", "mice"
    impute_method_categorical = []
    result_dict = {}
    for model in ["ridge"]:
        run_pipeline(model)

    result_dict = pd.DataFrame(result_dict)
    # Windows does not allow ':' in file names
    path = f"{args.output_path}".replace(":", "_")
    result_dict.to_csv(path)
    print(path)


