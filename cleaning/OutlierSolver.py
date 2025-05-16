import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats
import numpy as np


class OutlierSolver:

    def __init__(self, train_dataset, test_dataset, outlier_detection_method,
                 outlier_repair_method, target_column, train_test_split,
                 columns_to_ignore=None):

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.outlier_detection_method = outlier_detection_method
        self.outlier_repair_method = outlier_repair_method
        self.target_column = target_column
        self.train_test_split = train_test_split
        self.columns_to_ignore = columns_to_ignore

        # Check if path exists and load it into a pandas dataframe
        try:
            if isinstance(train_dataset, pd.DataFrame):
                self.train_df = train_dataset.copy()
                self.test_df = test_dataset.copy()
            else:
                self.train_df = pd.read_csv(train_dataset)
                if test_dataset is not None:
                    self.test_df = pd.read_csv(test_dataset)
                else:
                    # Split the train_dataset into test and train
                    self.test_df = self.train_df.sample(frac=train_test_split, random_state=42)
                    self.train_df = self.train_df.drop(self.test_df.index)
            self.df = pd.concat([self.train_df, self.test_df])
        except:
            print("File not found")
            exit(1)

        # Check if the target column exists
        if target_column not in self.df.columns:
            print("Target Column not found")
            exit(1)

    # Detect outliers on a dataset using IQR method is self.outlier_detection_method is 'iqr'
    # Detect cleaning on a dataset using Isolation Forest Method is self.outlier_detection_method is 'if'
    def detect_outliers_iqr_and_replace_with_stats(self):
        categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        for column in numerical_columns:
            if column != self.target_column and column not in self.columns_to_ignore:
                q1 = self.train_df[column].quantile(0.25)
                q3 = self.train_df[column].quantile(0.75)
                iqr = q3 - q1
                fence_low = q1 - (1.5 * iqr)
                fence_up = q3 + (1.5 * iqr)
                if self.outlier_repair_method == 'mean':
                    self.train_df[column] = self.train_df[column].apply(
                        lambda x: self.train_df[column].mean() if x < fence_low or x > fence_up else x)
                    self.test_df[column] = self.test_df[column].apply(
                        lambda x: self.train_df[column].mean() if x < fence_low or x > fence_up else x)
                elif self.outlier_repair_method == 'median':
                    self.train_df[column] = self.train_df[column].apply(
                        lambda x: self.train_df[column].median() if x < fence_low or x > fence_up else x)
                    self.test_df[column] = self.test_df[column].apply(
                        lambda x: self.train_df[column].median() if x < fence_low or x > fence_up else x)
                elif self.outlier_repair_method == 'mode':
                    col_mode = self.train_df[column].mode().iloc[0]
                    mask_train = ((self.train_df[column] < fence_low) | (self.train_df[column] > fence_up))

                    mask_test = (self.test_df[column] < fence_low) | (self.test_df[column] > fence_up)
                    self.train_df.loc[mask_train, column] = col_mode
                    self.test_df.loc[mask_test, column] = col_mode


                else:
                    print("Invalid Outlier Repair Method)")
                    exit(1)

    def detect_with_isolation_forest_and_replace_with_stats(self):
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        for column in numerical_columns:
            if column != self.target_column:
                clf = IsolationForest(random_state=42).fit(self.train_df[column].values.reshape(-1, 1))
                outliers = clf.predict(self.train_df[column].values.reshape(-1, 1))
                outliers = pd.Series(outliers).replace([-1, 1], [True, False])
                if self.outlier_repair_method == 'mean':
                    self.train_df[column] = self.train_df[column].apply(
                        lambda x: self.train_df[column].mean() if x in outliers else x)
                    self.test_df[column] = self.test_df[column].apply(
                        lambda x: self.train_df[column].mean() if x in outliers else x)
                elif self.outlier_repair_method == 'median':
                    self.train_df[column] = self.train_df[column].apply(
                        lambda x: self.train_df[column].median() if x in outliers else x)
                    self.test_df[column] = self.test_df[column].apply(
                        lambda x: self.train_df[column].median() if x in outliers else x)

                elif self.outlier_repair_method == 'mode':
                    col_mode = self.train_df[column].mode().iloc[0]
                    mask_train = self.train_df[column].isin(outliers)

                    self.train_df.loc[mask_train, column] = col_mode
                    # Replace outliers in test data
                    mask_test = self.test_df[column].isin(outliers)
                    self.test_df.loc[mask_test, column] = col_mode
                else:
                    print("Invalid Outlier Repair Method")
                    exit(1)

    def detect_with_zscore_and_replace_with_stats(self):
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        for column in numerical_columns:
            if column != self.target_column:

                # Predict outliers in train data
                z = np.abs(stats.zscore(self.train_df[column]))
                outliers = np.where(z > 0.5)
                outliers = outliers[0]

                # Predict outliers in test data using train data
                z_test = np.abs(stats.zscore(self.test_df[column]))
                outliers_test = np.where(z_test > 0.5)
                outliers_test = outliers_test[0]

                if self.outlier_repair_method == 'mean':
                    self.train_df[column] = self.train_df[column].apply(
                        lambda x: self.train_df[column].mean() if x in outliers else x)
                    self.test_df[column] = self.test_df[column].apply(
                        lambda x: self.train_df[column].mean() if x in outliers else x)
                elif self.outlier_repair_method == 'median':
                    self.train_df[column] = self.train_df[column].apply(
                        lambda x: self.train_df[column].median() if x in outliers else x)
                    self.test_df[column] = self.test_df[column].apply(
                        lambda x: self.train_df[column].median() if x in outliers else x)
                elif self.outlier_repair_method == 'mode':
                    col_mode = self.train_df[column].mode().iloc[0]
                    mask_train = self.train_df[column].isin(outliers)
                    self.train_df.loc[mask_train, column] = col_mode

                    # Replace outliers in test data
                    mask_test = self.test_df[column].isin(outliers)
                    self.test_df.loc[mask_test, column] = col_mode
                else:
                    print("Invalid Outlier Repair Method")
                    exit(1)


    def detect_and_repair(self):
        if self.outlier_detection_method == 'iqr':
            self.detect_outliers_iqr_and_replace_with_stats()
        elif self.outlier_detection_method == 'if':
            self.detect_with_isolation_forest_and_replace_with_stats()
        elif self.outlier_detection_method == 'zscore':
            self.detect_with_zscore_and_replace_with_stats()
        else:
            print("Invalid Outlier Detection Method")
            exit(1)





