import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class MissingValueSolverCurrency:

    # All arguments from argparse in main should be passed to the constructor
    def __init__(self, train_dataset, test_dataset, drop_threshold, impute_method_numerical,
                 impute_method_categorical, target_column,
                 train_test_split, columns_to_ignore,
                 percentage_select=1):

        # Check if Impute Method Numerical is valid. It should be either mean, mode or median
        if impute_method_numerical not in ['mean', 'mode', 'median', 'knn', 'mice', 'xgboost', ""]:
            print("Invalid Impute Method Numerical. It should be either mean, mode, median, knn")
            exit(1)

        # Check if Impute Method Categorical is valid. It should be only mode
        if impute_method_categorical not in ['mode', "dummy", ""]:
            print("Invalid Impute Method Categorical. It should be only mode")
            exit(1)

        # Check if path exists and load it into a pandas dataframe
        try:
            if isinstance(train_dataset, pd.DataFrame):
                self.train_df = train_dataset.copy()
                self.test_df = test_dataset.copy()
            else:
                self.train_df = pd.read_csv(train_dataset)
                if percentage_select < 1:
                    self.train_df = self.train_df.sample(frac=percentage_select, random_state=42)

                if test_dataset is not None:
                    self.test_df = pd.read_csv(test_dataset)
                else:
                    # Split the train_dataset into test and train
                    self.test_df = self.train_df.sample(frac=train_test_split, random_state=42)
                    self.train_df = self.train_df.drop(self.test_df.index)
                    self.train_df = self.randomly_drop_data(self.train_df, ["Open","High","Low","Adj Close","Volume","day_5","day_30","day_365","std_5","std_365"], 0.01)
            self.df = pd.concat([self.train_df, self.test_df])
        except:
            print("File not found")
            exit(1)

        # Check numerical columns that have 0 missing values
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Check if target column exists in the dataset
        if target_column not in self.df.columns.tolist():
            print("Target Column not found in the dataset")
            exit(1)

        self.drop_threshold = drop_threshold
        self.columns_to_ignore = columns_to_ignore
        self.impute_method_numerical = impute_method_numerical
        self.impute_method_categorical = impute_method_categorical
        self.target_column = target_column

        self.categorical_impute_list = ['mode', 'dummy']
        self.numerical_impute_list = ['mean', 'mode', 'median', 'knn', 'mice']

        self.numerical_impute_stats = {}
        self.categorical_impute_stats = {}

        for numerical_impute in self.numerical_impute_list:
            self.numerical_impute_stats[numerical_impute] = {}

        for categorical_impute in self.categorical_impute_list:
            self.categorical_impute_stats[categorical_impute] = {}

    # Function to detect missing values in the dataset and show all the columns with missing values and their
    # corresponding percentage of missing values
    def detect_missing_values(self, dataset):

        # Find percentage of missing values in each column
        missing_values_percentage = dataset.isnull().sum() / dataset.shape[0] * 100

        # Print all columns with missing values and their corresponding percentage of missing values
        # print(missing_values_percentage[missing_values_percentage > 0])

    def print_missing_value_stats(self):
        categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        return categorical_columns, numerical_columns

    # Function to drop all columns with missing values above a certain threshold
    def drop_missing_values(self):

        # Figure out columns having more than drop_threshold percentage missing values
        missing_percentage = self.df.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage > self.drop_threshold].index.tolist()

        # Drop these columns in both train and test dataset
        self.train_df = self.train_df.drop(columns_to_drop, axis=1)
        self.test_df = self.test_df.drop(columns_to_drop, axis=1)
        self.df = self.df.drop(columns_to_drop, axis=1)

        # Drop the rows from tran and test dataset which have missing values in target column
        self.train_df = self.train_df.dropna(subset=[self.target_column])
        self.test_df = self.test_df.dropna(subset=[self.target_column])
        self.df = self.df.dropna(subset=[self.target_column])

        # Drop columns in train and test dataset which are categorical and have more than 500 unique values
        categorical_columns, _ = self.print_missing_value_stats()
        print(self.df[categorical_columns].nunique() > 500)
        for col in categorical_columns:
            if self.df[col].nunique() > 500:
                self.train_df = self.train_df.drop(col, axis=1)
                self.test_df = self.test_df.drop(col, axis=1)
                self.df = self.df.drop(col, axis=1)

    # Function to impute missing values in numerical columns using self.impute_method_numerical.
    # If it is equal to mean, then mean should be used to impute missing values.
    # If it is equal to mode, then mode should be used to impute missing values.
    # If it is equal to median, then median should be used to impute missing values.
    def impute_missing_values_numerical(self):
        # Select All the Numerical Columns
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Use Mean method to impute values in numerical columns
        if self.impute_method_numerical == 'mean':
            for col in numerical_columns:
                mean_value = self.train_df[col].mean()
                self.put_imputation_stats(col, 'mean', mean_value)
                self.train_df[col] = self.train_df[col].fillna(mean_value)
                self.test_df[col] = self.test_df[col].fillna(mean_value)

        # Use Mode method to impute values in numerical columns
        elif self.impute_method_numerical == 'mode':
            for col in numerical_columns:
                mode_value = self.train_df[col].mode()[0]
                self.put_imputation_stats(col, 'mode', mode_value)
                self.train_df[col] = self.train_df[col].fillna(mode_value)
                self.test_df[col] = self.test_df[col].fillna(mode_value)

        # Use Median method to impute values in numerical columns
        elif self.impute_method_numerical == 'median':
            for col in numerical_columns:
                median_value = self.train_df[col].median()
                self.put_imputation_stats(col, 'median', median_value)
                self.train_df[col] = self.train_df[col].fillna(median_value)
                self.test_df[col] = self.test_df[col].fillna(median_value)

        elif self.impute_method_numerical == "knn":

            df_numerical_train = self.train_df[numerical_columns]
            df_numerical_test = self.test_df[numerical_columns]

            imputer = KNNImputer(n_neighbors=5)
            df_numerical_train_imputed = imputer.fit_transform(df_numerical_train)
            df_numerical_test_imputed = imputer.transform(df_numerical_test)

            df_numerical_train_imputed = pd.DataFrame(df_numerical_train_imputed,
                                             columns=numerical_columns, index=self.train_df.index)
            df_numerical_test_imputed = pd.DataFrame(df_numerical_test_imputed,
                                             columns=numerical_columns, index=self.test_df.index)
            self.train_df[numerical_columns] = df_numerical_train_imputed
            self.test_df[numerical_columns] = df_numerical_test_imputed

        elif self.impute_method_numerical == "mice":
            df_numerical_train = self.train_df[numerical_columns]
            df_numerical_test = self.test_df[numerical_columns]
            imputer = IterativeImputer(max_iter=1000, random_state=0, estimator=KNeighborsRegressor(n_neighbors=5), initial_strategy="mean")

            # imputer = IterativeImputer(max_iter=1000, random_state=0)
            df_numerical_train_imputed = imputer.fit_transform(df_numerical_train)
            df_numerical_test_imputed = imputer.transform(df_numerical_test)

            df_numerical_train_imputed = pd.DataFrame(df_numerical_train_imputed,
                                    columns=numerical_columns, index=self.train_df.index)
            df_numerical_test_imputed = pd.DataFrame(df_numerical_test_imputed,
                                    columns=numerical_columns, index=self.test_df.index)

            self.train_df[numerical_columns] = df_numerical_train_imputed
            self.test_df[numerical_columns] = df_numerical_test_imputed
        
        elif self.impute_method_numerical == "":
            for col in numerical_columns:
                # Dropping rows with missing values in the specified column
                self.train_df = self.train_df.dropna(subset=[col])
                # self.test_df = self.test_df.dropna(subset=[col]) # Should not delete from test set
                mean_value = self.train_df[col].mean()
                self.put_imputation_stats(col, 'mean', mean_value)
                self.test_df[col] = self.test_df[col].fillna(mean_value)


    # Function to impute missing values in categorical columns using self.impute_method_categorical.
    # If it is equal to mode, then mode should be used to impute missing values.
    def impute_missing_values_categorical(self):
        # Print All the Categorical Columns
        categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()

        # Use Mode method to impute values in categorical columns
        if self.impute_method_categorical == 'mode':
            for col in categorical_columns:
                mode_value = self.train_df[col].mode()[0]
                self.put_imputation_stats(col, 'mode', mode_value, type="categorical")
                self.train_df[col] = self.train_df[col].fillna(mode_value)
                self.test_df[col] = self.test_df[col].fillna(mode_value)


        # Use Dummy method to impute values in categorical columns with a new value - "Missing"
        elif self.impute_method_categorical == 'dummy':
            for col in categorical_columns:
                self.put_imputation_stats(col, 'dummy', "Missing", type="categorical")
                self.train_df[col] = self.train_df[col].fillna("Missing")
                self.test_df[col] = self.test_df[col].fillna("Missing")

    def detect_and_repair(self):

        # Use Mean method to impute values in numerical columns
        self.impute_missing_values_numerical()

        # Use Mode method to impute values in categorical columns
        self.impute_missing_values_categorical()


    def put_imputation_stats(self, col_name, impute_method, impute_value, type="numerical"):
        if type == "numerical":
            self.numerical_impute_stats[impute_method][col_name] = impute_value
        else:
            self.categorical_impute_stats[impute_method][col_name] = impute_value

    def get_imputation_stats(self):
        print(pd.DataFrame(self.numerical_impute_stats).T)

    def randomly_drop_data(self, df, target_columns, drop_ratio):
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