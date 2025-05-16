import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from train.HyperParameterTuner import HyperParameterTuner as HyperParameterTuner
import warnings
warnings.filterwarnings('ignore')


class Trainer():
    def __init__(self, task_type, target_column, ml_model, metric, train_df, test_df):
        self.task_type = task_type
        self.target_column = target_column
        self.ml_model = ml_model
        self.metric = metric
        self.tuner = HyperParameterTuner()
        self.df = pd.concat([train_df, test_df])
        self.train_df = train_df
        self.test_df = test_df

    def drop_all_missing_values(self):
        # Figure out columns with missing values in self.df
        columns_with_missing_values = self.df.columns[self.df.isnull().any()].tolist()

        # drop these columns in all the datasets
        self.train_df = self.train_df.drop(columns_with_missing_values, axis=1)
        self.test_df = self.test_df.drop(columns_with_missing_values, axis=1)

    def apply_standardization(self):
        # Standard scaler
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(self.train_df)
        self.train_df = pd.DataFrame(standardized_data, columns=self.train_df.columns)
        standardized_data = scaler.transform(self.test_df)
        self.test_df = pd.DataFrame(standardized_data, columns=self.test_df.columns)

    def preprocess_data(self, drop_mv=True, std_scale=True):
        if drop_mv:
            self.drop_all_missing_values()

        if std_scale:
            self.apply_standardization()

        x_train_input = self.train_df.drop(self.target_column, axis=1)
        y_train_target = self.train_df[[self.target_column]]

        x_test_input = self.test_df.drop(self.target_column, axis=1)
        y_test_target = self.test_df[[self.target_column]]

        if self.task_type == "classification":
            le = LabelEncoder()
            y_train_target = le.fit_transform(y_train_target)
            y_test_target = le.transform(y_test_target)

        # Combine the datasets
        combined = pd.concat([x_train_input, x_test_input], axis=0, ignore_index=True)
        # Apply pd.get_dummies to the combined dataset
        combined_dummies = pd.get_dummies(combined)

        # Split the datasets back into training and testing datasets
        x_train_input = combined_dummies.iloc[:len(x_train_input)]
        x_test_input = combined_dummies.iloc[len(x_train_input):]

        return x_train_input, y_train_target, x_test_input, y_test_target


    def train_model(self, x_train_input, y_train_target, x_test_input, y_test_target):

        if self.task_type == "regression" and self.ml_model == "linear_regression":
            model = LinearRegression()
            model.fit(x_train_input, y_train_target)
            y_pred = model.predict(x_test_input)
            return y_pred, y_test_target, model

        model = None

        # Depending on self.ml_model, train a logistic regression,
        # decision tree, random forest, naive bayes or knn model
        if self.ml_model == 'logistic_regression':
            model = LogisticRegression()
        elif self.ml_model == 'decision_tree':
            model = DecisionTreeClassifier()
        elif self.ml_model == 'random_forest':
            model = RandomForestClassifier()
        elif self.ml_model == 'naive_bayes':
            model = GaussianNB()
        elif self.ml_model == 'knn':
            model = KNeighborsClassifier()
        elif self.ml_model == 'ada_boost':
            model = AdaBoostClassifier()

        model.fit(x_train_input, y_train_target)
        y_pred = model.predict(x_test_input)
        return y_pred, y_test_target, model

    def calculate_metric(self, y_pred, y_test):
        # Depending on self.metric, calculate f1_score, accuracy_score or roc_auc_score
        metric_value = None
        if self.task_type == "classification":
            if self.metric == 'f1_score':
                metric_value = f1_score(y_test, y_pred, average='weighted')
            elif self.metric == 'accuracy_score':
                metric_value = accuracy_score(y_test, y_pred)
            elif self.metric == 'roc_auc_score':
                metric_value = roc_auc_score(y_test, y_pred)
        elif self.task_type == "regression":
            if self.metric == 'r2_score':
                metric_value = r2_score(y_test, y_pred)
            elif self.metric == 'mean_squared_error':
                metric_value = mean_squared_error(y_test, y_pred)
            elif self.metric == 'mean_absolute_error':
                metric_value = mean_absolute_error(y_test, y_pred)

        return metric_value


    def ml_pipeline(self, drop_mv=True, std_scale=True):
        x_train_input, y_train_target, x_test_input, y_test_target = self.preprocess_data(drop_mv, std_scale)
        result_dict = {}
        test_metric = self.tuner.run_model_with_hyperparameters(x_train_input, y_train_target, x_test_input,
                                                                y_test_target, self.ml_model)
        test_metric = round(test_metric, 4)
        test_metric = f'{test_metric:.4f}'

        result_dict[self.ml_model] = test_metric

        return result_dict
