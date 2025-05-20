from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
import numpy as np
# from xgboost import XGBRegressor
import optuna
import logging
import pandas as pd
import os
optuna.logging.set_verbosity(optuna.logging.WARNING)

class HyperParameterTuner():

    def __init__(self):

        self.logistic_regression_param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 1000, 10000],
        }

        self.random_forest_param_grid = {
            'n_estimators': [10, 100, 1000],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 50, 100, 200, 500, 700, 1000],
            'min_samples_split': [2, 10, 100],
        }

        self.decision_tree_param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 100, 1000],
            'min_samples_split': [2, 10, 100],
        }

        self.svm_param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        }

        self.linear_regression_param_grid = {
            'fit_intercept': [True, False],
        }

        self.ridge_regression_param_grid = {
            'fit_intercept': [True, False],
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        }

        self.lasso_regression_param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'selection': ['cyclic', 'random'],
        }

        self.cached_hyperparameter_results = {
            'logistic_regression': {},
            'random_forest': {},
            'decision_tree': {},
            'linear_regression': {},
            'ridge_regression': {},
            'lasso_regression': {},
            'svm': {},
        }

        self.logger = logging.getLogger()

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def run_model_with_hyperparameters(self, X_train, y_train, X_test, y_test, model_name):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        if model_name == 'logistic_regression':
            return self.run_logistic_regression_model()
        elif model_name == 'random_forest':
            return self.run_random_forest_model()
        elif model_name == 'decision_tree':
            return self.run_decision_tree()
        elif model_name == 'naive_bayes':
            return self.run_naive_bayes_model()
        elif model_name == 'knn':
            return self.run_knn_model()
        elif model_name == 'ada_boost':
            return self.run_ada_boost_model()
        elif model_name == 'linear_regression':
            return self.run_linear_regression_model_with_optimal_hyperparameters()
        elif model_name == 'ridge':
            return self.run_ridge_regression_model_with_optimal_hyperparameters()
        elif model_name == 'lasso':
            return self.run_lasso_regression_model_with_optimal_hyperparameters()
        elif model_name == 'rf_regressor':
            return self.run_random_forest_regression_with_optimal_hyperparameters()
        elif model_name == 'svm':
            return self.run_svm_model()
        elif model_name == "xgboost":
            return self.run_xgboost_model_with_optimal_hyperparameters()
        else:
            raise NotImplementedError

    def run_logistic_regression_model_with_optimal_hyperparameters(self):
        if len(self.cached_hyperparameter_results['logistic_regression']) > 0:
            model = LogisticRegression(**self.cached_hyperparameter_results['logistic_regression'])
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            return f1_score(self.y_test, y_pred, average='weighted')
        else:
            self.logger.info("Beginning Logistic Regression Hyperparameter Tuning ...")
            clf = GridSearchCV(LogisticRegression(), self.logistic_regression_param_grid, cv=5, scoring='f1_macro')
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            self.cached_hyperparameter_results['logistic_regression'] = clf.best_params_
            self.logger.info("The Best Parameters for Logistic Regression are: ", clf.best_params_)
            return f1_score(self.y_test, y_pred, average='weighted')

    def run_svm_model_with_optimal_hyperparameters(self):
        if len(self.cached_hyperparameter_results['svm']) > 0:
            model = SVC(**self.cached_hyperparameter_results['svm'])
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            return f1_score(self.y_test, y_pred, average='weighted')
        else:
            self.logger.info("Beginning SVM Hyperparameter Tuning ...")
            clf = GridSearchCV(SVC(), self.svm_param_grid, cv=5, scoring='f1_macro')
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            test_metric = f1_score(self.y_test, y_pred, average='weighted')
            self.cached_hyperparameter_results['svm'] = clf.best_params_
            self.logger.info("The Best Parameters for SVM are: ", clf.best_params_)
            return test_metric

    def run_svm_model(self):
        model = SVC()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def run_decision_tree(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def run_lasso_regression_model_with_optimal_hyperparameters(self):
        if len(self.cached_hyperparameter_results['lasso_regression']) > 0:
            model = Lasso(**self.cached_hyperparameter_results['lasso_regression'])
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            return np.sqrt(mean_squared_error(self.y_test, y_pred))
        else:
            self.logger.info("Beginning Lasso Regression Hyperparameter Tuning ...")
            clf = GridSearchCV(Lasso(), self.lasso_regression_param_grid, cv=5, scoring='neg_mean_squared_error')
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            self.cached_hyperparameter_results['lasso_regression'] = clf.best_params_
            self.logger.info("The Best Parameters for Lasso Regression are: ", clf.best_params_)
            return np.sqrt(mean_squared_error(self.y_test, y_pred))

    def run_decision_tree_model_with_optimal_hyperparameters(self):
        if len(self.cached_hyperparameter_results['decision_tree']) > 0:
            model = DecisionTreeClassifier(**self.cached_hyperparameter_results['decision_tree'])
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            return f1_score(self.y_test, y_pred, average='weighted')
        else:
            self.logger.info("Beginning Decision Tree Hyperparameter Tuning ...")
            clf = GridSearchCV(DecisionTreeClassifier(), self.decision_tree_param_grid, cv=5, scoring='f1_macro')
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            self.cached_hyperparameter_results['decision_tree'] = clf.best_params_
            self.logger.info("The Best Parameters for Decision Tree are: ", clf.best_params_)
            return f1_score(self.y_test, y_pred, average='weighted')

    def run_linear_regression_model_with_optimal_hyperparameters(self):
        if len(self.cached_hyperparameter_results['linear_regression']) > 0:
            model = LinearRegression(**self.cached_hyperparameter_results['linear_regression'])
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            return np.sqrt(mean_squared_error(self.y_test, y_pred))
        else:
            self.logger.info("Beginning Linear Regression Hyperparameter Tuning ...")
            clf = GridSearchCV(LinearRegression(), self.linear_regression_param_grid, cv=5,
                               scoring='neg_mean_squared_error')
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            self.cached_hyperparameter_results['linear_regression'] = clf.best_params_
            self.logger.info("The Best Parameters for Linear Regression are: ", clf.best_params_)
            return np.sqrt(mean_squared_error(self.y_test, y_pred))
            
    def run_ridge_regression_model_with_optimal_hyperparameters(self):
        if len(self.cached_hyperparameter_results['ridge_regression']) > 0:
            model = Ridge(**self.cached_hyperparameter_results['ridge_regression'])
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            return np.sqrt(mean_squared_error(self.y_test, y_pred))
        else:
            self.logger.info("Beginning Ridge Regression Hyperparameter Tuning ...")
            clf = GridSearchCV(Ridge(), self.ridge_regression_param_grid, cv=5, scoring='neg_mean_squared_error')
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            self.cached_hyperparameter_results['ridge_regression'] = clf.best_params_
            self.logger.info("The Best Parameters for Ridge Regression are: ", clf.best_params_)
            return np.sqrt(mean_squared_error(self.y_test, y_pred))

    def run_SVM_model(self):
        model = SVC()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def run_logistic_regression_model(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def run_random_forest_model(self):
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def run_decision_tree_model(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def run_naive_bayes_model(self):
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def run_knn_model(self):
        model = KNeighborsClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def _xgboost_objective(self, trial):
        # Define hyperparameters to search
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
            'max_depth': trial.suggest_int('max_depth', 6, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'verbosity': 0,
            'device': 'cuda'
        }

        # Initialize XGBoost regressor with the suggested parameters
        xgb = XGBRegressor(**param)

        # Fit the model on training data
        xgb.fit(self.X_train, self.y_train)

        # Predict on the validation set
        y_pred = xgb.predict(self.X_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

        return rmse

    def run_xgboost_model_with_optimal_hyperparameters(self):
        # Perform hyperparameter optimization using Optuna
        #study = optuna.create_study(direction='minimize')
        #study.optimize(self._xgboost_objective, n_trials=20)

        #best_trial = study.best_trial
        # Use the best parameters to train the final model
        #best_params = best_trial.params
        best_params = {'n_estimators': 500, 'max_depth': 15, 'subsample': 0.9}
        xgb_best = XGBRegressor(**best_params)
        xgb_best.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        y_pred_test = xgb_best.predict(self.X_test)

        # Calculate RMSE on the test set
        return np.sqrt(mean_squared_error(self.y_test, y_pred_test))

    def _rf_objective(self, trial):
        # Define hyperparameters to search
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 10, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
        }

        # Initialize XGBoost regressor with the suggested parameters
        rf = RandomForestRegressor(**param)

        # Fit the model on training data
        rf.fit(self.X_train, self.y_train)

        # Predict on the validation set
        y_pred = rf.predict(self.X_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

        return rmse

    def run_random_forest_regression_with_optimal_hyperparameters(self):
        # Perform hyperparameter optimization using Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(self._rf_objective, n_trials=20)

        best_trial = study.best_trial
        # Use the best parameters to train the final model
        best_params = best_trial.params
        rf_best = RandomForestRegressor(**best_params)
        rf_best.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        y_pred_test = rf_best.predict(self.X_test)

        # Calculate RMSE on the test set
        return np.sqrt(mean_squared_error(self.y_test, y_pred_test))

    def run_ada_boost_model(self):
        model = AdaBoostClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def store_configs_in_file(self):
        import json
        with open('hyperparameter_configs.json', 'w') as fp:
            json.dump(self.cached_hyperparameter_results, fp)