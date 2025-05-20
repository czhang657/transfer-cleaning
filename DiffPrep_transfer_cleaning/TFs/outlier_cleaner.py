# import warnings
# import time
# import numpy as np
# from copy import deepcopy
# from sklearn.impute import SimpleImputer
# import time

# class ZSOutlierDetector(object):
#     """ Values out of nstd x std are considered as outliers"""
#     def __init__(self, nstd=3):
#         super(ZSOutlierDetector, self).__init__()
#         self.nstd = nstd
        
#     def fit(self, X):
#         tic = time.time()
#         mean = X.mean(axis=0)
#         std = X.std(axis=0)
#         cut_off = std * self.nstd
#         self.lower = mean - cut_off
#         self.upper = mean + cut_off
#         self.lower = self.lower.reshape(1, -1)
#         self.upper = self.upper.reshape(1, -1)

#     def detect(self, X):
#         great = X > self.upper
#         low = X < self.lower
#         preds = np.logical_or(great, low)
#         return preds

# class IQROutlierDetector(object):
#     """Interquartile Range Methods"""
#     def __init__(self, k=1.5):
#         super(IQROutlierDetector, self).__init__()
#         self.k = k
        
#     def fit(self, X):
#         q25 = np.percentile(X, 25, axis=0)
#         q75 = np.percentile(X, 75, axis=0)
#         iqr = q75 - q25
#         cut_off = iqr * self.k
#         self.lower = q25 - cut_off
#         self.upper = q75 + cut_off

#         self.lower = self.lower.reshape(1, -1)
#         self.upper = self.upper.reshape(1, -1)

#     def detect(self, X):
#         great = X > self.upper
#         low = X < self.lower
#         preds = np.logical_or(great, low)
#         return preds

# class MADOutlierDetector(object):
#     def __init__(self, nmad=2.5):
#         super(MADOutlierDetector, self).__init__()
#         self.nmad = nmad
        
#     def fit(self, X):
#         median = np.median(X, axis=0, keepdims=True)
#         mad = np.median(np.abs(X - median), axis=0, keepdims=True)

#         self.lower = median - self.nmad * mad
#         self.upper = median + self.nmad * mad
#         self.lower = self.lower.reshape(1, -1)
#         self.upper = self.upper.reshape(1, -1)
        
#     def detect(self, X):
#         great = X > self.upper
#         low = X < self.lower
#         preds = np.logical_or(great, low)
#         return preds

# class OutlierCleaner(object):
#     """ Detect outliers and repair with mean imputation
#     Available methods:
#         - 'ZS': detects outliers using the robust Zscore as a function
#         - of median and median absolute deviation (MAD)
#         - 'IQR': detects outliers using Q1 and Q3 +/- 1.5*InterQuartile Range
#         - 'MAD': median absolute deviation
#         - 'LOF': detects outliers using Local Outlier Factor
#         - 'OCSVM': detects outliers using one-class svm
#     """
#     def __init__(self, method):
#         super(OutlierCleaner, self).__init__()
#         self.method = method
#         if self.method == "ZS":
#             self.detector = ZSOutlierDetector()
#         elif self.method == "IQR":
#             self.detector = IQROutlierDetector()
#         elif self.method == "MAD":
#             self.detector = MADOutlierDetector()
#         elif "ZS" in self.method:
#             nstd = float(self.method.split("_")[1])
#             self.detector = ZSOutlierDetector(nstd=nstd)
#         elif "IQR" in self.method:
#             k = float(self.method.split("_")[1])
#             self.detector = IQROutlierDetector(k=k)
#         elif "MAD" in self.method:
#             nmad = float(self.method.split("_")[1])
#             self.detector = MADOutlierDetector(nmad=nmad)
#         else:
#             raise Exception("Invalid normalization method: {}".format(method))

#         self.repairer = SimpleImputer()
    
#     def fit(self, X):
#         self.detector.fit(X)
#         indicator = self.detector.detect(X)
#         X_clean = deepcopy(X)
#         X_clean[indicator] = np.nan
#         self.repairer.fit(X_clean)

#     def transform(self, X):
#         indicator = self.detector.detect(X)
#         X_trans = deepcopy(X)
#         X_trans[indicator] = np.nan
#         X_trans = self.repairer.transform(X_trans)
#         return X_trans

#     def fit_transform(self, X):
#         self.fit(X)
#         X_trans = self.transform(X)
#         return X_trans

# # X = np.zeros((20, 3))
# # X[0, :] = 1
# # outlier = OutlierCleaner("ZS")
# # X_trans = outlier.fit_transform(X)
# # print(X)
# # print(X_trans)

import warnings
import time
import numpy as np
from copy import deepcopy
from sklearn.impute import SimpleImputer
import time

class ZSOutlierDetector(object):
    """ Values out of nstd x std are considered as outliers"""
    def __init__(self, nstd=3):
        super(ZSOutlierDetector, self).__init__()
        self.nstd = nstd
        
    def fit(self, X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        cut_off = std * self.nstd
        self.lower = mean - cut_off
        self.upper = mean + cut_off
        self.lower = self.lower.reshape(1, -1)
        self.upper = self.upper.reshape(1, -1)

    def detect(self, X):
        great = X > self.upper
        low = X < self.lower
        preds = np.logical_or(great, low)
        return preds

class IQROutlierDetector(object):
    """Interquartile Range Methods"""
    def __init__(self, k=1.5):
        super(IQROutlierDetector, self).__init__()
        self.k = k
        
    def fit(self, X):
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25
        cut_off = iqr * self.k
        self.lower = q25 - cut_off
        self.upper = q75 + cut_off
        self.lower = self.lower.reshape(1, -1)
        self.upper = self.upper.reshape(1, -1)

    def detect(self, X):
        great = X > self.upper
        low = X < self.lower
        preds = np.logical_or(great, low)
        return preds

class MADOutlierDetector(object):
    def __init__(self, nmad=2.5):
        super(MADOutlierDetector, self).__init__()
        self.nmad = nmad
        
    def fit(self, X):
        median = np.median(X, axis=0, keepdims=True)
        mad = np.median(np.abs(X - median), axis=0, keepdims=True)
        self.lower = median - self.nmad * mad
        self.upper = median + self.nmad * mad
        self.lower = self.lower.reshape(1, -1)
        self.upper = self.upper.reshape(1, -1)
        
    def detect(self, X):
        great = X > self.upper
        low = X < self.lower
        preds = np.logical_or(great, low)
        return preds

class OutlierCleaner(object):
    """ Detect outliers and repair with mean imputation
    Avoids dropping columns even if entire column is NaN.
    """
    def __init__(self, method):
        super(OutlierCleaner, self).__init__()
        self.method = method
        if self.method == "ZS":
            self.detector = ZSOutlierDetector()
        elif self.method == "IQR":
            self.detector = IQROutlierDetector()
        elif self.method == "MAD":
            self.detector = MADOutlierDetector()
        elif "ZS" in self.method:
            nstd = float(self.method.split("_")[1])
            self.detector = ZSOutlierDetector(nstd=nstd)
        elif "IQR" in self.method:
            k = float(self.method.split("_")[1])
            self.detector = IQROutlierDetector(k=k)
        elif "MAD" in self.method:
            nmad = float(self.method.split("_")[1])
            self.detector = MADOutlierDetector(nmad=nmad)
        else:
            raise Exception("Invalid normalization method: {}".format(method))

        # set keep_empty_features=True if sklearn >= 1.1
        try:
            self.repairer = SimpleImputer(keep_empty_features=True)
        except TypeError:
            # fallback for older sklearn
            self.repairer = SimpleImputer()

    def fit(self, X):
        self.detector.fit(X)
        indicator = self.detector.detect(X)
        X_clean = deepcopy(X)
        X_clean[indicator] = np.nan
        self.repairer.fit(X_clean)

        # check dropped columns (sklearn <1.1 will drop empty cols)
        if hasattr(self.repairer, 'statistics_') and X_clean.shape[1] != self.repairer.statistics_.shape[0]:
            missing_cols = X_clean.shape[1] - self.repairer.statistics_.shape[0]
            self.dropped_columns = missing_cols
            warnings.warn(f"{missing_cols} columns were dropped by SimpleImputer; padding will be applied in transform.")
        else:
            self.dropped_columns = 0

    def transform(self, X):
        indicator = self.detector.detect(X)
        X_trans = deepcopy(X)
        X_trans[indicator] = np.nan
        X_imputed = self.repairer.transform(X_trans)

        # if columns were dropped (older sklearn), pad back with zeros
        if self.dropped_columns > 0:
            pad = np.zeros((X_imputed.shape[0], self.dropped_columns))
            X_imputed = np.concatenate([X_imputed, pad], axis=1)

        return X_imputed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Example test
# X = np.zeros((20, 3))
# X[0, :] = 1000
# outlier = OutlierCleaner("ZS")
# X_trans = outlier.fit_transform(X)
# print(X)
# print(X_trans)
