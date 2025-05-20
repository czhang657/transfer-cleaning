# import torch
# import torch.nn as nn
# import numpy as np
# from copy import deepcopy
# from torch.distributions.utils import logits_to_probs, probs_to_logits
# from TFs.mv_imputer import NumMVIdentity, CatMVIdentity

# def is_contain_mv(df):
#     return df.isnull().values.sum() > 0

# class Transformer(nn.Module):
#     def __init__(self, name, tf_options, in_features, init_tf=(None, None), diff_method="num_diff"):
#         super(Transformer, self).__init__()
#         self.name = name
#         self.tf_options = tf_options
#         self.tf_methods = [tf.method for tf in self.tf_options]
#         self.num_tf_options = len(tf_options)
#         self.diff_method = diff_method
#         self.in_features = in_features
#         self.out_features = in_features
#         self.init_tf_option, self.init_p = init_tf
#         self.init_parameters()

#     def init_parameters(self):
#         if self.init_tf_option is None:
#             tf_prob_logits = torch.randn(self.out_features, self.num_tf_options)
#         else:
#             init_tf_probs = torch.ones(self.out_features, self.num_tf_options) * (
#                 (1 - self.init_p) / (self.num_tf_options - 1))
#             init_idx = self.tf_methods.index(self.init_tf_option.method)
#             init_tf_probs[:, init_idx] = self.init_p
#             tf_prob_logits = probs_to_logits(init_tf_probs)
#         self.tf_prob_logits = nn.Parameter(tf_prob_logits, requires_grad=True)
#         self.tf_prob_sample = None
#         self.is_sampled = False

#     def numerical_diff(self, X, eps=1e-6):
#         X = X.detach().numpy()
#         X_pos = X + eps
#         X_neg = X - eps
#         X_grads = []
#         for tf in self.tf_options:
#             f1 = tf.transform(X_pos)
#             f2 = tf.transform(X_neg)
#             grad = (f1 - f2) / (2 * eps)
#             X_grads.append(np.expand_dims(grad, axis=-1))
#         X_grads = np.concatenate(X_grads, axis=2)
#         X_sample_grad = (X_grads * self.tf_prob_sample.detach().numpy()).sum(axis=2)
#         return torch.Tensor(X_sample_grad)

#     def forward(self, X, is_fit, X_type, max_only=False, require_grad=True):
#         X_trans = []
#         for tf in self.tf_options:
#             X_t = tf.fit_transform(X.detach().numpy()) if is_fit else tf.transform(X.detach().numpy())
#             X_t = torch.Tensor(X_t).unsqueeze(-1)
#             X_trans.append(X_t)
#         X_trans = torch.cat(X_trans, dim=2)
#         return self.select_X_sample(X, X_trans, max_only, require_grad)

#     def select_X_sample(self, X, X_trans, max_only, require_grad):
#         tf_prob_sample = self.sample_with_max_probs() if max_only else self.tf_prob_sample
#         X_trans_sample = (X_trans * tf_prob_sample.unsqueeze(0)).sum(axis=2)
#         if not require_grad:
#             return X_trans_sample
#         if self.diff_method == "num_diff":
#             X_grad = self.numerical_diff(X)
#         else:
#             raise Exception(f"invalid diff method {self.diff_method}")
#         return X_trans_sample + (X_grad * X - (X_grad * X).detach())

#     def categorical_max(self, logits):
#         max_idx = torch.argmax(logits, dim=1)
#         max_sample = torch.zeros_like(logits)
#         max_sample[np.arange(max_sample.shape[0]), max_idx] = 1
#         return max_sample

#     def categorical_sample(self, logits, temperature, use_sample=True):
#         if not use_sample:
#             samples = logits_to_probs(logits, is_binary=False)
#         else:
#             samples = torch.distributions.RelaxedOneHotCategorical(temperature, logits=logits).rsample()
#             indicator = torch.max(samples, dim=-1, keepdim=True)[1]
#             one_h = torch.zeros_like(samples).scatter_(-1, indicator, 1.0)
#             diff = one_h - samples.detach()
#             samples = samples + diff
#         return samples

#     def sample(self, temperature, use_sample=True):
#         self.tf_prob_sample = self.categorical_sample(self.tf_prob_logits, temperature, use_sample)
#         self.is_sampled = True

#     def sample_with_max_probs(self):
#         return self.categorical_max(self.tf_prob_logits)

# class FirstTransformer(nn.Module):
#     def __init__(self, num_tf_options, cat_tf_options, init_num_tf=(None, None), init_cat_tf=(None, None)):
#         super(FirstTransformer, self).__init__()
#         self.name = "missing_value_imputation"
#         self.num_tf_options = num_tf_options
#         self.cat_tf_options = cat_tf_options
#         self.num_tf_methods = [tf.method for tf in num_tf_options]
#         self.cat_tf_methods = [tf.method for tf in cat_tf_options]
#         self.num_num_tf_options = len(num_tf_options)
#         self.num_cat_tf_options = len(cat_tf_options)
#         self.init_num_tf_option, self.init_num_p = init_num_tf
#         self.init_cat_tf_option, self.init_cat_p = init_cat_tf
#         self.cache = {}
#         self.contain_num = False
#         self.contain_cat = False
#         self.is_sampled = False  # ✅ ← 加入这个标志
#         self.num_tf_prob_sample = None
#         self.cat_tf_prob_sample = None

#     def fit_transform(self, X):
#         X_num = X.select_dtypes(include='number')
#         X_cat = X.select_dtypes(exclude='number')
#         self.num_columns = X_num.columns
#         self.cat_columns = X_cat.columns

#         self.contain_num = X_num.shape[1] > 0
#         self.contain_cat = X_cat.shape[1] > 0

#         X_num_trans = []
#         X_cat_trans = []
#         self.out_num_features = 0
#         self.out_cat_features = 0
#         self.cache["train"] = {"X_num_trans": None, "X_cat_trans": None}

#         if self.contain_num:
#             for tf in self.num_tf_options:
#                 X_num_t = tf.fit_transform(X_num.values)
#                 X_num_trans.append(X_num_t)
#             X_num_trans = torch.Tensor(np.array(X_num_trans)).permute(1, 2, 0)
#             self.cache["train"]["X_num_trans"] = X_num_trans
#             self.out_num_features = X_num_trans.shape[1]

#         if self.contain_cat:
#             for tf in self.cat_tf_options:
#                 X_cat_t = tf.fit_transform(X_cat.values)
#                 X_cat_trans.append(X_cat_t)
#             X_cat_trans = torch.Tensor(np.array(X_cat_trans)).permute(1, 2, 0)
#             self.cache["train"]["X_cat_trans"] = X_cat_trans
#             self.out_cat_features = X_cat_trans.shape[1]

#         self.out_features = self.out_num_features + self.out_cat_features
#         self.init_parameters()

#     def init_parameters(self):
#         if self.contain_num:
#             if self.init_num_tf_option is None:
#                 num_tf_prob_logits = torch.randn(self.out_num_features, self.num_num_tf_options)
#             else:
#                 probs = torch.ones(self.out_num_features, self.num_num_tf_options) * (
#                         (1 - self.init_num_p) / (self.num_num_tf_options - 1))
#                 idx = self.num_tf_methods.index(self.init_num_tf_option.method)
#                 probs[:, idx] = self.init_num_p
#                 num_tf_prob_logits = probs_to_logits(probs)
#             self.num_tf_prob_logits = nn.Parameter(num_tf_prob_logits, requires_grad=True)
#         else:
#             self.num_tf_prob_logits = None

#         if self.contain_cat:
#             if self.init_cat_tf_option is None:
#                 cat_tf_prob_logits = torch.randn(self.out_cat_features, self.num_cat_tf_options)
#             else:
#                 probs = torch.ones(self.out_cat_features, self.num_cat_tf_options) * (
#                         (1 - self.init_cat_p) / (self.num_cat_tf_options - 1))
#                 idx = self.cat_tf_methods.index(self.init_cat_tf_option.method)
#                 probs[:, idx] = self.init_cat_p
#                 cat_tf_prob_logits = probs_to_logits(probs)
#             self.cat_tf_prob_logits = nn.Parameter(cat_tf_prob_logits, requires_grad=True)
#         else:
#             self.cat_tf_prob_logits = None

#     def pre_cache(self, X, X_type):
#         X_num = X[self.num_columns]
#         X_cat = X[self.cat_columns]
#         if self.contain_num:
#             X_num_trans = [tf.transform(X_num.values) for tf in self.num_tf_options]
#             X_num_trans = torch.Tensor(np.array(X_num_trans)).permute(1, 2, 0)
#         else:
#             X_num_trans = None
#         if self.contain_cat:
#             X_cat_trans = [tf.transform(X_cat.values) for tf in self.cat_tf_options]
#             X_cat_trans = torch.Tensor(np.array(X_cat_trans)).permute(1, 2, 0)
#         else:
#             X_cat_trans = None
#         self.cache[X_type] = {"X_num_trans": X_num_trans, "X_cat_trans": X_cat_trans}

#     def forward(self, X, is_fit, X_type, max_only=False, require_grad=True):
#         indices = X.index
#         X_num_trans = self.cache[X_type]["X_num_trans"][indices] if self.contain_num else None
#         X_cat_trans = self.cache[X_type]["X_cat_trans"][indices] if self.contain_cat else None
#         return self.select_X_sample(X_num_trans, X_cat_trans, max_only)

#     def select_X_sample(self, X_num_trans, X_cat_trans, max_only):
#         num_tf_prob_sample, cat_tf_prob_sample = self.sample_with_max_probs() if max_only else (
#             self.num_tf_prob_sample, self.cat_tf_prob_sample)
#         X_num_sample = (X_num_trans * num_tf_prob_sample.unsqueeze(0)).sum(axis=2) if X_num_trans is not None else None
#         X_cat_sample = (X_cat_trans * cat_tf_prob_sample.unsqueeze(0)).sum(axis=2) if X_cat_trans is not None else None
#         if X_num_sample is None:
#             return X_cat_sample
#         if X_cat_sample is None:
#             return X_num_sample
#         return torch.cat((X_num_sample, X_cat_sample), dim=1)

#     def sample_with_max_probs(self):
#         num_tf_prob_sample = self._categorical_max(self.num_tf_prob_logits) if self.num_tf_prob_logits is not None else None
#         cat_tf_prob_sample = self._categorical_max(self.cat_tf_prob_logits) if self.cat_tf_prob_logits is not None else None
#         return num_tf_prob_sample, cat_tf_prob_sample

#     def _categorical_max(self, logits):
#         max_idx = torch.argmax(logits, dim=1)
#         max_sample = torch.zeros_like(logits)
#         max_sample[np.arange(max_sample.shape[0]), max_idx] = 1
#         return max_sample

#     def sample(self, temperature=0.1, use_sample=True):
#         if self.num_tf_prob_logits is not None:
#             self.num_tf_prob_sample = self._categorical_sample(self.num_tf_prob_logits, temperature, use_sample)
#         if self.cat_tf_prob_logits is not None:
#             self.cat_tf_prob_sample = self._categorical_sample(self.cat_tf_prob_logits, temperature, use_sample)
#         self.is_sampled = True

#     def _categorical_sample(self, logits, temperature, use_sample=True):
#         if not use_sample:
#             samples = logits_to_probs(logits, is_binary=False)
#         else:
#             samples = torch.distributions.RelaxedOneHotCategorical(temperature, logits=logits).rsample()
#             indicator = torch.max(samples, dim=-1, keepdim=True)[1]
#             one_h = torch.zeros_like(samples).scatter_(-1, indicator, 1.0)
#             diff = one_h - samples.detach()
#             samples = samples + diff
#         return samples


# def is_contain_mv(df):
#     return df.isnull().values.sum() > 0

# class DiffPrepFixPipeline(nn.Module):
#     def __init__(self, prep_space, temperature=0.1, use_sample=False, diff_method="num_diff", init_method="default"):
#         super(DiffPrepFixPipeline, self).__init__()
#         self.prep_space = prep_space
#         self.temperature = temperature
#         self.use_sample = use_sample
#         self.diff_method = diff_method
#         self.is_fitted = False
#         self.init_method = init_method

#     def init_parameters(self, X_train, X_val, X_test):
#         if "Date" in X_train.columns:
#             X_train = X_train.drop(columns=["Date"])
#         if "Date" in X_val.columns:
#             X_val = X_val.drop(columns=["Date"])
#         if "Date" in X_test.columns:
#             X_test = X_test.drop(columns=["Date"])

#         pipeline = []
#         self.contain_mv = is_contain_mv(X_train) or is_contain_mv(X_val) or is_contain_mv(X_test)

#         if self.contain_mv:
#         #     first_tf_dict = self.prep_space[0]
#         #     init_num_tf = next(tf for tf in first_tf_dict["num_tf_options"] if tf.method.lower() == "mice")
#         #     init_cat_tf = next((tf for tf in first_tf_dict["cat_tf_options"] if tf.method.lower() == "identity"),
#         #                     first_tf_dict["cat_tf_options"][0])
#         #     first_transformer = FirstTransformer(first_tf_dict["num_tf_options"], first_tf_dict["cat_tf_options"],
#         #                                         init_num_tf=(init_num_tf, 1.0), init_cat_tf=(init_cat_tf, 1.0))
#             first_tf_dict = self.prep_space[0]
#             first_transformer = FirstTransformer(first_tf_dict["num_tf_options"], first_tf_dict["cat_tf_options"],
#                                                 init_num_tf=(None, None), init_cat_tf=(None, None))


#         else:
#             first_transformer = FirstTransformer([NumMVIdentity()], [CatMVIdentity()],
#                                                 init_num_tf=(NumMVIdentity(), 1.0), init_cat_tf=(CatMVIdentity(), 1.0))

#         first_transformer.fit_transform(X_train)
#         first_transformer.pre_cache(X_val, "val")
#         first_transformer.pre_cache(X_test, "test")

#         pipeline.append(first_transformer)
#         in_features = first_transformer.out_features

#         for tf_dict in self.prep_space[1:]:
#             if tf_dict["name"] == "normalization":
#                 init_tf = next(tf for tf in tf_dict["tf_options"] if tf.method == "ZS")
#             elif tf_dict["name"] in ["cleaning_outliers", "discretization"]:
#                 init_tf = next(tf for tf in tf_dict["tf_options"] if tf.method == "identity")
#             else:
#                 init_tf = tf_dict["default"]

#             transformer = Transformer(tf_dict["name"], tf_dict["tf_options"], in_features,
#                                     init_tf=(init_tf, 1.0), diff_method=self.diff_method)
#             pipeline.append(transformer)

#         self.pipeline = nn.ModuleList(pipeline)
#         self.out_features = in_features

#     def forward(self, X, is_fit, X_type, resample=False, max_only=False, require_grad=True):
#         X_output = deepcopy(X)
#         for transformer in self.pipeline:
#             if resample or not transformer.is_sampled:
#                 transformer.sample(temperature=self.temperature, use_sample=self.use_sample)
#             X_output = transformer(X_output, is_fit, X_type, max_only=max_only, require_grad=require_grad)
#         return X_output

#     def fit(self, X):
#         self.is_fitted = True
#         final_X = self.forward(X, is_fit=True, X_type="train", resample=True)
#         # print("[DEBUG] Final dataset before training:", final_X)
#         return final_X

#     def transform(self, X, X_type, max_only=False, resample=False, require_grad=True):
#         if not self.is_fitted:
#             raise Exception("transformer is not fitted")
#         return self.forward(X, is_fit=False, X_type=X_type, resample=resample, max_only=max_only, require_grad=require_grad)

#     def get_final_dataset(self, X, X_type):
#         return self.forward(X, is_fit=False, X_type=X_type, resample=False, max_only=True)

#     def best_config(self):
#         config = {}
#         for transformer in self.pipeline:
#             if hasattr(transformer, 'num_tf_methods'):
#                 if transformer.contain_num:
#                     num_idx = torch.argmax(transformer.num_tf_prob_logits, dim=1)
#                     num_methods = [transformer.num_tf_methods[i] for i in num_idx]
#                     config["num_tf"] = num_methods
#                 if transformer.contain_cat:
#                     cat_idx = torch.argmax(transformer.cat_tf_prob_logits, dim=1)
#                     cat_methods = [transformer.cat_tf_methods[i] for i in cat_idx]
#                     config["cat_tf"] = cat_methods
#             else:
#                 tf_idx = torch.argmax(transformer.tf_prob_logits, dim=1)
#                 tf_methods = [transformer.tf_methods[i] for i in tf_idx]
#                 config[transformer.name] = tf_methods
#         return config
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.distributions.utils import logits_to_probs, probs_to_logits
from TFs.mv_imputer import NumMVIdentity, CatMVIdentity

def is_contain_mv(df):
    return df.isnull().values.sum() > 0

class Transformer(nn.Module):
    def __init__(self, name, tf_options, in_features, init_tf=(None, None), diff_method="num_diff"):
        super(Transformer, self).__init__()
        self.name = name
        self.tf_options = tf_options
        self.tf_methods = [tf.method for tf in self.tf_options]
        self.num_tf_options = len(tf_options)
        self.diff_method = diff_method
        self.in_features = in_features
        self.out_features = in_features
        self.init_tf_option, self.init_p = init_tf
        self.init_parameters()

    def init_parameters(self):
        if self.init_tf_option is None:
            tf_prob_logits = torch.randn(self.out_features, self.num_tf_options)
        else:
            init_tf_probs = torch.ones(self.out_features, self.num_tf_options) * (
                (1 - self.init_p) / (self.num_tf_options - 1))
            init_idx = self.tf_methods.index(self.init_tf_option.method)
            init_tf_probs[:, init_idx] = self.init_p
            tf_prob_logits = probs_to_logits(init_tf_probs)
        self.tf_prob_logits = nn.Parameter(tf_prob_logits, requires_grad=True)
        self.tf_prob_sample = None
        self.is_sampled = False

    def numerical_diff(self, X, eps=1e-6):
        X = X.detach().numpy()
        X_pos = X + eps
        X_neg = X - eps
        X_grads = []
        for tf in self.tf_options:
            f1 = tf.transform(X_pos)
            f2 = tf.transform(X_neg)
            grad = (f1 - f2) / (2 * eps)
            X_grads.append(np.expand_dims(grad, axis=-1))
        X_grads = np.concatenate(X_grads, axis=2)
        X_sample_grad = (X_grads * self.tf_prob_sample.detach().numpy()).sum(axis=2)
        return torch.Tensor(X_sample_grad)

    def forward(self, X, is_fit, X_type, max_only=False, require_grad=True):
        X_trans = []
        for tf in self.tf_options:
            X_t = tf.fit_transform(X.detach().numpy()) if is_fit else tf.transform(X.detach().numpy())
            X_t = torch.Tensor(X_t).unsqueeze(-1)
            X_trans.append(X_t)
        X_trans = torch.cat(X_trans, dim=2)
        return self.select_X_sample(X, X_trans, max_only, require_grad)

    def select_X_sample(self, X, X_trans, max_only, require_grad):
        tf_prob_sample = self.sample_with_max_probs() if max_only else self.tf_prob_sample
        X_trans_sample = (X_trans * tf_prob_sample.unsqueeze(0)).sum(axis=2)
        if not require_grad:
            return X_trans_sample
        if self.diff_method == "num_diff":
            X_grad = self.numerical_diff(X)
        else:
            raise Exception(f"invalid diff method {self.diff_method}")
        return X_trans_sample + (X_grad * X - (X_grad * X).detach())

    def categorical_max(self, logits):
        max_idx = torch.argmax(logits, dim=1)
        max_sample = torch.zeros_like(logits)
        max_sample[np.arange(max_sample.shape[0]), max_idx] = 1
        return max_sample

    def categorical_sample(self, logits, temperature, use_sample=True):
        if not use_sample:
            samples = logits_to_probs(logits, is_binary=False)
        else:
            samples = torch.distributions.RelaxedOneHotCategorical(temperature, logits=logits).rsample()
            indicator = torch.max(samples, dim=-1, keepdim=True)[1]
            one_h = torch.zeros_like(samples).scatter_(-1, indicator, 1.0)
            diff = one_h - samples.detach()
            samples = samples + diff
        return samples

    def sample(self, temperature, use_sample=True):
        self.tf_prob_sample = self.categorical_sample(self.tf_prob_logits, temperature, use_sample)
        self.is_sampled = True

    def sample_with_max_probs(self):
        return self.categorical_max(self.tf_prob_logits)

class DiffPrepFixPipeline(nn.Module):
    def __init__(self, prep_space, temperature=0.1, use_sample=False, diff_method="num_diff", init_method="default"):
        super(DiffPrepFixPipeline, self).__init__()
        self.prep_space = prep_space
        self.temperature = temperature
        self.use_sample = use_sample
        self.diff_method = diff_method
        self.is_fitted = False
        self.init_method = init_method

    def init_parameters(self, X_train, X_val, X_test):
        if "Date" in X_train.columns:
            X_train = X_train.drop(columns=["Date"])
        if "Date" in X_val.columns:
            X_val = X_val.drop(columns=["Date"])
        if "Date" in X_test.columns:
            X_test = X_test.drop(columns=["Date"])

        pipeline = []
        self.contain_mv = is_contain_mv(X_train) or is_contain_mv(X_val) or is_contain_mv(X_test)

        if self.contain_mv:
            first_tf_dict = self.prep_space[0]
            first_transformer = FirstTransformer(first_tf_dict["num_tf_options"], first_tf_dict["cat_tf_options"],
                                                init_num_tf=(None, None), init_cat_tf=(None, None))
        else:
            first_transformer = FirstTransformer([NumMVIdentity()], [CatMVIdentity()],
                                                init_num_tf=(NumMVIdentity(), 1.0), init_cat_tf=(CatMVIdentity(), 1.0))

        first_transformer.fit_transform(X_train)
        first_transformer.pre_cache(X_val, "val")
        first_transformer.pre_cache(X_test, "test")

        pipeline.append(first_transformer)
        in_features = first_transformer.out_features

        for tf_dict in self.prep_space[1:]:
            if tf_dict["name"] != "normalization":
                continue
            init_tf = next(tf for tf in tf_dict["tf_options"] if tf.method == "ZS")
            transformer = Transformer(tf_dict["name"], tf_dict["tf_options"], in_features,
                                      init_tf=(init_tf, 1.0), diff_method=self.diff_method)
            pipeline.append(transformer)

        self.pipeline = nn.ModuleList(pipeline)
        self.out_features = in_features

    def forward(self, X, is_fit, X_type, resample=False, max_only=False, require_grad=True):
        X_output = deepcopy(X)
        for transformer in self.pipeline:
            if resample or not transformer.is_sampled:
                transformer.sample(temperature=self.temperature, use_sample=self.use_sample)
            X_output = transformer(X_output, is_fit, X_type, max_only=max_only, require_grad=require_grad)
        return X_output

    def fit(self, X):
        self.is_fitted = True
        final_X = self.forward(X, is_fit=True, X_type="train", resample=True)
        return final_X

    def transform(self, X, X_type, max_only=False, resample=False, require_grad=True):
        if not self.is_fitted:
            raise Exception("transformer is not fitted")
        return self.forward(X, is_fit=False, X_type=X_type, resample=resample, max_only=max_only, require_grad=require_grad)

    def get_final_dataset(self, X, X_type):
        return self.forward(X, is_fit=False, X_type=X_type, resample=False, max_only=True)

    def best_config(self):
        config = {}
        for transformer in self.pipeline:
            if hasattr(transformer, 'num_tf_methods'):
                if transformer.contain_num:
                    num_idx = torch.argmax(transformer.num_tf_prob_logits, dim=1)
                    num_methods = [transformer.num_tf_methods[i] for i in num_idx]
                    config["num_tf"] = num_methods
                if transformer.contain_cat:
                    cat_idx = torch.argmax(transformer.cat_tf_prob_logits, dim=1)
                    cat_methods = [transformer.cat_tf_methods[i] for i in cat_idx]
                    config["cat_tf"] = cat_methods
            else:
                tf_idx = torch.argmax(transformer.tf_prob_logits, dim=1)
                tf_methods = [transformer.tf_methods[i] for i in tf_idx]
                config[transformer.name] = tf_methods
        return config

class FirstTransformer(nn.Module):
    def __init__(self, num_tf_options, cat_tf_options, init_num_tf=(None, None), init_cat_tf=(None, None)):
        super(FirstTransformer, self).__init__()
        self.name = "missing_value_imputation"
        self.num_tf_options = num_tf_options
        self.cat_tf_options = cat_tf_options
        self.num_tf_methods = [tf.method for tf in num_tf_options]
        self.cat_tf_methods = [tf.method for tf in cat_tf_options]
        self.num_num_tf_options = len(num_tf_options)
        self.num_cat_tf_options = len(cat_tf_options)
        self.init_num_tf_option, self.init_num_p = init_num_tf
        self.init_cat_tf_option, self.init_cat_p = init_cat_tf
        self.cache = {}
        self.contain_num = False
        self.contain_cat = False
        self.is_sampled = False  # ✅ ← 加入这个标志
        self.num_tf_prob_sample = None
        self.cat_tf_prob_sample = None

    def fit_transform(self, X):
        X_num = X.select_dtypes(include='number')
        X_cat = X.select_dtypes(exclude='number')
        self.num_columns = X_num.columns
        self.cat_columns = X_cat.columns

        self.contain_num = X_num.shape[1] > 0
        self.contain_cat = X_cat.shape[1] > 0

        X_num_trans = []
        X_cat_trans = []
        self.out_num_features = 0
        self.out_cat_features = 0
        self.cache["train"] = {"X_num_trans": None, "X_cat_trans": None}

        if self.contain_num:
            for tf in self.num_tf_options:
                X_num_t = tf.fit_transform(X_num.values)
                X_num_trans.append(X_num_t)
            X_num_trans = torch.Tensor(np.array(X_num_trans)).permute(1, 2, 0)
            self.cache["train"]["X_num_trans"] = X_num_trans
            self.out_num_features = X_num_trans.shape[1]

        if self.contain_cat:
            for tf in self.cat_tf_options:
                X_cat_t = tf.fit_transform(X_cat.values)
                X_cat_trans.append(X_cat_t)
            X_cat_trans = torch.Tensor(np.array(X_cat_trans)).permute(1, 2, 0)
            self.cache["train"]["X_cat_trans"] = X_cat_trans
            self.out_cat_features = X_cat_trans.shape[1]

        self.out_features = self.out_num_features + self.out_cat_features
        self.init_parameters()

    def init_parameters(self):
        if self.contain_num:
            if self.init_num_tf_option is None:
                num_tf_prob_logits = torch.randn(self.out_num_features, self.num_num_tf_options)
            else:
                probs = torch.ones(self.out_num_features, self.num_num_tf_options) * (
                        (1 - self.init_num_p) / (self.num_num_tf_options - 1))
                idx = self.num_tf_methods.index(self.init_num_tf_option.method)
                probs[:, idx] = self.init_num_p
                num_tf_prob_logits = probs_to_logits(probs)
            self.num_tf_prob_logits = nn.Parameter(num_tf_prob_logits, requires_grad=True)
        else:
            self.num_tf_prob_logits = None

        if self.contain_cat:
            if self.init_cat_tf_option is None:
                cat_tf_prob_logits = torch.randn(self.out_cat_features, self.num_cat_tf_options)
            else:
                probs = torch.ones(self.out_cat_features, self.num_cat_tf_options) * (
                        (1 - self.init_cat_p) / (self.num_cat_tf_options - 1))
                idx = self.cat_tf_methods.index(self.init_cat_tf_option.method)
                probs[:, idx] = self.init_cat_p
                cat_tf_prob_logits = probs_to_logits(probs)
            self.cat_tf_prob_logits = nn.Parameter(cat_tf_prob_logits, requires_grad=True)
        else:
            self.cat_tf_prob_logits = None

    def pre_cache(self, X, X_type):
        X_num = X[self.num_columns]
        X_cat = X[self.cat_columns]
        if self.contain_num:
            X_num_trans = [tf.transform(X_num.values) for tf in self.num_tf_options]
            X_num_trans = torch.Tensor(np.array(X_num_trans)).permute(1, 2, 0)
        else:
            X_num_trans = None
        if self.contain_cat:
            X_cat_trans = [tf.transform(X_cat.values) for tf in self.cat_tf_options]
            X_cat_trans = torch.Tensor(np.array(X_cat_trans)).permute(1, 2, 0)
        else:
            X_cat_trans = None
        self.cache[X_type] = {"X_num_trans": X_num_trans, "X_cat_trans": X_cat_trans}

    def forward(self, X, is_fit, X_type, max_only=False, require_grad=True):
        indices = X.index
        X_num_trans = self.cache[X_type]["X_num_trans"][indices] if self.contain_num else None
        X_cat_trans = self.cache[X_type]["X_cat_trans"][indices] if self.contain_cat else None
        return self.select_X_sample(X_num_trans, X_cat_trans, max_only)

    def select_X_sample(self, X_num_trans, X_cat_trans, max_only):
        num_tf_prob_sample, cat_tf_prob_sample = self.sample_with_max_probs() if max_only else (
            self.num_tf_prob_sample, self.cat_tf_prob_sample)
        X_num_sample = (X_num_trans * num_tf_prob_sample.unsqueeze(0)).sum(axis=2) if X_num_trans is not None else None
        X_cat_sample = (X_cat_trans * cat_tf_prob_sample.unsqueeze(0)).sum(axis=2) if X_cat_trans is not None else None
        if X_num_sample is None:
            return X_cat_sample
        if X_cat_sample is None:
            return X_num_sample
        return torch.cat((X_num_sample, X_cat_sample), dim=1)

    def sample_with_max_probs(self):
        num_tf_prob_sample = self._categorical_max(self.num_tf_prob_logits) if self.num_tf_prob_logits is not None else None
        cat_tf_prob_sample = self._categorical_max(self.cat_tf_prob_logits) if self.cat_tf_prob_logits is not None else None
        return num_tf_prob_sample, cat_tf_prob_sample

    def _categorical_max(self, logits):
        max_idx = torch.argmax(logits, dim=1)
        max_sample = torch.zeros_like(logits)
        max_sample[np.arange(max_sample.shape[0]), max_idx] = 1
        return max_sample

    def sample(self, temperature=0.1, use_sample=True):
        if self.num_tf_prob_logits is not None:
            self.num_tf_prob_sample = self._categorical_sample(self.num_tf_prob_logits, temperature, use_sample)
        if self.cat_tf_prob_logits is not None:
            self.cat_tf_prob_sample = self._categorical_sample(self.cat_tf_prob_logits, temperature, use_sample)
        self.is_sampled = True

    def _categorical_sample(self, logits, temperature, use_sample=True):
        if not use_sample:
            samples = logits_to_probs(logits, is_binary=False)
        else:
            samples = torch.distributions.RelaxedOneHotCategorical(temperature, logits=logits).rsample()
            indicator = torch.max(samples, dim=-1, keepdim=True)[1]
            one_h = torch.zeros_like(samples).scatter_(-1, indicator, 1.0)
            diff = one_h - samples.detach()
            samples = samples + diff
        return samples