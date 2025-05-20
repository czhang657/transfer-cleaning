import torch
from tqdm import tqdm
from copy import deepcopy
from torch.autograd import Variable
import numpy as np
import time

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

def make_batch(X, y, batch_size, shuffle=False):
    """Create a batch generator"""
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    start_idx = 0

    while True:
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield X.iloc[batch_idx], y[batch_idx]

        start_idx += batch_size
        if start_idx >= X.shape[0]:
            break

def take_random_batch(X, y, batch_size):
    """Take a random batch"""
    indices = np.random.permutation(X.shape[0])
    batch_idx = indices[:batch_size]
    return X.iloc[batch_idx], y[batch_idx]

class DiffPrepSGD(object):
    def __init__(self, prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
                 model_scheduler, prep_pipeline_scheduler, params, writer=None):
        self.prep_pipeline = prep_pipeline
        self.model = model
        self.loss_fn = loss_fn
        self.model_optimizer = model_optimizer
        self.prep_pipeline_optimizer = prep_pipeline_optimizer
        self.model_scheduler = model_scheduler
        self.prep_pipeline_scheduler = prep_pipeline_scheduler
        self.params = params
        self.device = self.params["device"]
        self.writer = writer

    def forward_propagate(self, X, y, X_type, require_transform_grad=False,
                          require_model_grad=False, max_only=False):
        """Forward pass through prep pipeline and model"""
        with torch.set_grad_enabled(require_transform_grad):
            X_trans = self.prep_pipeline.transform(
                X, X_type=X_type, max_only=max_only, resample=False, require_grad=require_transform_grad)

        if X_type == "train":
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(require_model_grad or require_transform_grad):
            X_trans = X_trans.to(self.device)
            output = self.model(X_trans)
        # y = y.to(self.device)
        # y = y.to(self.device).view(-1, 1)
        y = y.to(self.device).float().view(-1, 1)

        loss = self.loss_fn(output, y)
        return output, loss

    def fit(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None, verbose=True):
        """Train the model and prep pipeline"""
        best_val_loss = float("inf")
        best_model = None
        best_result = None

        if self.writer is not None:
            self.log_prep_pipeline(global_step=-1)

        if verbose:
            pbar = tqdm(range(self.params["num_epochs"]))

        patience = self.params["patience"]
        e = 0
        patience_counter = 0

        while e < self.params["num_epochs"]:
            tic = time.time()
            self.global_step = e

            tr_loss = self.train(X_train, y_train, X_val, y_val)

            if self.writer is not None:
                self.log_prep_pipeline(global_step=e)

            val_loss = self.evaluate(X_val, y_val, X_type='val', max_only=False)
            test_loss = self.evaluate(X_test, y_test, X_type='test', max_only=False)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_result = {
                    "best_epoch": e,
                    "best_val_loss": val_loss,
                    "best_tr_loss": tr_loss,
                    "best_test_loss": test_loss,
                }
                best_model = {
                    "prep_pipeline": deepcopy(self.prep_pipeline.state_dict()),
                    "end_model": deepcopy(self.model.state_dict())
                }
            else:
                patience_counter += 1

            model_lr = self.model_optimizer.param_groups[0]['lr']

            if self.model_scheduler is not None:
                self.model_scheduler.step(val_loss)

            if self.prep_pipeline_scheduler is not None:
                self.prep_pipeline_scheduler.step(val_loss)

            if self.writer is not None:
                self.writer.add_scalar('tr_loss', tr_loss, global_step=e)
                self.writer.add_scalar('val_loss', val_loss, global_step=e)
                self.writer.add_scalar('test_loss', test_loss, global_step=e)
                self.writer.add_scalar('model_lr', model_lr, global_step=e)

            epoch_time = str(int((time.time() - tic) * 100)) + "s"

            if patience_counter >= patience:
                break

            e += 1

            if verbose:
                pbar.set_postfix(tr_loss=tr_loss, val_loss=val_loss, epoch_time=epoch_time)
                pbar.update(1)

        if verbose:
            pbar.close()

        return best_result, best_model

    def train(self, X_train, y_train, X_val, y_val):
        """Train for one epoch"""
        X_val_batch, y_val_batch = take_random_batch(X_val, y_val, self.params["pipeline_update_sample_size"])
        X_train_batch, y_train_batch = take_random_batch(X_train, y_train, self.params["pipeline_update_sample_size"])

        if not self.prep_pipeline.is_fitted:
            self.prep_pipeline.fit(X_train_batch)
        self.update_prep_pipeline(X_train_batch, y_train_batch, X_val_batch, y_val_batch)
        self.prep_pipeline.fit(X_train_batch)

        tr_loss = 0
        n_batches = 0
        X_train_iter = make_batch(X_train, y_train, self.params["batch_size"], shuffle=False)

        for X_train_batch, y_train_batch in X_train_iter:
            loss = self.update_model(X_train_batch, y_train_batch)
            tr_loss += loss
            n_batches += 1

        avg_loss = tr_loss / n_batches
        return avg_loss

    def evaluate(self, X, y, X_type, max_only=True):
        """Evaluate loss (no accuracy for regression)"""
        _, loss = self.forward_propagate(X, y, X_type=X_type, max_only=max_only)
        return loss.item()

    def update_model(self, X_train, y_train):
        """Single model update step"""
        self.model_optimizer.zero_grad()
        _, loss_train = self.forward_propagate(
            X_train, y_train, X_type='train', require_model_grad=True)
        loss_train.backward()
        self.model_optimizer.step()
        return loss_train.item()

    def update_prep_pipeline(self, X_train, y_train, X_val, y_val):
        """Update prep pipeline parameters"""
        self.prep_pipeline_optimizer.zero_grad()

        dval_dalpha, dval_dw = self.compute_dval(X_train, y_train, X_val, y_val)
        hessian_product = self.compute_hessian_product(X_train, y_train, dval_dw)

        for i, alpha in enumerate(self.prep_pipeline.parameters()):
            dval = dval_dalpha[i]
            dtrain = hessian_product[i]
            dalpha = dval - self.model_optimizer.param_groups[0]['lr'] * dtrain

            if alpha.grad is None:
                alpha.grad = Variable(dalpha.data.clone())
            else:
                alpha.grad.data.copy_(dalpha.data.clone())

        self.prep_pipeline_optimizer.step()

    def compute_dval(self, X_train, y_train, X_val, y_val):
        """Compute gradient of validation loss w.r.t. prep pipeline"""
        model_backup = deepcopy(self.model.state_dict())
        self.update_model(X_train, y_train)
        self.model_optimizer.zero_grad()
        _, loss_val = self.forward_propagate(
            X_val, y_val, X_type='val', require_transform_grad=True, require_model_grad=True)
        loss_val.backward(retain_graph=True)

        dval_dalpha = [param.grad.data.clone() for param in self.prep_pipeline.parameters()]
        dval_dw = [param.grad.data.clone() for param in self.model.parameters()]
        self.model.load_state_dict(model_backup)
        return dval_dalpha, dval_dw

    def compute_hessian_product(self, X_train, y_train, dval_dw):
        """Compute Hessian-vector product for prep pipeline update"""
        model_backup = deepcopy(self.model.state_dict())
        eps = 0.001 * _concat(self.model.parameters()).data.detach().norm() / _concat(dval_dw).data.detach().norm()

        for w, dw in zip(self.model.parameters(), dval_dw):
            w.data += eps * dw
        _, loss_train = self.forward_propagate(X_train, y_train, X_type='train', require_transform_grad=True)
        grads_p = torch.autograd.grad(loss_train, self.prep_pipeline.parameters(), retain_graph=True, allow_unused=True)

        for w, dw in zip(self.model.parameters(), dval_dw):
            w.data -= 2 * eps * dw
        _, loss_train = self.forward_propagate(X_train, y_train, X_type='train', require_transform_grad=True)
        grads_n = torch.autograd.grad(loss_train, self.prep_pipeline.parameters(), retain_graph=True, allow_unused=True)

        hessian_product = [(x - y).div_(2 * eps.cpu()) for x, y in zip(grads_p, grads_n)]
        self.model.load_state_dict(model_backup)
        return hessian_product

    def log_prep_pipeline(self, global_step):
        """Log prep pipeline state"""
        self.writer.add_pipeline(self.prep_pipeline.pipeline, global_step)
        if self.params["method"] in ["diffprep_flex"]:
            self.writer.add_pipeline_alpha(self.prep_pipeline.alpha_probs, global_step)
