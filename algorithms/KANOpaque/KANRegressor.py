from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score
import torch 
from kan import *


class KANRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, width=[3,3,1], grid=5, k=3, symbolic=[], prune=False):
        self.width = width 
        self.grid = grid 
        self.k = k
        self.symbolic = symbolic
        self.prune = prune
        self.is_fitted = False

    def fit(self, X_train, y_train):
        dim = 1 if len(X_train.shape) == 1 else X_train.shape[1]
        self.model = KAN(width=[dim] + self.width, grid=5, k=2, device=torch.device('cpu'))
        dataset = {'train_input': torch.from_numpy(X_train),
                   'train_label': torch.from_numpy(y_train),
                   'test_input' : torch.from_numpy(X_train),
                   'test_label' : torch.from_numpy(y_train),
                  }
        self.model.double()
        self.model.train(dataset, opt="LBFGS", steps=50)
        if self.prune:
            self.model.prune()
        if len(self.symbolic) > 0:
            self.model.auto_symbolic(lib=self.symbolic)
            self.model.train(dataset, opt="LBFGS", steps=50)
        self.is_fitted = True

    def predict(self, X_test):
         #check_is_fitted(self)
         X_test = check_array(X_test, accept_sparse=False)
         return self.model(torch.from_numpy(X_test))
    def score(self, X_test, y_test):
        X_test = check_array(X_test, accept_sparse=False)
        y_hat = self.model(torch.from_numpy(X_test))
        return r2_score(y_test, y_hat.cpu().detach().numpy())
