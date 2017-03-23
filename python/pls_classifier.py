import numpy as np
import sklearn
import warnings

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.linear_model import LinearRegression


def _to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical models
    '''
    if not nb_classes:
        if 0 in y:
            nb_classes = np.max(y) + 1
        else:
            nb_classes = np.max(y)
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


class PLSClassifier(BaseEstimator, ClassifierMixin):
    __name__ = 'MultiLayeredPLS'

    def __init__(self, estimator=None, n_iter=1500, eps=1e-6, n_comp=10, mode='regression'):
        warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

        self.n_iter = n_iter
        self.eps = eps
        self.n_comp = n_comp
        self.mode = mode
        self.estimator = estimator

        self.estimator_ = None
        self.pls = None

    def fit(self, X, y):
        # if X is not np.array or y is not np.array:
        #     print('x and y must be of type np.array')
        #     raise ValueError
        if X.shape[0] != y.shape[0]:
            raise ValueError()

        if self.estimator is None:
            self.estimator_ = LinearRegression()
        else:
            self.estimator_ = sklearn.base.clone(self.estimator_)

        self.classes_, target = np.unique(y, return_inverse=True)

        target[target == 0] = -1

        if self.mode == 'canonical':
            self.pls = PLSCanonical(n_components=self.n_comp, scale=True, max_iter=self.n_iter, tol=self.eps)
        elif self.mode == 'regression':
            self.pls = PLSRegression(n_components=self.n_comp, scale=True, max_iter=self.n_iter, tol=self.eps)
        proj_x, proj_y = self.pls.fit_transform(X, target)

        self.estimator_.fit(proj_x, target)

        return self

    def predict_value(self, x):
        resp = self.decision_function(x)
        if resp.ndim == 1:
            ans = np.zeros(resp.shape, dtype=np.int32)
            ans[resp > 0] = self.classes_[1]
            ans[resp <= 0] = self.classes_[0]
        else:
            ans = self.classes_[np.argmax(resp, axis=1)]

        return ans

    def predict_confidence(self, x):
        resp = self.decision_function(x)
        return resp[0]

    def decision_function(self, x):
        x = np.array(x).reshape((1, -1))
        proj = self.pls.transform(x)
        resp = self.estimator_.predict(proj)
        return resp

    def predict_proba(self, x):
        resp = self.decision_function(x)
        resp = np.min(-1, resp)
        resp = np.max(1, resp)
        resp -= 1
        resp /= 2
        # resp = np.exp(resp)
        # for r in range(len(resp)):
        #     resp[r] /= np.sum(resp[r])

        return resp
