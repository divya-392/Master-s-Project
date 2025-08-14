
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

class MBGWOSelector(BaseEstimator, TransformerMixin):
    """
    Modified Binary Grey Wolf Optimization (MBGWO) for feature selection.

    Enhancements:
    - Chaotic initialization (logistic map)
    - Adaptive convergence parameter a(t) decreasing 2 -> 0
    - Bit-flip mutation
    - CV-based fitness: alpha * CV-accuracy + (1 - alpha) * (1 - (#selected / total))
    """
    def __init__(self, n_wolves=20, n_iter=30, alpha_weight=0.8, pm=0.02,
                 base_estimator=None, cv=5, random_state=0, min_features=1):
        self.n_wolves = n_wolves
        self.n_iter = n_iter
        self.alpha_weight = alpha_weight
        self.pm = pm
        self.base_estimator = base_estimator
        self.cv = cv
        self.random_state = random_state
        self.min_features = min_features

    def _set_random(self):
        self._rng = np.random.RandomState(self.random_state)

    def _chaotic_init(self, n_wolves, n_features):
        x = self._rng.uniform(0.1, 0.9, size=(n_wolves, n_features))
        r = 3.99
        for _ in range(10):
            x = r * x * (1 - x)
        return (x > 0.5).astype(int)

    def _binarize(self, x):
        s = 1 / (1 + np.exp(-x))
        return (self._rng.rand(*x.shape) < s).astype(int)

    def _ensure_min_features(self, mask):
        if mask.sum() < self.min_features:
            off_idx = np.where(mask == 0)[0]
            self._rng.shuffle(off_idx)
            need = self.min_features - int(mask.sum())
            mask[off_idx[:need]] = 1
        return mask

    def _fitness(self, X, y, mask):
        if mask.sum() == 0:
            return 0.0
        Xs = X[:, mask == 1]
        est = self.base_estimator or LogisticRegression(max_iter=200, solver="liblinear")
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        acc = cross_val_score(est, Xs, y, cv=cv, scoring="accuracy").mean()
        size_term = 1.0 - (mask.sum() / mask.size)
        fit = self.alpha_weight * acc + (1 - self.alpha_weight) * size_term
        return fit

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self._set_random()

        n_features = X.shape[1]
        wolves = self._chaotic_init(self.n_wolves, n_features)
        fitness = np.array([self._fitness(X, y, w.copy()) for w in wolves])

        def rank_wolves(wolves, fitness):
            idx = np.argsort(fitness)[::-1]
            return wolves[idx[0]].copy(), wolves[idx[1]].copy(), wolves[idx[2]].copy(), fitness[idx[0]]

        alpha, beta, delta, best_fit = rank_wolves(wolves, fitness)
        self.history_ = [best_fit]

        pos = wolves.astype(float)

        for t in range(self.n_iter):
            a = 2 - 2 * (t / (self.n_iter - 1 + 1e-9))

            for i in range(self.n_wolves):
                for d in range(n_features):
                    r1, r2 = self._rng.rand(), self._rng.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[d] - pos[i, d])
                    X1 = alpha[d] - A1 * D_alpha

                    r1, r2 = self._rng.rand(), self._rng.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[d] - pos[i, d])
                    X2 = beta[d] - A2 * D_beta

                    r1, r2 = self._rng.rand(), self._rng.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[d] - pos[i, d])
                    X3 = delta[d] - A3 * D_delta

                    pos[i, d] = (X1 + X2 + X3) / 3.0

                bin_mask = self._binarize(pos[i, :])

                mut = self._rng.rand(n_features) < self.pm
                bin_mask[mut] = 1 - bin_mask[mut]

                bin_mask = self._ensure_min_features(bin_mask)

                f = self._fitness(X, y, bin_mask)

                if f >= fitness[i]:
                    wolves[i, :] = bin_mask
                    fitness[i] = f

            alpha, beta, delta, best_fit = rank_wolves(wolves, fitness)
            self.history_.append(best_fit)

        self.best_mask_ = alpha.copy()
        self.best_score_ = best_fit
        self.support_ = self.best_mask_.astype(bool)
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        if not hasattr(self, "support_"):
            raise RuntimeError("Call fit before transform.")
        return np.asarray(X)[:, self.support_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
