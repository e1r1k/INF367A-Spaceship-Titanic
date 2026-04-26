import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from scipy.linalg import qr
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
import pandas as pd

def generate_random_rotation(n_features, rng):
    """
    Generates a uniformly distributed random rotation matrix R
    Uses the Indirect Method
    """

    # Create an n x n matrix of independent standard normal variates
    A = rng.standard_normal(size=(n_features, n_features))

    # Householder QR decomposition
    Q, R = qr(A)

    # Ensure a proper rotation matrix
    # If the determinant is -1, flip the sign of the first column
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

# Code inspired from https://tommyodland.com/articles/2023/random-rotation-ensembles-in-python/index.html
class RandomRotationEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, random_state=None, base_learner=None, use_rotation=True, **tree_params):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.base_learner = base_learner
        self.use_rotation = use_rotation
        self.tree_params = tree_params

    def get_params(self, deep=True):
        params = {
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "base_learner": self.base_learner,
            "use_rotation": self.use_rotation,
        }
        params.update(self.tree_params)
        return params
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ["n_estimators", "random_state", "base_learner", "use_rotation"]:
                setattr(self, key, value)
            else:
                self.tree_params[key] = value
        return self
    
    def _prepare_data(self, X, R=None, rng=None):
        """ 
        Helper used to seperate, rotate and recombine features 
        """
        num_cols = [c for c in X.columns if X[c].nunique() >= 10]
        cat_cols = [c for c in X.columns if c not in num_cols]

        if self.use_rotation and len(num_cols)>0:
            if R is None:
                R = generate_random_rotation(len(num_cols), rng)

            # Rotate only numerical features
            X_num_rotated = X[num_cols].values @ R
            X_out = np.hstack([X_num_rotated, X[cat_cols].values])
            return X_out, R
        else:
            return X.values, None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.rotations_ = []

        for _ in range(self.n_estimators):
            X_rotated, R = self._prepare_data(X, rng=rng)

            tree_seed = rng.integers(0, 2**32 -1)

            if self.base_learner == "extra":
                tree = ExtraTreeClassifier(random_state=tree_seed, **self.tree_params)
                tree.fit(X_rotated, y)
            elif self.base_learner == "rf":
                tree = DecisionTreeClassifier(random_state=tree_seed, max_features="sqrt", **self.tree_params)
                # Adding a bootstrapping step to create the "Bagging" effect of a Random Forest
                indices = rng.integers(0, X.shape[0], X.shape[0])

                X_sample = X_rotated.values[indices] if hasattr(X_rotated, "values")else X_rotated[indices]
                y_sample = y.values[indices] if hasattr(y, "values") else y[indices]
                tree.fit(X_sample, y_sample)
            else: 
                # Ensemble of trees with rotation as only source of diversity 
                # expect for internal differences from their seeds
                tree = DecisionTreeClassifier(random_state=tree_seed, **self.tree_params)
                tree.fit(X_rotated, y)
            
            # Store the tree and its specific rotation
            self.trees_.append(tree)
            self.rotations_.append(R)

        return self     

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        all_probs = []
        # Each tree must see the test data in its specific rotated space
        for tree, R in zip(self.trees_, self.rotations_):
            X_rotated, _ = self._prepare_data(X, R=R)
            all_probs.append(tree.predict_proba(X_rotated))
        
        # Aggregate by averaging
        return np.mean(all_probs, axis = 0)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
        
