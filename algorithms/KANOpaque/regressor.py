from sklearn.model_selection import GridSearchCV
from KANRegressor import KANRegressor

# Based on Bingo
hyper_params = {"width": [[5,1], [5,3,1],[3,3,1]], "prune": [True, False]}
#hyper_params = {"width": [[5,1]], "prune": [True]}

est_nocv = KANRegressor()

is_opaque = True

N_FOLDS = 3
est = GridSearchCV(est_nocv, hyper_params)

def model(est, X=None):
    return None # est.best_estimator_.model.symbolic_formula(var=X)
