from KANRegressor import KANRegressor
from sklearn.model_selection import  GridSearchCV

# Based on Bingo
hyper_params = {"width": [[5,1], [5,3,1],[3,3,1]], "prune": [True, False], "symbolic": [['x','x^2','x^3','x^4','exp','sin','logabs','sqrtabs']]}
#hyper_params = {"width": [[5,1]], "prune": [True], "symbolic": [['x','x^2','x^3','x^4','exp','sin','logabs','sqrtabs']]}

est_nocv = KANRegressor()

is_opaque = False

N_FOLDS = 3
est = GridSearchCV(est_nocv, hyper_params)

def model(est, X=None):
    return est.best_estimator_.model.symbolic_formula(var=X)
