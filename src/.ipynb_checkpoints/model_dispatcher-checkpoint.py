from sklearn.linear_model import LogisticRegression
from sklearn import ensemble

models = {
    "logisticreg": LogisticRegression(verbose = False, n_jobs = 100),
    "rmc": ensemble.RandomForestClassifier(max_depth=3)
}