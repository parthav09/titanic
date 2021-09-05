from sklearn.linear_model import LogisticRegression
from sklearn import ensemble

models = {
    "logisticreg": LogisticRegression(),
    "rmc": ensemble.RandomForestClassifier(max_depth=3)
}