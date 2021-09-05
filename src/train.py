import numpy as np
import pandas as pd
import argparse
import config
import joblib
import model_dispatcher
from sklearn import metrics
import os

df = pd.read_csv(config.TRAIN_FOLDS)


def run(fold, arg_model):
    df = pd.read_csv(config.TRAIN_FOLDS)
    features = [
        f for f in df.columns if f not in ("PassengerId", "Survived", "kfold")
    ]

    # creating the training set
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # creating the validation set
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    X_train = df_train[features]
    X_valid = df_valid[features]
    y_valid = df_valid.Survived

    model = model_dispatcher.models[arg_model]
    model.fit(X_train, df_train.Survived)
    preds = model.predict(X_valid)
    score = metrics.accuracy_score(y_valid, preds)
    # print auc
    print(f"Fold = {fold}, F1 score = {score}")

    # save the model
    joblib.dump(
        model,
        os.path.join(config.MODEL_OUTPUT, f"rmc/{arg_model}_{fold}.bin")
    )
    return score

if __name__ == "__main__":
    scores = []
    ap = argparse.ArgumentParser()

    ap.add_argument("-m", "--model", required=True)
    args = ap.parse_args()
    for fold_ in range(5):
        score = run(fold=fold_, arg_model=args.model)
        scores.append(score)

