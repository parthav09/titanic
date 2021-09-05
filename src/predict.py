import os.path

import pandas as pd
import numpy as np
from sklearn import preprocessing
import config
import joblib
def get_preds(test):
    test_preds = 0
    model_path = os.path.join(config.MODEL_OUTPUT, "rmc/")
    for fold in range(5):
        model = joblib.load(os.path.join(model_path, f"rmc_{fold}.bin"))
        temp_preds = model.predict(test)
        test_preds += temp_preds / 5
    return test_preds
def main():
    df_test = pd.read_csv(config.TEST_CLEANED)
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION)
    sample_sub['PassengerId'] = df_test['PassengerId']

    features = [
        f for f in df_test.columns if f not in ("PassengerId", "Survived")
    ]

    print(df_test.head())
    test = df_test[features].values
    final_preds = get_preds(test)
    print(len(final_preds))
    sample_sub['Survived'] = [int(i) for i in final_preds]

    print(sample_sub.head(10))

    sample_sub.to_csv("../submissions/rmc.csv", index=False)

if __name__ == "__main__":
    main()