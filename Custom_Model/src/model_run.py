import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
import argparse
mlflow.set_tracking_uri("http://localhost:5000")
parser =  argparse.ArgumentParser()
parser.add_argument("--threshold" , type=float ,  default=0.5)
args_ =parser.parse_args()
threshold = args_.threshold

clf = mlflow.pyfunc.load_model("models:/Mymodel/Production")

x_test = pd.DataFrame({"x1": np.arange(40, 51), "x2": np.arange(90, 101)})
preds = clf.predict(x_test , params={"threshold":threshold})
print("predictions") 
print(preds)


# mlflow run . -e model_run -P threshold=0.7 --env-manager local