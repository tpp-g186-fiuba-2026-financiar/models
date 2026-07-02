import modal
import numpy as np
import pandas as pd
from sklearn import svm

REG = None

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "arch", "pandas" ,"numpy", "scikit-learn")
app = modal.App("svm-model")

@app.function()
def train_classifier(x, y):
    # Given training data x and lavels y trains a new svm classifier
    # X can be 2 dimensional, and y has to be floating values
    # Return models
    clf = svm.SVC()
    clf.fit(x, y)
    return clf

@app.function()
def train_regressor(x, y):
    # Given training data x and lavels y trains a new svm regression
    # X can be 2 dimensional, and y has to be floating values
    clf = svm.SVR()
    clf.fit(x, y)
    return clf

@app.function(image=image)
@modal.fastapi_endpoint()
def main(ticker: str):
    # Not yet defined
    return {"prediction": 10}