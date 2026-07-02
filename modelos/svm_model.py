import modal
import numpy as np
import pandas as pd
from sklearn import svm
import requests
import json

REG = None

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "arch", "pandas" ,"numpy", "scikit-learn", "requests")
app = modal.App("svm-model")

@app.function()
def get_ticker_data_and_transform(ticker: str):
    returns = requests.post(f"https://data-colector.onrender.com/historical-data/{ticker}")
    json_returns =  json.loads(returns.text)
    df = pd.DataFrame(json_returns["data"])
    df.index = df['ts']
    df['close_amount'] = df['close_amount'].astype(float)
    df['open_amount'] = df['open_amount'].astype(float)
    df['high_amount'] = df['high_amount'].astype(float)
    df['low_amount'] = df['low_amount'].astype(float)

    df['Open-Close'] = df['close_amount'] - df['open_amount']
    df['High-Low'] = df['high_amount'] - df['low_amount']
    return df

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
    df = get_ticker_data_and_transform.local(ticker)
    X = df[['Open-Close', 'High-Low']]
    y = np.where(df['close_amount'].shift(-1) > df['close_amount'], 1, 0)
    model = train_classifier.local(X, y)
    df['Predicted_Signal'] = model.predict(X)

    if df.iloc[-1]['Predicted_Signal'] == 0:
        return {"prediction": "Buy"}
    else:
        return {"prediction" : "Sell"}