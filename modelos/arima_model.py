import modal
import numpy as np
import pandas as pd
from arch import arch_model
import requests
import json
from statsmodels.tsa.arima.model import ARIMA

REG = None

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "arch", "pandas" ,"numpy", "requests", "statsmodels")
app = modal.App("arima-model")

@app.function()
def get_ticker_data(ticker: str):
    returns = requests.post(f"https://data-colector.onrender.com/historical-data/{ticker}")
    json_returns =  json.loads(returns.text)
    closing = [float(x["close_amount"]) for x in json_returns["data"]]
    return closing

@app.function()
def train_model(datos, ticker: str, steps: int, media_movil: int):
    serie = pd.Series(datos)
    modelo = ARIMA(serie, order=(1, 1, media_movil))

    resultado = modelo.fit()

    print(resultado.summary())

    pronostico = resultado.forecast(steps=steps)
    return pronostico


@app.function(image=image)
@modal.fastapi_endpoint()
def main(ticker: str, predictions: int):
    datos = get_ticker_data.local(ticker)
    return {"prediction": train_model.local(datos, ticker, predictions, 7).to_list(),
    "valor_actual": datos[-1],
    "cant_predicciones": predictions}