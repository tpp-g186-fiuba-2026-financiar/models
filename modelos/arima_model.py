import modal
import numpy as np
import pandas as pd
from arch import arch_model
import requests
import json
from statsmodels.tsa.arima.model import ARIMA

REG = None

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "arch", "pandas" ,"numpy", "requests", "statsmodels")
app = modal.App("garch-model")

@app.function()
def get_ticker_data(ticker: str):
    returns = requests.post(f"https://data-colector.onrender.com/historical-data/{ticker}")
    json_returns =  json.loads(returns.text)
    closing = [float(x["close_amount"]) for x in json_returns["data"]]
    return closing

@app.function()
def train_model(ticker: str):
    datos = [10, 12, 14, 15, 18, 20, 22, 25, 28, 30]
    serie = pd.Series(datos)

    # 2. Definir el modelo ARIMA con los parámetros (p, d, q)
    # p = retardo autorregresivo, d = diferenciación, q = media móvil
    modelo = ARIMA(serie, order=(1, 1, 1))

    # 3. Ajustar el modelo
    resultado = modelo.fit()

    # 4. Ver el resumen estadístico
    print(resultado.summary())

    # 5. Hacer un pronóstico para los siguientes 3 pasos
    pronostico = resultado.forecast(steps=1)


@app.function(image=image)
@modal.fastapi_endpoint()
def main(ticker: str):
    return {"prediction": train_model.local(ticker).to_dict()}