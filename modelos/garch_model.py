import modal
import numpy as np
import pandas as pd
from arch import arch_model
import requests
import json

REG = None

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "arch", "pandas" ,"numpy", "requests")
app = modal.App("garch-model")

@app.function()
def get_ticker_data(ticker: str):
    returns = requests.post(f"https://data-colector.onrender.com/historical-data/{ticker}")
    json_returns =  json.loads(returns.text)
    closing = [float(x["close_amount"]) for x in json_returns["data"]]
    return closing

@app.function()
def train_model(ticker: str):
    returns = get_ticker_data.local(ticker)

    # 2. Definir el modelo GARCH(1,1)
    # Por defecto incluye una media constante
    am = arch_model(returns, vol='Garch', p=1, q=1)

    # 3. Ajustar el modelo a los datos
    res = am.fit(update_freq=5)

    # 4. Mostrar el resumen de los resultados
    print(res.summary())

    # 5. Pronosticar la volatilidad para los próximos 5 días
    forecasts = res.forecast(horizon=5)
    return forecasts.variance[-1:]

@app.function(image=image)
@modal.fastapi_endpoint()
def main(ticker: str):
    return {"prediction": train_model.local(ticker).to_dict()}