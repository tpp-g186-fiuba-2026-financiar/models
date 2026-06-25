import modal
import numpy as np
import pandas as pd
from arch import arch_model

REG = None

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "arch", "pandas" ,"numpy")
app = modal.App("garch-model")

@app.function()
def train_model():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 1, 1000)) 

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
    return {"prediction": train_model.local().to_dict()}