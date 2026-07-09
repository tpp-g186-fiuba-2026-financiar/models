# Modelos

## Modal Set-Up
Primero para linkear tu computadora a Modal se debe hacer:
```
pip install modal
```
La version de pip debe ser al menos 26 para que funcione
```
python3 -m modal setup
```
## Corriendo Modal
Una vez que se tiene linkeado la computadora con la cuenta se debe correr:
```
modal serve example.py
```
Si modal no es un comando reconocido entonces se debe usar:
```
python -m modal serve example.py
```
## Accediendo a la pagina
Si se esta corriendo con serve se puede acceder en la ruta:
```
https://financiar186--example-get-started-main-dev.modal.run/?x=10
```
Donde x es el parametro de la funcion

## Modelos de tendencia (LSTM / XGBoost)

`modelos/lstm_trend_model.py` y `modelos/xgboost_trend_model.py` son
reimplementaciones independientes de los modelos de tendencia de `api-ml`
(mismas features, mismo target). Igual que el resto de `modelos/*.py`
(arima/garch/svm): archivo autocontenido, entrena on-demand solo con el
ticker pedido via `data-colector`, sin pooling entre tickers ni artefactos
persistidos -- asi no dependen de que el servicio de `api-ml` en Render
este arriba.

```
modal serve modelos/lstm_trend_model.py
modal serve modelos/xgboost_trend_model.py
```

Se consultan con el ticker como parametro:
```
https://financiar186--lstm-trend-model-main-dev.modal.run/?ticker=GGAL
https://financiar186--xgboost-trend-model-main-dev.modal.run/?ticker=GGAL
```