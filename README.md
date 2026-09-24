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

## Modelos de tendencia (LSTM / XGBoost / Transformer)

`modelos/lstm_trend_model.py`, `modelos/xgboost_trend_model.py` y
`modelos/transformer_trend_model.py` son
reimplementaciones independientes de los modelos de tendencia de `api-ml`
(mismas features, mismo target). No usan los modelos deprecados de
`api-ml`: consumen datos frescos de `data-colector` y guardan sus artefactos
de produccion en el Modal Volume `trend-model-artifacts`.

Los cinco modelos con cron propio (el Transformer no: ver abajo) tienen jobs programados de lunes a viernes,
despues del cierre del mercado (zona horaria de Buenos Aires): LSTM 20:00,
XGBoost 21:00, SVM 21:30, ARIMA 22:00 y GARCH 23:00. Para cada ticker (y
cada horizonte, cuando aplica) el job:

1. entrena un candidato sin las ultimas 60 ruedas;
2. mide accuracy direccional y MAE sobre ese holdout temporal;
3. promueve de forma atomica el modelo si su accuracy supera al artefacto
   de produccion actual;
4. reentrena sobre toda la historia antes de persistir el promovido.

**Transformer (sin cron propio):** el plan gratis de Modal permite 5
funciones programadas por workspace y ya estan las 5 ocupadas, asi que
`transformer_trend_model.py` no tiene `schedule=`. El job del LSTM (20:00)
lo dispara con `.spawn()` al arrancar, y corre en paralelo en su propio
contenedor con el mismo criterio de promocion. El LSTM tiene que estar
desplegado para que esto ocurra; si el Transformer no esta desplegado, el
LSTM lo ignora y sigue. Si algun dia se libera un cron (o se pasa a un plan
pago), se le puede volver a poner un `modal.Cron` propio.

En cada corrida se consulta `/available-tickers` de `data-colector`: todos
los tickers disponibles se entrenan automaticamente para los horizontes 1
a 5. Si un ticker no tiene historia suficiente, su error queda registrado
y el lote continua con el siguiente.

```
modal serve modelos/lstm_trend_model.py
modal serve modelos/xgboost_trend_model.py
modal serve modelos/transformer_trend_model.py
```

Para activar endpoints y cron hay que desplegar ambas apps (un `serve` no
es un deployment permanente):

```
modal deploy modelos/lstm_trend_model.py
modal deploy modelos/xgboost_trend_model.py
modal deploy modelos/transformer_trend_model.py
modal deploy modelos/svm_model.py
modal deploy modelos/arima_model.py
modal deploy modelos/garch_model.py
```

Despues del primer deploy se debe crear el set inicial sin esperar a la
proxima ejecucion programada:

```
modal run modelos/lstm_trend_model.py::retrain_models
modal run modelos/xgboost_trend_model.py::retrain_models
modal run modelos/transformer_trend_model.py::retrain_models
modal run modelos/svm_model.py::retrain_models
modal run modelos/arima_model.py::retrain_models
modal run modelos/garch_model.py::retrain_models
```

Se consultan con el ticker como parametro:
```
https://financiar186--lstm-trend-model-main-dev.modal.run/?ticker=GGAL
https://financiar186--xgboost-trend-model-main-dev.modal.run/?ticker=GGAL
https://financiar186--transformer-trend-model-main-dev.modal.run/?ticker=GGAL
```

El endpoint solo hace feature engineering e inferencia. Nunca entrena. Si
todavia no existe un artefacto para el ticker/horizonte pedido responde un
error inmediato en vez de bloquear la API durante minutos.

## Seguimiento operativo

Las instrucciones para controlar semanalmente los jobs, interpretar sus
resultados, probar produccion y responder ante fallas estan en
[`SEGUIMIENTO_SEMANAL.md`](SEGUIMIENTO_SEMANAL.md).

La explicacion de los tests, el backtest walk-forward y el paper trading de
`api-ml`, junto con el proceso para seguir la calidad predictiva semana a
semana, esta en [`SEGUIMIENTO_CALIDAD.md`](SEGUIMIENTO_CALIDAD.md).
