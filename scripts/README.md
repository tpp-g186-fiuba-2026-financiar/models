# Cómo se prueban cambios de hiperparámetros y features en este repo

Esto documenta el método que usamos para ajustar `lstm_trend_model.py` y `xgboost_trend_model.py` (issue #160), para que la próxima vez que alguien quiera probar un cambio (otra ventana, otro horizonte, otra feature, otra arquitectura) use el mismo proceso en vez de empezar de cero o decidir a ojo.

## Setup para correr esto local

Los modelos de `modelos/*.py` están pensados para correr en Modal, pero para probar combinaciones rápido no tiene sentido desplegar cada vez. Por eso hay un venv local aparte:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

`requirements-dev.txt` tiene lo necesario para entrenar y evaluar acá (torch, xgboost, pandas, etc.) — no es lo que corre en producción, eso lo define el `modal.Image` de cada archivo en `modelos/`.

## Los dos scripts

- **`tune_hyperparams.py`** — primera ronda: ventana de días, horizonte de predicción y arquitectura (capas/neuronas en LSTM, árboles/profundidad en XGBoost).
- **`tune_features_and_horizon.py`** — segunda ronda, reusa las funciones del primero: prueba features nuevas y acota el horizonte a un rango útil. Cualquier cosa nueva que se quiera probar en el futuro puede sumarse como una tercera ronda del mismo estilo, reusando las mismas funciones en vez de reescribir la lógica de entrenamiento y evaluación.

Se corren así:

```bash
.venv/bin/python scripts/tune_hyperparams.py
.venv/bin/python scripts/tune_features_and_horizon.py
```

## Cómo se separan los datos (esto es lo importante para reusar)

No alcanza con entrenar una vez y mirar qué tan bien predice sobre los mismos datos con los que entrenó — así cualquier combinación parece buena. El método que usamos es un backtest walk-forward:

1. **Tickers de prueba:** GGAL, YPFD, ALUA (bancos/energía/industria, elegidos porque tienen el historial completo, sin el problema de dato de ECOG).
2. **Se separan los últimos ~200 días de cada ticker** como si fueran "todavía no pasaron" — no se tocan para el primer entrenamiento.
3. **Dentro de esos ~200 días, se reentrena cada 40-50 días** (varió entre rondas), siempre usando solo los datos disponibles hasta ese punto — nunca datos de después. Este es el punto que no se puede romper: si se filtra un solo dato del futuro, el número de acierto queda inflado y no sirve para nada.
4. **Entre un reentrenamiento y el siguiente**, el modelo queda fijo y predice día por día hacia adelante, comparando la dirección que predijo contra la que realmente pasó.
5. Se cuenta cuántas veces acertó la dirección sobre el total de predicciones (across los 3 tickers) — ese es el % de acierto que se reporta. También se mide el error promedio (MAE) entre el retorno predicho y el real.

Esto simula lo más parecido posible a "si este modelo hubiera estado corriendo en vivo esos meses, ¿cuántas veces le achuntaba?" — y como es el mismo método para cada combinación que se prueba, las comparaciones son parejas entre sí.

**Antes de entrenar, los datos pasan por una limpieza:** si el historial de un ticker tiene un salto de precio imposible de explicar como movimiento real (ver el caso de ECOG más abajo), se capa ese retorno puntual y se reconstruye la serie de precios a partir de ahí, en vez de entrenar con el dato roto adentro o descartar el ticker entero. Esto vive en `clean_dataframe`/`clean_close_series`, duplicado en los dos archivos de `modelos/` (no en estos scripts, porque también tiene que correr en producción, no solo acá).

## Resultados Sprint 15

Cambié 3 cosas en `models` (LSTM y XGBoost), todo probado con el método de arriba, no a ojo:

1. **Ventana y arquitectura.** LSTM pasó de ventana=30/1 capa a ventana=45/2 capas — con eso el acierto direccional subió de ~57% a ~63%. XGBoost se quedó con lo que ya tenía (ventana=30, 300 árboles, profundidad 4), ya era lo mejor.

2. **Features nuevas del EDA (#135).** Probé las dos que salieron ahí (`dist_max60` y `rsi_vol_interaction`) de verdad, no solo la idea:
   - En LSTM ayudan bastante: 50.4% → 58.3% de acierto.
   - En XGBoost empeoran (54.7% → 52.5%), así que ahí no se agregaron.

3. **El horizonte de predicción ya no es fijo.** Había encontrado que 10 días predecía mejor que 5, pero lo descartamos porque es poco útil para el cliente: que la predicción diga "en 10 días va a subir" no le sirve mucho para decidir algo hoy. Ahora es un parámetro (`horizon=1..5`, default 5) que quien pide la predicción puede elegir. Con el rango acotado a 1-5, el acierto sigue subiendo hacia el 5, pero a 1 día el error de magnitud es un tercio — hay trade-off real entre los dos.

Encontramos también que ECOG tiene un dato roto en el historial: el 19/08/2025 el precio cae ~90% y el volumen sube ~10x el mismo día, y el nuevo nivel se sostiene las semanas siguientes sin rebote — compatible con un split 10:1 que no se ajustó retroactivamente en el histórico, no con un error aleatorio de un solo día. Antes esto rompía el entrenamiento; ahora se detecta y se repara solo (se capa ese retorno puntual y se reconstruye la serie de precios), sin descartar el ticker. La respuesta avisa `price_data_repaired_days` cuando pasa esto, para que no quede escondido.

Esto se probó con 3 tickers y ~570-590 predicciones — resultados prometedores pero no concluyentes. Antes de darlo por cerrado del todo, valdría la pena correrlo con más tickers.

## Ir sumando resultados de sprints siguientes

(Sprint 16 en adelante: agregar acá abajo qué se probó y qué cambió, mismo formato que arriba.)

## Qué features quedó usando cada modelo

De esta forma, LSTM entrena con estos 10 features:

1. `log_return` (cuánto subió o bajó el precio ese día respecto al día anterior)
2. `log_volume_change` (cuánto cambió la cantidad de operaciones ese día respecto al anterior)
3. `range_pct` (qué tan grande fue el vaivén del precio en el día, entre el máximo y el mínimo de esa rueda)
4. `rsi_norm` (si la acción está "cara" porque la vienen comprando mucho, o "barata" porque la vienen vendiendo mucho, en las últimas dos semanas)
5. `sma20_ratio` (qué tan lejos está el precio de hoy del promedio de los últimos 20 días)
6. `macd_norm` (si el impulso de corto plazo va para arriba o para abajo)
7. `momentum_10` (cuánto subió o bajó la acción en los últimos 10 días)
8. `macro_rate_chg5` (cómo cambió en la última semana una tasa de interés de referencia de Estados Unidos, que afecta si entra o sale plata de mercados como el argentino)
9. `dist_max60` (qué tan lejos está el precio de hoy del valor más alto que tocó en los últimos 3 meses)
10. `rsi_vol_interaction` (combina el 4 y el 2: si hubo un volumen raro de operaciones justo cuando la acción no estaba ni muy cara ni muy barata)

Y XGBoost con estos 8 (las mismas de siempre, sin las 2 últimas — se probaron ahí también y dieron peor):

1. `log_return` (cuánto subió o bajó el precio ese día respecto al día anterior)
2. `log_volume_change` (cuánto cambió la cantidad de operaciones ese día respecto al anterior)
3. `range_pct` (qué tan grande fue el vaivén del precio en el día, entre el máximo y el mínimo de esa rueda)
4. `rsi_norm` (si la acción está "cara" porque la vienen comprando mucho, o "barata" porque la vienen vendiendo mucho, en las últimas dos semanas)
5. `sma20_ratio` (qué tan lejos está el precio de hoy del promedio de los últimos 20 días)
6. `macd_norm` (si el impulso de corto plazo va para arriba o para abajo)
7. `momentum_10` (cuánto subió o bajó la acción en los últimos 10 días)
8. `macro_rate_chg5` (cómo cambió en la última semana una tasa de interés de referencia de Estados Unidos, que afecta si entra o sale plata de mercados como el argentino)
