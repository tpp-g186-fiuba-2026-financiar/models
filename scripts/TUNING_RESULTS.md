# Búsqueda de ventana, horizonte y arquitectura (issue #160)

## Qué se hizo

Antes los valores de `WINDOW`/`HORIZON`/arquitectura en `lstm_trend_model.py`
y `xgboost_trend_model.py` estaban puestos a ojo (30 días de ventana,
horizonte de 5 días, LSTM de 1 capa). Se corrió `tune_hyperparams.py`, que
prueba varias combinaciones con un backtest walk-forward de verdad: entrena
con los datos hasta un punto, predice hacia adelante sin mirar el futuro,
reentrena más adelante y sigue. Se usó GGAL, YPFD y ALUA (bancos/energía/
industria, con historial completo), 200 ruedas de evaluación por ticker,
reentrenando cada 40 días.

Métrica: cuántas veces la dirección predicha (sube/baja) coincidió con la
real, sobre ~570-591 predicciones en total (las 3 acciones juntas).

**Ojo con esto:** 570 predicciones y 3 tickers es una muestra chica para
sacar conclusiones firmes — las diferencias de 1-2 puntos porcentuales entre
combinaciones pueden ser ruido. Los cambios que se aplicaron son los que
salieron consistentemente mejor en más de una prueba, no el número más alto
de una sola tabla.

## LSTM — ventana x horizonte (arquitectura por defecto: 32 neuronas, 1 capa)

| ventana | horizonte | acierto | error promedio |
|---|---|---|---|
| 20 | 3 | 53.3% | 0.0513 |
| 20 | 5 | 53.3% | 0.0654 |
| 20 | 7 | 51.3% | 0.0891 |
| 20 | 10 | 53.2% | 0.1042 |
| 30 | 3 | 53.1% | 0.0498 |
| **30** | **5 (el que usábamos)** | **56.1%** | **0.0702** |
| 30 | 7 | 53.4% | 0.0793 |
| 30 | 10 | 54.9% | 0.1103 |
| 45 | 3 | 53.3% | 0.0468 |
| 45 | 5 | 51.1% | 0.0659 |
| 45 | 7 | 51.8% | 0.0869 |
| **45** | **10 (nuevo)** | **56.8%** | **0.0964** |

## LSTM — arquitectura (con ventana=45, horizonte=10)

| hidden | capas | acierto | error promedio |
|---|---|---|---|
| 16 | 1 | 47.7% | 0.0965 |
| 16 | 2 | 55.8% | 0.0934 |
| **32** | **1 (el que usábamos)** | **56.8%** | **0.0964** |
| **32** | **2 (nuevo)** | **63.3%** | **0.0950** |
| 64 | 1 | 55.1% | 0.1083 |
| 64 | 2 | 52.5% | 0.1069 |

**32 neuronas + 2 capas se despega claro del resto** (63.3% contra 56.8% con
1 capa) — es el cambio más grande de toda la búsqueda.

## XGBoost — ventana x horizonte (arquitectura por defecto: 300 árboles, profundidad 4)

| ventana | horizonte | acierto | error promedio |
|---|---|---|---|
| 20 | 3 | 53.3% | 0.0442 |
| 20 | 5 | 54.4% | 0.0585 |
| 20 | 7 | 55.8% | 0.0689 |
| 20 | 10 | 58.6% | 0.0825 |
| **30** | **5 (el que usábamos)** | **56.1%** | **0.0565** |
| 30 | 7 | 55.4% | 0.0672 |
| **30** | **10 (nuevo)** | **59.8%** | **0.0811** |
| 45 | 3 | 51.6% | 0.0447 |
| 45 | 5 | 48.5% | 0.0585 |
| 45 | 7 | 51.3% | 0.0705 |
| 45 | 10 | 52.8% | 0.0848 |

## XGBoost — arquitectura (con ventana=30, horizonte=10)

| árboles | profundidad | acierto |
|---|---|---|
| 200 | 3 | 56.7% |
| 200 | 4 | 57.7% |
| 200 | 5 | 59.6% |
| 300 | 3 | 56.7% |
| **300** | **4 (la que ya usábamos)** | **59.8%** |
| 300 | 5 | 57.9% |
| 450 | 3 | 56.7% |
| 450 | 4 | 57.9% |
| 450 | 5 | 58.1% |

La arquitectura actual (300/4) ya era la mejor de todas una vez cambiado el
horizonte — no hizo falta tocarla.

## Qué se cambió (después de la ronda 1)

- **`WINDOW` de LSTM: 30 → 45 días.** XGBoost se queda en 30 (fue lo que
  mejor dio para ese modelo).
- **LSTM: 1 capa → 2 capas** (se mantienen las 32 neuronas). El cambio más
  grande de toda la búsqueda.
- XGBoost se deja con la arquitectura que ya tenía (300 árboles, profundidad
  4).
- El horizonte de 10 días **se descartó después**, ver ronda 2 más abajo —
  predecía mejor pero el equipo lo vio poco útil para un cliente real
  ("¿en 10 días subió o bajó?" dice poco día a día).

---

## Ronda 2 — features nuevas del EDA + horizonte acotado a 1-5 días

`scripts/tune_features_and_horizon.py`. Dos cambios de rumbo respecto a la
ronda 1:

1. **El horizonte de 10 días se sacó.** Predecía mejor en el backtest, pero
   no es útil para el producto — se volvió a probar el rango 1-5 días (el
   que sirve de verdad) y el horizonte quedó como **parámetro de `main()`**,
   no una constante fija. Quien pide la predicción elige entre 1 y 5 días.
2. **Se probaron las 2 features que salieron del EDA** (issue #135,
   `api-ml/notebooks/eda.ipynb` secciones 13.b y 13.c) como candidatas
   reales, no solo como idea: `dist_max60` (distancia al máximo de 60
   ruedas) y `rsi_vol_interaction` (RSI x volumen "raro", z-score contra el
   propio promedio de 60 días).

Mismo método que la ronda 1 (walk-forward, GGAL/YPFD/ALUA), reentrenando
cada 50 días esta vez (en vez de 40) para acotar el tiempo de corrida.

### ¿Las features nuevas ayudan? (con ventana=45 LSTM / ventana=30 XGBoost, horizonte=5)

| Modelo | Features | Acierto | Error promedio |
|---|---|---|---|
| LSTM | 8 (actuales) | 50.4% | 0.0771 |
| **LSTM** | **10 (con las 2 nuevas)** | **58.3%** | **0.0699** |
| **XGBoost** | **8 (actuales)** | **54.7%** | **0.0571** |
| XGBoost | 10 (con las 2 nuevas) | 52.5% | 0.0564 |

Resultado dispar a propósito: **en LSTM ayudan bastante** (+7.9 puntos), **en
XGBoost empeoran un poco** (-2.2 puntos) — probablemente porque los árboles
de XGBoost ya capturan una interacción como RSI×volumen solos con sus
propios splits, y sumarla ya calculada de más le resta en vez de sumar. Por
eso quedó una diferencia real entre los dos archivos: `lstm_trend_model.py`
tiene 10 features, `xgboost_trend_model.py` se queda con las 8 de siempre.

(Nota: la primera corrida de XGBoost con 10 features tardó 737 segundos
contra los 9s normales — se sospechó un cuelgue del sistema, no del código,
y se repitió aislada: dio exactamente el mismo 52.5% en 9 segundos. El
número es real, el tiempo raro fue una casualidad de esa corrida.)

### Horizonte 1 a 5 días

| Horizonte | LSTM (10 features) | XGBoost (8 features) |
|---|---|---|
| 1 día | 51.1% (error 0.023) | 51.1% (error 0.022) |
| 2 días | 50.0% (error 0.040) | 51.0% (error 0.034) |
| 3 días | 53.0% (error 0.052) | 51.8% (error 0.044) |
| 4 días | 53.9% (error 0.065) | 52.0% (error 0.051) |
| 5 días | 58.3% (error 0.070) | 54.7% (error 0.057) |

El acierto crece con el horizonte en los dos modelos, incluso acotado a
1-5 — por eso el default quedó en 5. Pero ojo con el trade-off: a 1 día el
error de magnitud es un tercio del de 5 días, aunque acertar la dirección
ahí sea casi como tirar una moneda (~51%). Por eso quedó elegible como
parámetro y no fijo: para "¿conviene mirar esto ya?" capaz sirve más un
horizonte corto con error chico; para "¿hacia dónde va la tendencia?" un
horizonte de 5 acierta más la dirección.

## Qué se cambió (después de la ronda 2)

- **LSTM (`lstm_trend_model.py`): 8 → 10 features**, suma `dist_max60` y
  `rsi_vol_interaction`.
- **XGBoost (`xgboost_trend_model.py`): sin cambios en features**, se
  probaron las mismas 2 y dieron peor.
- **`HORIZON` deja de ser una constante fija** en los dos archivos. Pasa a
  ser un parámetro de `main(ticker, horizon=...)`, validado entre
  `MIN_HORIZON=1` y `MAX_HORIZON=5`, con `DEFAULT_HORIZON=5`.

## Qué falta

Esto valida la idea con 3 tickers. Antes de darlo por cerrado del todo,
convendría correrlo con más tickers (o los 22 del universo completo) y con
más ruedas de evaluación, para confirmar que el salto de 1 a 2 capas en
LSTM, las 2 features nuevas, y que el acierto siga subiendo con el
horizonte, se sostienen y no es ruido de una muestra chica. También falta
que backend-website/frontend expongan el selector de horizonte al usuario
final — hoy el parámetro existe en el modelo pero nadie en la capa de
arriba lo está mandando todavía.
