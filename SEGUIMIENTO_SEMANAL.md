# Cómo controlar que los modelos de Modal estén funcionando

Esta guía sirve para responder tres preguntas:

1. ¿Los entrenamientos automáticos se ejecutaron?
2. ¿Los modelos nuevos mejoraron o se mantuvo el anterior?
3. ¿Los endpoints de producción responden?

Esto controla que el sistema funcione. Para saber si las predicciones fueron
buenas o malas, ver [`SEGUIMIENTO_CALIDAD.md`](SEGUIMIENTO_CALIDAD.md).

## Cuándo se entrena cada modelo

Modal ejecuta los entrenamientos de lunes a viernes:

| Modelo | Horario de Argentina |
| --- | --- |
| LSTM | 20:00 |
| XGBoost | 21:00 |
| SVM | 21:30 |
| ARIMA | 22:00 |
| GARCH | 23:00 |

No hace falta dejar una computadora encendida. Modal ejecuta todo en sus
servidores.

## Qué revisar cada lunes

### Paso 1: comprobar que las cinco aplicaciones estén activas

Ejecutar:

```bash
modal app list
```

Deben aparecer:

```text
lstm-trend-model
xgboost-trend-model
svm-model
arima-model
garch-model
```

Si aparecen las cinco como desplegadas, está bien. Si falta alguna, su
entrenamiento automático y su endpoint no están activos.

### Paso 2: mirar si entrenaron durante la semana

```bash
modal app logs lstm-trend-model --since 7d --timestamps
modal app logs xgboost-trend-model --since 7d --timestamps
modal app logs svm-model --since 7d --timestamps
modal app logs arima-model --since 7d --timestamps
modal app logs garch-model --since 7d --timestamps
```

En los logs aparece un resultado por ticker. Por ejemplo:

```json
{"ticker":"GGAL","promoted":true}
```

Significa que el modelo nuevo fue mejor y pasó a producción.

```json
{"ticker":"GGAL","promoted":false}
```

Significa que entrenó correctamente, pero no mejoró. Se descartó el candidato
y se sigue usando el modelo anterior. **No es un error.**

```json
{"ticker":"OIL","error":"..."}
```

Significa que falló ese ticker. Los demás continuaron entrenándose.

Resumen:

```text
promoted: true  → se usa el modelo nuevo
promoted: false → se conserva el modelo anterior
error           → falló ese ticker
App completed   → terminó la corrida
```

Si aparece un traceback y nunca aparece `App completed`, falló la corrida
completa.

### Paso 3: probar que producción responda

Copiar y ejecutar:

```bash
curl --fail --show-error "https://matimorales01--lstm-trend-model-main.modal.run?ticker=GGAL&horizon=5"
curl --fail --show-error "https://matimorales01--xgboost-trend-model-main.modal.run?ticker=GGAL&horizon=5"
curl --fail --show-error "https://matimorales01--svm-model-main.modal.run?ticker=GGAL"
curl --fail --show-error "https://matimorales01--arima-model-main.modal.run?ticker=GGAL&predictions=5"
curl --fail --show-error "https://matimorales01--garch-model-main.modal.run?ticker=GGAL"
```

Las cinco llamadas deben devolver una predicción.

También deben mostrar:

```json
"model_version": "..."
```

y:

```json
"backtest": {...}
```

En LSTM y XGBoost además debe aparecer:

```json
"source": "modal (artefacto preentrenado)"
```

Eso confirma que el endpoint utilizó el modelo guardado y no entrenó durante
la llamada.

## ¿Una versión vieja significa que está fallando?

No necesariamente.

Por ejemplo, si durante toda la semana aparece:

```json
"promoted": false
```

Modal entrenó modelos nuevos, pero ninguno superó al que ya estaba activo. Por
eso `model_version` puede seguir mostrando una fecha anterior.

Sólo hay un problema si:

- no aparecen corridas nuevas en los logs;
- aparece un error completo;
- el endpoint no responde;
- el artefacto no existe.

## Error conocido de OIL

`OIL` tiene precios inválidos para los cálculos logarítmicos. Actualmente puede
fallar en:

- LSTM;
- XGBoost;
- GARCH.

Ese error ya es conocido. Si falla otro ticker o si `OIL` empieza a provocar
la caída del lote completo, hay que investigarlo.

## Qué hacer si algo falla

### Falló un solo ticker

Si aparece:

```json
{"ticker":"XXX","error":"..."}
```

1. Revisar si ocurrió una sola vez o varios días seguidos.
2. Comprobar si `data-colector` tiene histórico para ese ticker.
3. Si el error se repite, crear una tarea para corregir sus datos.

El modelo anterior continúa activo si ya existía uno guardado.

### Falló toda la corrida

1. Leer el primer error de los logs.
2. Corregir el código o esperar si fue una caída temporal del collector.
3. Volver a desplegar solamente el modelo afectado:

```bash
modal deploy modelos/<archivo_del_modelo>.py
```

4. Ejecutar el entrenamiento inmediatamente:

```bash
modal run modelos/<archivo_del_modelo>.py::retrain_models
```

5. Esperar hasta ver:

```text
App completed
```

### Falló el endpoint

Primero confirmar que la URL no tenga `-dev`.

Correcta:

```text
https://matimorales01--lstm-trend-model-main.modal.run
```

Temporal y no apta para producción:

```text
https://matimorales01--lstm-trend-model-main-dev.modal.run
```

Después revisar los logs de esa aplicación y confirmar que exista un modelo
entrenado para el ticker solicitado.

## Los modelos guardados

Modal guarda los modelos activos en Volumes. Para comprobar que existen:

```bash
modal volume list
```

Deben aparecer:

```text
trend-model-artifacts
svm-model-artifacts
arima-model-artifacts
garch-model-artifacts
```

No hay que borrar ni modificar manualmente esos archivos. Son los modelos que
usan los endpoints de producción.

## Plantilla semanal

```md
### Control semanal de Modal - YYYY-MM-DD

| Modelo | Entrenó esta semana | Última corrida terminó | Endpoint responde | Promociones | Errores |
| --- | --- | --- | --- | --- | --- |
| LSTM | Sí / No | Sí / No | Sí / No | | |
| XGBoost | Sí / No | Sí / No | Sí / No | | |
| SVM | Sí / No | Sí / No | Sí / No | | |
| ARIMA | Sí / No | Sí / No | Sí / No | | |
| GARCH | Sí / No | Sí / No | Sí / No | | |

- Tickers con errores repetidos:
- Acciones realizadas:
- Pendientes:
```

## Cuando se cambia el código

Hacer `git push` no actualiza Modal.

Después de mergear un cambio hay que desplegar nuevamente el modelo afectado:

```bash
modal deploy modelos/<archivo_del_modelo>.py
```

Si no se quiere esperar al entrenamiento automático nocturno:

```bash
modal run modelos/<archivo_del_modelo>.py::retrain_models
```

En resumen:

```text
git push    → guarda el código en GitHub
modal deploy → actualiza la aplicación y el cron
modal run    → fuerza un entrenamiento ahora
```
