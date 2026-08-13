# Cómo saber si los modelos de producción predicen bien

Hay dos herramientas diferentes:

- **Backtest:** sirve para probar un cambio antes de llevarlo a producción.
- **Paper trading:** sirve para saber cómo predijo el modelo real que está en
  Modal.

Para el seguimiento semanal de producción hay que mirar **paper trading**.

## Qué hace el paper trading

Todos los días guarda lo que predijo cada modelo:

```text
GGAL + LSTM Modal + 5 días → predijo baja
```

Cuando pasan las cinco ruedas, busca qué ocurrió realmente:

```text
GGAL efectivamente bajó → acierto
GGAL subió → error
```

Con todas las predicciones ya comprobadas calcula un resumen. El archivo está
en el repo `api-ml`:

```text
models/paper_trading_ledger.json
```

GitHub Actions lo actualiza automáticamente de lunes a viernes.

## Qué mirar cada semana

Abrir `api-ml/models/paper_trading_ledger.json` y buscar `summary`.

Dentro de `summary` deben aparecer los modelos productivos:

```json
{
  "summary": {
    "lstm-modal": {
      "overall": {}
    },
    "xgboost-modal": {
      "overall": {}
    }
  }
}
```

Si no aparecen `lstm-modal` y `xgboost-modal`, no estamos midiendo producción:
estamos mirando solamente los modelos viejos/locales de `api-ml`.

## Cómo leer el resultado

Ejemplo:

```json
{
  "n_predictions": 100,
  "directional_accuracy": 0.58,
  "signal_hit_rate": 0.61,
  "neutral_rate": 0.20,
  "mae_logret": 0.03,
  "rmse_logret": 0.05,
  "avg_strategy_logret": 0.004,
  "avg_buy_hold_logret": 0.001
}
```

Interpretación simple:

- `n_predictions`: cuántas predicciones ya se pudieron comprobar. Cuantas más,
  más confiable es el resultado.
- `directional_accuracy`: porcentaje de veces que acertó si subía o bajaba.
  `0.58` significa 58%.
- `signal_hit_rate`: aciertos cuando el modelo realmente dijo alza o baja,
  ignorando los neutrales. Más alto es mejor.
- `neutral_rate`: cuántas veces evitó decidir. Si es muy alto, el modelo puede
  parecer bueno porque casi nunca toma posición.
- `mae_logret` y `rmse_logret`: tamaño del error. Más bajo es mejor.
- `avg_strategy_logret`: resultado teórico siguiendo sus señales.
- `avg_buy_hold_logret`: resultado de simplemente comprar y mantener.

## Decisión rápida

El modelo viene razonablemente bien si:

- tiene al menos 50 predicciones comprobadas;
- `directional_accuracy` está por encima de `0.50`;
- el resultado no depende de un solo ticker;
- `neutral_rate` no es exageradamente alto;
- `avg_strategy_logret` supera a `avg_buy_hold_logret`.

El modelo necesita revisión si:

- la accuracy queda debajo de 50% durante varias semanas;
- aumentan MAE o RMSE;
- deja de generar predicciones nuevas;
- sólo funciona bien para uno o dos tickers;
- el resultado de la estrategia queda debajo de comprar y mantener.

No conviene decidir por una sola semana ni con muy pocas predicciones. Lo
importante es mirar la tendencia durante varias semanas.

## Checklist semanal

Cada lunes:

1. Abrir el último commit automático de
   `api-ml/models/paper_trading_ledger.json`.
2. Confirmar que `updated_at` sea reciente.
3. Confirmar que aparezcan `lstm-modal` y `xgboost-modal` en `summary`.
4. Anotar cuántas predicciones comprobadas tiene cada uno.
5. Anotar accuracy, signal hit rate, neutral rate, MAE y resultado de estrategia.
6. Comparar contra la semana anterior.
7. Revisar `per_ticker` para comprobar que la mejora sea general.

Plantilla para la discussion:

```md
### Seguimiento semanal de predicciones - YYYY-MM-DD

| Modelo de producción | Predicciones comprobadas | Accuracy | Acierto de señales | Neutrales | MAE | Estrategia | Buy & hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM Modal | | | | | | | |
| XGBoost Modal | | | | | | | |

- ¿Mejoró o empeoró respecto de la semana anterior?:
- Tickers con mejor resultado:
- Tickers con peor resultado:
- ¿Hay suficientes datos para concluir?:
- Acción: mantener / investigar / probar un cambio
```

## Cómo activar la medición de producción

En GitHub, dentro del repo `api-ml`, configurar estos Actions secrets:

```text
MODAL_LSTM_URL=https://matimorales01--lstm-trend-model-main.modal.run
MODAL_XGBOOST_URL=https://matimorales01--xgboost-trend-model-main.modal.run
```

Después ejecutar manualmente el workflow `Paper trading diario` una vez.

La primera ejecución guarda predicciones como `pending`. No puede saber si
acertaron hasta que pase su horizonte. Después de aproximadamente cinco ruedas,
las siguientes corridas empiezan a moverlas a `resolved` y aparecen las
métricas de `lstm-modal` y `xgboost-modal`.

## Para qué queda el backtest

El backtest se usa cuando queremos probar nuevas features o hiperparámetros:

```bash
cd /Users/matimorales01/Desktop/tp_profesional/api-ml
python -m src.backtest --tickers GGAL YPFD ALUA --models lstm xgboost
```

Permite comparar una versión anterior contra una nueva sobre el mismo pasado.
No mide directamente los endpoints reales de Modal porque esos endpoints no
aceptan un histórico cortado en una fecha vieja.

En resumen:

```text
Backtest      → ¿vale la pena probar este cambio?
Paper trading → ¿el modelo real de producción está acertando?
```

## Tests

```bash
cd /Users/matimorales01/Desktop/tp_profesional/api-ml
python -m pytest -v
```

Los tests indican si el código funciona. No indican si el modelo predice bien.
