Splits Aleatorios vs Validación Temporal
========================================

``sklearn.model_selection.train_test_split`` es útil cuando las observaciones
pueden tratarse como aproximadamente independientes e idénticamente distribuidas.
Esa no es la pregunta que Jano busca responder.

Cuando los datos están correlacionados en el tiempo, la pregunta suele ser más
operativa:

   ¿Cómo se habría comportado el modelo si solo hubiese visto el pasado y luego
   tuviera que predecir el futuro?

Un split aleatorio puede ocultar esa pregunta porque mezcla fechas entre train y
test.

El primer snippet asume que scikit-learn está instalado solo para ilustrar el
baseline común. Jano no requiere scikit-learn.

El Problema
-----------

Imaginá un dataset diario donde la distribución del target cambia cerca del final
del período:

.. code-block:: python

   import pandas as pd
   from sklearn.model_selection import train_test_split

   frame = pd.DataFrame(
       {
           "timestamp": pd.date_range("2025-01-01", periods=120, freq="D"),
           "feature": range(120),
           "target": [0] * 80 + [1] * 40,
       }
   )

   train_random, test_random = train_test_split(
       frame,
       test_size=0.2,
       shuffle=True,
       random_state=7,
   )

   temporal_leakage = (
       train_random["timestamp"].max() > test_random["timestamp"].min()
   )

   print(temporal_leakage)
   # True

El problema no es que scikit-learn esté mal. ``train_test_split`` hace lo que
debe hacer: muestreo aleatorio. El problema es que el muestreo aleatorio es la
abstracción equivocada para una validación temporal parecida a producción.

En este setup, train puede contener observaciones de fechas posteriores a algunas
observaciones de test. Si el target cambia en el tiempo, la evaluación puede
volverse demasiado optimista porque el modelo ya vio parte del régimen futuro.

La Versión Con Jano
-------------------

Con Jano, el split no se define como un porcentaje aleatorio de filas. Se define
como una política temporal:

.. code-block:: python

   import pandas as pd

   from jano import TemporalPartitionSpec, WalkForwardPolicy

   frame = pd.DataFrame(
       {
           "timestamp": pd.date_range("2025-01-01", periods=120, freq="D"),
           "feature": range(120),
           "target": [0] * 80 + [1] * 40,
       }
   )

   policy = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="60D",
           test_size="14D",
           gap_before_test="1D",
       ),
       step="14D",
       strategy="rolling",
   )

   plan = policy.plan(frame, title="Validación temporal productiva")

   print(
       plan.to_frame()[
           [
               "iteration",
               "train_start",
               "train_end",
               "train_rows",
               "test_start",
               "test_end",
               "test_rows",
           ]
       ].head()
   )

El plan vuelve explícito el contrato temporal antes de entrenar cualquier modelo:

.. code-block:: text

    iteration train_start  train_end  train_rows test_start   test_end  test_rows
            0  2025-01-01 2025-03-02          60 2025-03-03 2025-03-17         14
            1  2025-01-15 2025-03-16          60 2025-03-17 2025-03-31         14
            2  2025-01-29 2025-03-30          60 2025-03-31 2025-04-14         14
            3  2025-02-12 2025-04-13          60 2025-04-14 2025-04-28         14

Qué Cambia
----------

La diferencia está en el contrato de evaluación:

- ``train_test_split`` responde: ¿este modelo generaliza a una muestra aleatoria
  del mismo período mezclado?
- Jano responde: ¿cómo se comportaría este modelo a medida que avanza el tiempo
  bajo una política concreta de entrenamiento y evaluación?

Eso te da:

- ventanas ordenadas de train y test
- duración explícita de train/test
- gaps explícitos para modelar latencia de datos o labels
- folds repetidos en lugar de una única estimación estática
- un objeto ``plan()`` que puede inspeccionarse, filtrarse y auditarse antes de
  hacer slicing del dataset

Ahí es donde entra Jano: no como reemplazo de scikit-learn, sino como la capa de
validación temporal previa al entrenamiento del modelo.
