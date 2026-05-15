Uso con IA
==========

Jano incluye documentación e integraciones pensadas para uso asistido por IA.
Estos archivos ayudan a que agentes usen la librería correctamente, ejecuten
workflows estables y modifiquen el repositorio sin romper límites
arquitectónicos.

Las tres superficies son:

- notas de arquitectura para contexto de diseño,
- una guía para agentes y archivos de reglas por herramienta,
- y un servidor MCP opcional para ejecución local de tools.

Notas de arquitectura
---------------------

El mapa técnico de diseño vive en ``docs/architecture/``.

Incluye:

- ADRs para decisiones aceptadas,
- specs para comportamiento esperado,
- RFCs para propuestas de diseño abiertas.

Estos archivos sirven cuando un agente va a modificar Jano. Explican
restricciones como:

- el splitter sigue siendo agnóstico al modelo,
- la iteración manual de folds sigue siendo pública,
- los resultados del runner son data-first,
- los studies componen primitivas de menor nivel.

Guía para agentes y adaptadores
-------------------------------

La guía canónica para agentes es:

``docs/ai/jano-agent-guide.md``

Explica:

- cuándo usar ``TemporalBacktestSplitter``,
- cuándo usar ``WalkForwardPolicy`` y ``plan()``,
- cuándo usar ``WalkForwardRunner``,
- cómo consumir ``metric_trajectory()``, ``fold_summary()`` y ``report_data()``,
- y qué reglas de leakage temporal respetar.

Los adaptadores por herramienta apuntan a esa guía canónica:

- ``skills/jano/SKILL.md`` para uso estilo Codex skill,
- ``CLAUDE.md`` para guia de repositorio en Claude Code o Claude Desktop,
- ``.cursor/rules/jano.mdc`` para reglas de Cursor.

Servidor MCP
------------

El servidor MCP es código ejecutable, no solo documentación. Expone un conjunto
chico de tools locales para que clientes compatibles con MCP inspeccionen
datasets y corran workflows de Jano.

Usa MCP cuando un agente debe ejecutar operaciones sobre archivos locales:

- previsualizar un dataset local,
- construir un plan walk-forward,
- correr una simulación walk-forward,
- ejecutar un baseline simple sobre los mismos folds con
  ``run_walk_forward_baseline_model``,
- comparar policies de reentrenamiento con
  ``compare_retrain_policy_baselines``,
- buscar una ventana de train history con
  ``find_train_history_window_baseline``,
- monitorear decay con train fijo usando ``monitor_decay_baseline``.

Usa la guía para agentes o la skill cuando un agente necesita razonar sobre Jano
o escribir código Python con la librería.

La tool de baseline sirve para chequeos rápidos con ``model="mean"`` en
regresión numérica o ``model="majority_class"`` en clasificación. Para modelos
productivos conviene escribir Python con ``WalkForwardRunner`` y controlar
explícitamente features, estimadores y métricas.

Las tools de studies baseline sirven para triage operativo y exploración de
hipótesis temporales antes de escribir código de modelo propio.

En resumen:

- las notas de arquitectura explican por que y hacia donde va el proyecto,
- la guía para agentes explica cómo usar Jano correctamente,
- MCP da tools locales que los agentes pueden invocar.
