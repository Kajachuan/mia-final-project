# Transcripción polifónica de guitarra: primera versión

Esta implementación explora una limitación explícita de [Rashmi et al. (2026)](https://doi.org/10.62110/sciencein.jist.2026.v14.1582): después de separar la mezcla con Demucs, su sistema usa torchcrepe para obtener **un único F0 por stem**. Una guitarra o piano que toca acordes conserva varias notas en el mismo stem. La primera versión aquí **separa la mezcla con Demucs v4 (`htdemucs_6s`)** y luego predice un conjunto de notas MIDI simultáneas dentro del stem `guitar`.

## Datos y decisión

El paper evalúa con **Annotated-VocalSet** y **Vocadito**, datasets de voz solista. No son adecuados para medir notas simultáneas dentro de un instrumento. Esta versión usa [GuitarSet 1.1.0](https://doi.org/10.5281/zenodo.3371780) (Xi et al., 2018): 360 fragmentos de seis intérpretes, audio de micrófono mono y seis anotaciones `note_midi` por fragmento. La entrada es **solo** `audio_mono-mic.zip` (656.9 MB); las señales hexafónicas se usan en la creación de las etiquetas del dataset y **no** se entregan al modelo. Se necesita también `annotation.zip` (39.1 MB). Las 360 anotaciones oficiales fueron inspeccionadas: hay 60 fragmentos por intérprete y las notas redondeadas cubren MIDI 40–81.

La partición fija es por intérprete: `00`–`03` para entrenamiento (240 fragmentos), `04` para validación (60) y `05` para prueba (60). Evita la fuga de timbre/técnica entre fragmentos del mismo intérprete. Los mismos estilos y progresiones musicales aparecen en todas las particiones, por lo que **no** mide generalización a composiciones nuevas.

**Dataset complementario para evaluar el sistema completo:** [Slakh2100](https://www.slakh.com/) contiene mezclas, audio por instrumento y MIDI alineado. Permitirá comparar el MIDI de la guitarra de referencia con el resultado mezcla → separación → transcripción, y medir por separado el error del separador y el del estimador de notas. Es sintético y mucho más grande que GuitarSet; su [subconjunto pequeño oficial](https://www.slakh.com/) sirve para prototipos, pero no para una conclusión estadística. Todavía no se incluye Slakh en el entrenamiento ni se reportan resultados sobre él. GuitarSet sigue siendo la elección para entrenar y evaluar inicialmente las notas de guitarra acústica real.

## Método y aporte a contrastar

La estructura conserva **separación → dos ramas de análisis → fusión → MIDI**, pero cambia la unidad de predicción de un F0 por stem a una probabilidad por nota MIDI y por frame. No es una reproducción exacta de los pesos, umbrales o resultados numéricos del paper.

El umbral de confianza `0.15` de CREPE no se traslada a la CNN multi-etiqueta: sus probabilidades tienen otra calibración. El umbral final de notas se elige exclusivamente en validación; la regla de percentil 10 del paper sí se conserva en `paper_gate`.

| Componente del paper | Adaptación polifónica |
| --- | --- |
| Demucs de cuatro fuentes | `htdemucs_6s`: se toma el stem específico `guitar`. |
| CREPE: un F0 y confianza por frame | CNN multi-etiqueta sobre fundamental, armónicos 2–5 y subarmónicos: una probabilidad para **cada** pitch MIDI 40–82, más onsets. |
| Mel de 256 bandas + croma → máscara de voicing global | Se calculan Mel de 256 bandas, energía de la clase cromática y energía global. La máscara fija al percentil 10 se conserva como comparación; la propuesta aprende una compuerta **por pitch y frame** a partir de esos rasgos. |
| Fusión y MIDI por stem | Probabilidad de nota × compuerta → umbral elegido en validación → eventos MIDI simultáneos en la pista de guitarra. |

| Variante evaluada | Diferencia aislada |
| --- | --- |
| `baseline` | Estimador multi-pitch sin compuerta. |
| `paper_gate` | **El mismo checkpoint que `baseline`**, con máscara de voicing global al percentil 10 inspirada en el paper. |
| `gated` | La misma rama estimadora multi-pitch, con rama Mel/croma y compuerta aprendida por pitch y frame. |

Antes de esas variantes, el flujo de inferencia ejecuta el modelo preentrenado `htdemucs_6s` y guarda seis stems (`vocals`, `drums`, `bass`, `other`, `guitar`, `piano`). **Solo `guitar` pasa al predictor multi-etiqueta.** El [proyecto oficial de Demucs](https://github.com/facebookresearch/demucs#separation) describe esta variante como experimental y advierte que los stems pueden tener filtraciones y artefactos. Demucs no se entrena en GuitarSet ni en Kaggle: se utilizan sus pesos públicos preentrenados. El entrenamiento en Kaggle ajusta solamente el predictor de notas sobre guitarras limpias de GuitarSet. Esto crea un posible **cambio de dominio** entre el audio de entrenamiento y el stem separado; falta medirlo con mezclas anotadas.

Las tres variantes pueden activar varias notas a la vez. La hipótesis de investigación es que **una compuerta Mel/croma aprendida por nota** reduce falsos positivos sin eliminar notas reales de acordes, frente al estimador solo y frente a la máscara global fija. Es una hipótesis, no un resultado medido. Como siguiente comparación de tesis convendría incluir un baseline fuerte de transcripción de guitarra y una evaluación cruzada en otro dataset. Esta primera versión establece el protocolo y un primer experimento reproducible.

La pérdida combina BCE de activación de notas y BCE de onsets. La selección de época y umbral usa solo validación. Se reportan precisión, recall y F1 micro por frame, por frame con al menos dos notas, por nota con inicio y final y por nota con inicio solamente. El emparejamiento estándar de [mir_eval](https://mir-eval.readthedocs.io/stable/api/transcription.html) exige pitch a 50 cents e inicio a ±50 ms; con final, acepta ±máx(50 ms, 20 % de duración). Los ejemplos se reconstruyen como MIDI con acordes permitidos. La versión inicial transcribe únicamente guitarra y no modela expresividad o velocidad.

## Ejecutar en Kaggle

El notebook [train_guitarset.ipynb](kaggle/train_guitarset.ipynb) usa [alonhaviv/guitarset](https://www.kaggle.com/datasets/alonhaviv/guitarset) como **Add Input**. Lee directamente `/kaggle/input/guitarset/annotation` y `/kaggle/input/guitarset/audio_mono-mic`; no descarga los ZIP ni copia los WAV. Solo escribe características y checkpoints en `/kaggle/working`. Entrena `baseline` y `gated`, deriva `paper_gate` del checkpoint base, compara las tres métricas y hace una prueba funcional de Demucs. Activar GPU e Internet para clonar este repositorio y obtener los pesos de Demucs la primera vez; primero hay que publicar estos cambios en el remoto. La descarga oficial de Zenodo queda como alternativa para ejecutar el proyecto fuera de Kaggle.

Cada variante guarda `<variante>_last.pt` **cada 25 lotes y al finalizar cada época**, con modelo, optimizador, estado aleatorio, época, próximo lote, historial y mejor modelo. La escritura es atómica. `<variante>.pt` conserva solo el mejor modelo para inferencia. El parámetro `--max-runtime-minutes` detiene la corrida en un límite de lote y deja `<variante>_status.json` en estado `paused`; después se puede reanudar con `--resume`. Si Kaggle corta el proceso antes de guardar, se repetirán los lotes posteriores al último checkpoint disponible. Para conservar los archivos entre sesiones, usar **Save Version**, adjuntar la salida como **Notebook Output Files** en la siguiente sesión y completar `PREVIOUS_OUTPUT` en el notebook; [Kaggle documenta este mecanismo de reutilización](https://www.kaggle.com/docs/notebooks). La caché de características de la sesión anterior también puede reutilizarse desde esa entrada, sin recalcularla.

Los comandos equivalentes dentro de Kaggle, desde la raíz del proyecto, son:

```bash
pip install -r requirements.txt
python -m polytranscribe.prepare --source-root /kaggle/input/guitarset --root /kaggle/working/guitarset
python -m polytranscribe.train --cache /kaggle/working/guitarset/cache --output /kaggle/working/runs --variant baseline --epochs 12 --checkpoint-every 25 --max-runtime-minutes 180
python -m polytranscribe.train --cache /kaggle/working/guitarset/cache --output /kaggle/working/runs --variant gated --epochs 12 --checkpoint-every 25 --max-runtime-minutes 180
python -m polytranscribe.pipeline --audio path/to/mezcla.wav --checkpoint /kaggle/working/runs/gated.pt --output-dir /kaggle/working/runs/demo
```

Para continuar la variante base en otra sesión, por ejemplo, usar `python -m polytranscribe.train --cache /kaggle/input/<salida-anterior>/guitarset/cache --output /kaggle/working/runs --variant baseline --epochs 24 --resume /kaggle/input/<salida-anterior>/runs/baseline_last.pt`. `--epochs 24` indica el objetivo total acumulado, no 24 épocas adicionales. Mantener el tamaño de lote y los parámetros de características al reanudar.

El entrenamiento de `baseline` produce además `runs/paper_gate.pt` y `runs/paper_gate.json`, sin entrenar un tercer estimador. La preparación usa la versión 2 de las características; si hay una caché antigua, volver a ejecutar `prepare` antes de entrenar.

El último comando guarda los seis stems en `runs/demo/stems/htdemucs_6s/mezcla/`, transcribe `guitar` en `runs/demo/guitar.mid` y documenta el flujo en `runs/demo/pipeline.json`. Para probar directamente una guitarra aislada sin Demucs, existe `python -m polytranscribe.infer --audio guitarra.wav --checkpoint runs/gated.pt --midi guitarra.mid`.

La preparación en Kaggle no requiere una descarga de GuitarSet dentro del notebook ni usar las grabaciones hexafónicas como entrada del modelo. Una corrida completa con GPU de Kaggle sigue pendiente; el repositorio incluye una prueba que interrumpe y reanuda el entrenamiento desde un lote guardado, pero no resultados de precisión inventados. Para medir el aporte final habrá que incorporar mezclas con referencia MIDI de guitarra, por ejemplo Slakh2100, y reportar ambos escenarios: stem ideal y stem separado por Demucs.

## Evaluación sobre mezclas

El evaluador [eval_slakh.py](polytranscribe/eval_slakh.py) admite [BabySlakh](https://zenodo.org/records/4603870) para una prueba pequeña (20 temas sintetizados, WAV a 16 kHz) y Slakh2100-redux para una evaluación mayor. A partir de `metadata.yaml` selecciona los stems `inst_class: Guitar` y sus archivos MIDI individuales; usa `mix.wav` o `mix.flac` como mezcla. Reporta las métricas de notas sobre la **guitarra ideal** (suma de los stems de guitarra) y sobre la **guitarra estimada** por `htdemucs_6s`, además de la caída de F1. Incluye precisión, recall y F1 micro y macro por pista para notas con inicio/final, notas con inicio, frames y frames polifónicos. Los MIDI de referencia fuera del rango 40–82 se cuentan como errores en F1 por nota y se informan por separado; las métricas por frame se restringen al rango del modelo. BabySlakh sirve para verificar el protocolo, no como estimación final de generalización.

```bash
python -m polytranscribe.eval_slakh --data-root /kaggle/input/babyslakh/babyslakh_16k --checkpoint /kaggle/working/runs/gated.pt --output /kaggle/working/runs/slakh_eval --split any --max-tracks 3 --seconds 30
```

El evaluador escribe `summary.json` y un `metrics.json` por tema. Para Slakh2100-redux, usar `--split test` (valor predeterminado), `--max-tracks 0` y `--seconds 0` para los temas completos. La opción `--with-si-sdr` agrega SI-SDR entre la guitarra aislada y el stem separado. Es un diagnóstico opcional: Demucs es preentrenado y el resultado principal del trabajo es el F1 **mezcla → MIDI**. Aún no se ejecutó esta evaluación sobre BabySlakh o Slakh real en Kaggle; las pruebas locales usan una mezcla sintética mínima.

## Referencias

- Rashmi M. R. et al. (2026), *A modular approach to polyphonic music transcription using source separation and harmonic-gated pitch tracking*, Journal of Integrated Science and Technology 14(5), 1582. DOI: 10.62110/sciencein.jist.2026.v14.1582.
- Xi, Q., Bittner, R. M., Pauwels, J., Ye, X. y Bello, J. P. (2018), *GuitarSet: A Dataset for Guitar Transcription*, ISMIR. Dataset: https://doi.org/10.5281/zenodo.3371780.
