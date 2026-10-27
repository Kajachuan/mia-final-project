# Transcripción polifónica en cinco pistas

El sistema sigue la estructura modular de [Rashmi et al. (2026)](https://doi.org/10.62110/sciencein.jist.2026.v14.1582): **separación → estimación de notas → filtro armónico → MIDI**. El paper estima un solo F0 por stem; aquí las notas con altura se predicen de forma multi-etiqueta para permitir acordes dentro de guitarra y piano. `htdemucs_6s` separa seis fuentes; se transcriben `vocals`, `drums`, `bass`, `guitar` y `piano`. El audio `other` se conserva sin inventarle una pista de notas.

## Modelos y datos

| Stem | Datos para la primera versión | Salida |
| --- | --- | --- |
| `guitar`, `bass`, `piano`, `drums` | [Slakh2100-redux](https://www.kaggle.com/datasets/alonhaviv/slakh2100): mezcla y MIDI de cada fuente. Demucs genera los cuatro stems estimados a partir de la mezcla. | Notas MIDI simultáneas o nueve clases de golpes de batería. |
| `vocals` | [Vocadito v3](https://zenodo.org/records/5578807): 40 voces con notas manuales. Se combinan con acompañamiento de Slakh del mismo split y se separan con Demucs. | Notas MIDI de voz. |
| Benchmark auxiliar | [GuitarSet 1.1.0](https://doi.org/10.5281/zenodo.3371780): guitarra acústica real con anotaciones por cuerda. | Evaluación o entrenamiento aislado de control. |

Slakh no contiene voces cantadas. El entrenamiento principal usa **dos fuentes**: Slakh para los cuatro instrumentos y Vocadito para la voz; GuitarSet queda como control externo. En producción hay **un transcriptor compartido** para voz, bajo, guitarra y piano, condicionado por el stem, y un segundo modelo para batería. Los cuatro stems con altura tienen rangos MIDI propios: voz 36–88, bajo 28–67, guitarra 40–82 y piano 21–108. La batería usa una arquitectura distinta para golpes. Demucs es un único separador preentrenado y **congelado** que se ejecuta una vez por mezcla. El dataset de Kaggle [alonhaviv/slakh2100](https://www.kaggle.com/datasets/alonhaviv/slakh2100) contiene `slakh2100_flac_redux` (aprox. 108 GB); `MAX_SLAKH_TRACKS` y `MIX_SECONDS` limitan cuánto se prepara dentro de Kaggle.

**Alcance experimental:** el notebook selecciona 100 temas de Slakh2100-redux y un fragmento de 20 segundos por tema por defecto. Para resultados finales habrá que ampliar la cobertura. Vocadito tiene solo 40 fragmentos y sus mezclas con Slakh son **sintéticas**; no demuestran rendimiento en voces de canciones reales. Conviene ampliar los datos de voz, por ejemplo con [Annotated-VocalSet](https://zenodo.org/records/7061507). Los modelos aún no se entrenaron en Kaggle y no se reportan precisiones inventadas.

## Método

El flujo de entrenamiento principal es **mezcla → Demucs congelado → stem estimado → características → transcriptor → pérdida contra notas MIDI**. [prepare_mixtures.py](polytranscribe/prepare_mixtures.py) ejecuta Demucs una vez por fragmento y guarda las características del stem estimado; esto evita repetir la separación en cada época y reproduce los ajustes de inferencia. Slakh aporta mezcla y MIDI alineado. Para voz, una grabación de Vocadito se suma a un acompañamiento Slakh de la misma partición antes de separarla. Los archivos completos de mezcla y stems no se duplican en `/kaggle/working`. Las notas se recortan y desplazan al mismo intervalo temporal que el audio.

El transcriptor compartido conserva la idea de dos ramas del paper. La **rama A** recibe mapas de fundamental, subarmónicos y armónicos 2–5. Bloques residuales con distintas dilataciones temporales capturan ataques y sostenidos; una atención por pitch seguida de atención temporal relaciona notas simultáneas y su evolución sin aplicar atención completa sobre todos los puntos del espectrograma. La atención usa la posición MIDI absoluta para alinear los rangos de los cuatro stems. Una condición aprendida de instrumento modula las características internas. Se predicen inicios y activaciones por pitch/frame; la activación también recibe la probabilidad de inicio, inspirada en [Onsets and Frames](https://arxiv.org/abs/1710.11153). La **rama B** aprende una compuerta local con energía Mel, croma y evidencia global de actividad, y modula la activación. La inferencia por ventanas limita la memoria de atención.

La hipótesis de la tesis es que la **compuerta armónica adaptativa** mejora la precisión de notas y reduce falsas activaciones en mezcla. `shared_plain` y `shared_gated` usan el mismo entrenamiento balanceado y difieren solo en la compuerta. El entrenamiento selecciona el mejor checkpoint por el promedio no ponderado del F1 de notas de los cuatro stems en validación y calibra un umbral por stem. Las CNN anteriores (`baseline`, `gated`, `paper_gate`) y los cuatro modelos separados (`advanced_plain`, `advanced_gated`) quedan disponibles como controles experimentales. La arquitectura avanzada sigue siendo un modelo propio y entrenable; [Perceiver TF](https://arxiv.org/abs/2306.10785) y [YourMT3+](https://arxiv.org/abs/2407.04822) sirven como referencias externas, no como pesos del transcriptor. La batería predice golpes multi-etiqueta sin compuerta de altura. Demucs se ejecuta **una vez por mezcla** y después se generan `five_stems.mid` y un MIDI por stem.

El experimento anterior de guitarra queda disponible en [train_guitarset.ipynb](kaggle/train_guitarset.ipynb): `baseline` no filtra, `paper_gate` aplica al **mismo checkpoint base** la máscara fija del paper y `gated` aprende una compuerta. El flujo principal [train_five_stems.ipynb](kaggle/train_five_stems.ipynb) entrena el transcriptor compartido y la batería. Comparar `shared_plain` con `shared_gated` evalúa la compuerta; comparar un transcriptor compartido con cuatro independientes evalúa el efecto de compartir pesos, pero requiere ejecutar esos controles separados y no forma parte del flujo económico por defecto.

## Entrenar en Kaggle

1. Adjuntar [Slakh2100](https://www.kaggle.com/datasets/alonhaviv/slakh2100) como **Add Input** y abrir [train_five_stems.ipynb](kaggle/train_five_stems.ipynb). El notebook busca `/kaggle/input/datasets/alonhaviv/slakh2100/slakh2100_flac_redux`, con alternativa para el montaje corto. GuitarSet es opcional para el benchmark externo.
2. `AUTO_DOWNLOAD_VOCADITO = True` descarga Vocadito v3 (58 MB) desde Zenodo, verifica MD5 y lo extrae. Si ya está adjuntado, indicar `VOCADITO_ROOT`. Ajustar `MAX_SLAKH_TRACKS` y `MIX_SECONDS` según el tiempo disponible; `0` en el primer parámetro procesa todos los temas.
3. Ejecutar el notebook con GPU e Internet para obtener los pesos de Demucs. La preparación escribe características en `/kaggle/working/mixed_cache/`. Si se interrumpe, adjuntar la salida anterior y fijar `PREVIOUS_OUTPUT` para reutilizar NPZ completos. El entrenamiento guarda `shared_gated.pt` y `shared_plain.pt` en `/kaggle/working/runs_five/` y batería en `/kaggle/working/runs_five/drums/`.

Los comandos principales, dentro de Kaggle y desde la raíz del repositorio, son:

```bash
python -m polytranscribe.download_aux --output /kaggle/working/aux --dataset vocadito
python -m polytranscribe.prepare_mixtures --slakh-root /kaggle/input/datasets/alonhaviv/slakh2100/slakh2100_flac_redux --vocadito-root /kaggle/working/aux/vocadito --output /kaggle/working/mixed_cache --max-slakh-tracks 100 --seconds 20
python -m polytranscribe.train_shared --guitar-cache /kaggle/working/mixed_cache/guitar --aux-cache /kaggle/working/mixed_cache --output /kaggle/working/runs_five --variant shared_gated --steps-per-stem 100
python -m polytranscribe.train --cache /kaggle/working/mixed_cache/drums --output /kaggle/working/runs_five/drums --variant advanced
python -m polytranscribe.pipeline --audio /ruta/mezcla.wav --models-root /kaggle/working/runs_five --output-dir /kaggle/working/demo
```

El notebook ejecuta tres entrenamientos: `shared_gated`, batería `advanced` y la ablación `shared_plain`. El entrenamiento compartido alterna lotes de los cuatro stems con altura para evitar que un instrumento domine; `STEPS_PER_STEM=100` limita los lotes de cada stem por época. Es un muestreo con reemplazo. El modelo compartido ocupa un solo checkpoint y se aplica a cada stem separado durante inferencia; las cuatro señales todavía requieren cuatro pasadas por la red. Compartir pesos reduce almacenamiento y permite transferencia entre instrumentos, pero **no divide por cuatro el costo de procesar todos los audios**. La opción `--models-root` acepta el checkpoint compartido y el de batería; los checkpoints separados anteriores siguen admitidos como alternativa.

Cada entrenamiento guarda `<variante>_last.pt` **cada 25 lotes y al cerrar cada época**, con modelo, optimizador, estado aleatorio, época, siguiente lote o paso y mejor modelo. `<variante>.pt` es el mejor checkpoint para inferencia. El notebook limita la duración de la sesión y permite reanudar desde una salida anterior fijando `PREVIOUS_OUTPUT`. Usar **Save Version** y adjuntar esa salida como **Notebook Output Files** en la próxima sesión; [Kaggle documenta su reutilización](https://www.kaggle.com/docs/notebooks). El checkpoint compartido vincula la reanudación con las cuatro cachés, las particiones y los parámetros de muestreo. Los checkpoints entrenados con stems ideales no se reanudan sobre la nueva caché de Demucs.

## Evaluación y límites

El entrenamiento informa precisión, recall y F1 por frame, en frames con dos o más notas, por nota con inicio y final, y por nota con inicio solamente, **por stem estimado**. Se usa el emparejamiento de [mir_eval](https://mir-eval.readthedocs.io/stable/api/transcription.html): inicio a ±50 ms, pitch a 50 cents y final a ±máx(50 ms, 20 % de duración). Para batería, el F1 de notas evalúa **clase de golpe e inicio**. Los umbrales se eligen por stem y la mejor época por promedio de F1 de notas en validación; el test no se usa para ajustar. El test instrumental evalúa mezcla Slakh → Demucs → MIDI en los fragmentos elegidos. La voz solo está medida en mezclas sintéticas, de modo que falta una prueba externa con voz real en mezcla.

Para una comparación externa reproducible, generar primero MIDI con [Basic Pitch](https://github.com/spotify/basic-pitch) u otro modelo sobre los 60 audios del intérprete 05 de GuitarSet, un `{stem}.mid` por fragmento. El evaluador [benchmark_external.py](polytranscribe/benchmark_external.py) lee esos MIDI y calcula las mismas métricas y tolerancias. El notebook tiene una celda opcional `EXTERNAL_MIDI_DIR`; **no ejecuta ni entrena el modelo externo automáticamente**. Esa métrica de GuitarSet no se compara directamente con el test principal de mezclas Slakh: habría que evaluar ambos modelos sobre las mismas entradas. Todavía no hay evidencia de superar el estado del arte.

El evaluador [eval_slakh.py](polytranscribe/eval_slakh.py) compara la transcripción de guitarra aislada con la obtenida tras Demucs sobre una mezcla Slakh y permite agregar SI-SDR como diagnóstico opcional. Slakh no aporta voz cantada, por lo que todavía no permite medir un F1 conjunto de las **cinco** pistas contra referencia en una sola mezcla. La variante de seis fuentes de Demucs también es experimental; sus autores advierten especialmente sobre la calidad del stem de piano. Esta limitación y la diferencia entre audio real y sintetizado deben informarse junto a cualquier resultado.

## Referencias

- Rashmi M. R. et al. (2026), *A modular approach to polyphonic music transcription using source separation and harmonic-gated pitch tracking*, Journal of Integrated Science and Technology 14(5), 1582. DOI: 10.62110/sciencein.jist.2026.v14.1582.
- Xi, Q. et al. (2018), *GuitarSet: A Dataset for Guitar Transcription*, ISMIR. Dataset: https://doi.org/10.5281/zenodo.3371780.
- Manilow, E. et al. (2019), *Cutting Music Source Separation Some Slakh*, WASPAA. Dataset: https://zenodo.org/records/4599666.
- Bittner, R. et al. (2021), *Vocadito: A dataset of solo vocals with f0, note, and lyric annotations*. Dataset: https://zenodo.org/records/5578807.
