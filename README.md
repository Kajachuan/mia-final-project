# Transcripción polifónica en cinco pistas

El sistema sigue la estructura modular de [Rashmi et al. (2026)](https://doi.org/10.62110/sciencein.jist.2026.v14.1582): **separación → estimación de notas → filtro armónico → MIDI**. El paper estima un solo F0 por stem; aquí las notas con altura se predicen de forma multi-etiqueta para permitir acordes dentro de guitarra y piano. `htdemucs_6s` separa seis fuentes; se transcriben `vocals`, `drums`, `bass`, `guitar` y `piano`. El audio `other` se conserva sin inventarle una pista de notas.

## Modelos y datos

| Stem | Datos para la primera versión | Salida |
| --- | --- | --- |
| `guitar` | [GuitarSet 1.1.0](https://doi.org/10.5281/zenodo.3371780): 360 fragmentos de guitarra acústica con notas por cuerda. En Kaggle se adjunta [alonhaviv/guitarset](https://www.kaggle.com/datasets/alonhaviv/guitarset). | Notas MIDI simultáneas. |
| `vocals` | [Vocadito v3](https://zenodo.org/records/5578807): 40 fragmentos de canto solista con notas manuales. | Notas MIDI de voz. |
| `bass`, `piano`, `drums` | [Slakh2100-redux](https://www.kaggle.com/datasets/alonhaviv/slakh2100): mezclas, stems y MIDI alineado, adjuntado en Kaggle como dataset. | Notas MIDI de bajo y piano; nueve clases de golpes de batería. |

Slakh no contiene voces cantadas, y GuitarSet solo anota guitarra. Por eso la primera versión usa **tres fuentes de datos**. Hay **cinco transcriptores con pesos independientes**, uno por stem; los cuatro modelos con altura comparten la misma arquitectura, pero tienen checkpoints y rangos MIDI propios: voz 36–88, bajo 28–67, guitarra 40–82 y piano 21–108. La batería usa una arquitectura distinta para golpes, con bloques residuales y atención temporal sobre Mel y nueve clases General MIDI: bombo, caja, hi-hat cerrado y abierto, tom, crash, ride, clap/rim y otra percusión. Demucs es un único separador preentrenado que se ejecuta una vez por mezcla. El dataset de Kaggle [alonhaviv/slakh2100](https://www.kaggle.com/datasets/alonhaviv/slakh2100) contiene `slakh2100_flac_redux` (aprox. 108 GB); `MAX_SLAKH_TRACKS` limita cuántos temas se procesan dentro de Kaggle.

**Alcance experimental:** el notebook selecciona 100 temas de Slakh2100-redux por defecto para limitar tiempo y espacio; para los resultados finales habrá que ampliar esa muestra. Vocadito tiene solo 40 fragmentos, por lo que también conviene ampliar los datos de voz, por ejemplo con [Annotated-VocalSet](https://zenodo.org/records/7061507). Los modelos aún no se entrenaron en Kaggle y no se reportan precisiones inventadas.

## Método

El transcriptor avanzado de cada stem con altura conserva la idea de dos ramas del paper. La **rama A** recibe mapas de fundamental, subarmónicos y armónicos 2–5. Bloques residuales con distintas dilataciones temporales capturan ataques y sostenidos; una atención por pitch seguida de atención temporal relaciona notas simultáneas y su evolución sin aplicar atención completa sobre todos los puntos del espectrograma. Predice inicios y activaciones por pitch/frame; la predicción de activación también recibe la probabilidad de inicio, inspirada en [Onsets and Frames](https://arxiv.org/abs/1710.11153). La **rama B** aprende una compuerta local con energía Mel, croma y evidencia global de actividad, y modula la activación. La inferencia por ventanas limita la memoria de atención.

La hipótesis de la tesis es que la **compuerta armónica adaptativa** mejora la precisión de notas y reduce falsas activaciones en mezcla, especialmente cuando hay varias notas simultáneas. `advanced_plain` y `advanced_gated` comparten toda la rama A y difieren solo en la compuerta. `advanced_paper_gate` aplica la compuerta fija inspirada en el paper al checkpoint de `advanced_plain`. Las CNN anteriores (`baseline`, `gated`, `paper_gate`) quedan como controles de capacidad reducida. La arquitectura avanzada sigue siendo un modelo propio y entrenable; [Perceiver TF](https://arxiv.org/abs/2306.10785) y [YourMT3+](https://arxiv.org/abs/2407.04822) sirven como referencias externas de atención y transcripción multitrack, no como pesos del transcriptor. La batería predice golpes multi-etiqueta sin compuerta de altura. Demucs se ejecuta **una vez por mezcla** y después se generan `five_stems.mid` y un MIDI por stem.

El experimento anterior de guitarra queda disponible en [train_guitarset.ipynb](kaggle/train_guitarset.ipynb): `baseline` no filtra, `paper_gate` aplica al **mismo checkpoint base** la máscara global fija al percentil 10 del paper y `gated` aprende una compuerta por pitch y frame. El flujo principal [train_five_stems.ipynb](kaggle/train_five_stems.ipynb) compara `advanced_plain`, `advanced_paper_gate` y `advanced_gated` en guitarra y entrena la variante avanzada para los demás stems. Comparar la arquitectura avanzada con la CNN simple responde otra pregunta experimental: cuánto aporta la capacidad del estimador respecto de la compuerta.

## Entrenar en Kaggle

1. Adjuntar [GuitarSet](https://www.kaggle.com/datasets/alonhaviv/guitarset) y [Slakh2100](https://www.kaggle.com/datasets/alonhaviv/slakh2100) como **Add Input** y abrir [train_five_stems.ipynb](kaggle/train_five_stems.ipynb). El notebook busca `/kaggle/input/datasets/alonhaviv/guitarset` y `/kaggle/input/datasets/alonhaviv/slakh2100/slakh2100_flac_redux`, con alternativas para el montaje corto de Kaggle. No descarga ninguno de estos dos datasets.
2. `AUTO_DOWNLOAD_VOCADITO = True` descarga Vocadito v3 (58 MB) desde Zenodo, verifica MD5 y lo extrae. Si ya está adjuntado, indicar `VOCADITO_ROOT`. Ajustar `MAX_SLAKH_TRACKS` para ampliar el subconjunto de Slakh; `0` procesa todos los temas. Al cambiar ese límite o la fuente de Slakh se reconstruye la caché instrumental.
3. Ejecutar el notebook con GPU e Internet. Escribe características en `/kaggle/working/five_cache`, checkpoints en `/kaggle/working/runs_five/<stem>/` y, cuando los cinco modelos completan el entrenamiento, prueba una mezcla y crea `five_stems.mid`.

Los comandos principales, dentro de Kaggle y desde la raíz del repositorio, son:

```bash
python -m polytranscribe.prepare --source-root /kaggle/input/datasets/alonhaviv/guitarset --root /kaggle/working/guitarset
python -m polytranscribe.download_aux --output /kaggle/working/aux --dataset vocadito
python -m polytranscribe.prepare_five --slakh-root /kaggle/input/datasets/alonhaviv/slakh2100/slakh2100_flac_redux --vocadito-root /kaggle/working/aux/vocadito --output /kaggle/working/five_cache --max-slakh-tracks 100
python -m polytranscribe.pipeline --audio /ruta/mezcla.wav --models-root /kaggle/working/runs_five --output-dir /kaggle/working/demo
```

El notebook ejecuta ocho entrenamientos sobre cinco stems: `baseline`, `gated`, `advanced_plain` y `advanced_gated` para guitarra, `advanced_gated` para voz, bajo y piano, y `advanced` para batería. Así las comparaciones de guitarra usan la misma partición, longitud de ventana y tamaño de lote. La opción `--models-root` requiere los cinco checkpoints de producción. Para la comparación de una sola guitarra se conserva `--checkpoint`. Los modelos avanzados usan ventanas de 256 frames y lotes de 4 en entrenamiento; ambos valores se pueden ajustar en Kaggle según la GPU.

Cada entrenamiento guarda `<variante>_last.pt` **cada 25 lotes y al cerrar cada época**, con modelo, optimizador, estado aleatorio, época, lote siguiente y mejor modelo. `<variante>.pt` es el mejor checkpoint para inferencia. El notebook limita la duración de la sesión y permite reanudar desde una salida anterior fijando `PREVIOUS_OUTPUT`. Usar **Save Version** y adjuntar esa salida como **Notebook Output Files** en la próxima sesión; [Kaggle documenta su reutilización](https://www.kaggle.com/docs/notebooks). Los checkpoints nuevos vinculan la reanudación con la fuente, los splits y el límite de temas de su caché. Los checkpoints anteriores de GuitarSet y Vocadito siguen siendo reanudables; los de bajo, piano y batería anteriores a esta comprobación se rechazan porque no permiten verificar con qué versión de Slakh se entrenaron. Al cambiar de BabySlakh a Slakh2100 hay que regenerar la caché y entrenar de nuevo esos tres modelos.

## Evaluación y límites

El entrenamiento informa precisión, recall y F1 por frame, en frames con dos o más notas, por nota con inicio y final, y por nota con inicio solamente. Se usa el emparejamiento de [mir_eval](https://mir-eval.readthedocs.io/stable/api/transcription.html): inicio a ±50 ms, pitch a 50 cents y final a ±máx(50 ms, 20 % de duración). Para batería, el F1 de notas evalúa **clase de golpe e inicio**. Los umbrales y la mejor época se eligen por **F1 de notas en validación**; el test no se usa para ajustar.

Para una comparación externa reproducible, generar primero MIDI con [Basic Pitch](https://github.com/spotify/basic-pitch) u otro modelo sobre los 60 audios del intérprete 05 de GuitarSet, un `{stem}.mid` por fragmento. El evaluador [benchmark_external.py](polytranscribe/benchmark_external.py) lee esos MIDI y calcula las mismas métricas y tolerancias que el experimento propio. El notebook tiene una celda opcional `EXTERNAL_MIDI_DIR` para ejecutarlo; **no ejecuta ni entrena el modelo externo automáticamente**. Antes de comparar con sistemas multitrack en mezclas, se necesitan referencias de mezcla con los cinco instrumentos, un protocolo de alineación común y resultados medidos; hoy no hay evidencia de superar el estado del arte.

El evaluador [eval_slakh.py](polytranscribe/eval_slakh.py) compara la transcripción de guitarra aislada con la obtenida tras Demucs sobre una mezcla Slakh y permite agregar SI-SDR como diagnóstico opcional. Slakh no aporta voz cantada, por lo que todavía no permite medir un F1 conjunto de las **cinco** pistas contra referencia en una sola mezcla. La variante de seis fuentes de Demucs también es experimental; sus autores advierten especialmente sobre la calidad del stem de piano. Esta limitación y la diferencia entre audio real y sintetizado deben informarse junto a cualquier resultado.

## Referencias

- Rashmi M. R. et al. (2026), *A modular approach to polyphonic music transcription using source separation and harmonic-gated pitch tracking*, Journal of Integrated Science and Technology 14(5), 1582. DOI: 10.62110/sciencein.jist.2026.v14.1582.
- Xi, Q. et al. (2018), *GuitarSet: A Dataset for Guitar Transcription*, ISMIR. Dataset: https://doi.org/10.5281/zenodo.3371780.
- Manilow, E. et al. (2019), *Cutting Music Source Separation Some Slakh*, WASPAA. Dataset: https://zenodo.org/records/4599666.
- Bittner, R. et al. (2021), *Vocadito: A dataset of solo vocals with f0, note, and lyric annotations*. Dataset: https://zenodo.org/records/5578807.
