# Transcripción polifónica en cinco pistas

El sistema sigue la estructura modular de [Rashmi et al. (2026)](https://doi.org/10.62110/sciencein.jist.2026.v14.1582): **separación → estimación de notas → filtro armónico → MIDI**. El paper estima un solo F0 por stem; aquí las notas con altura se predicen de forma multi-etiqueta para permitir acordes dentro de guitarra y piano. `htdemucs_6s` separa seis fuentes; se transcriben `vocals`, `drums`, `bass`, `guitar` y `piano`. El audio `other` se conserva sin inventarle una pista de notas.

## Modelos y datos

| Stem | Datos para la primera versión | Salida |
| --- | --- | --- |
| `guitar` | [GuitarSet 1.1.0](https://doi.org/10.5281/zenodo.3371780): 360 fragmentos de guitarra acústica con notas por cuerda. En Kaggle se adjunta [alonhaviv/guitarset](https://www.kaggle.com/datasets/alonhaviv/guitarset). | Notas MIDI simultáneas. |
| `vocals` | [Vocadito v3](https://zenodo.org/records/5578807): 40 fragmentos de canto solista con notas manuales. | Notas MIDI de voz. |
| `bass`, `piano`, `drums` | [Slakh2100-redux](https://zenodo.org/records/4599666): mezclas, stems y MIDI alineado. [BabySlakh](https://zenodo.org/records/4603844) aporta 20 temas a 16 kHz para probar el pipeline. | Notas MIDI de bajo y piano; nueve clases de golpes de batería. |

Slakh no contiene voces cantadas, y GuitarSet solo anota guitarra. Por eso la primera versión usa **tres fuentes de datos**. Los cuatro modelos con altura comparten la misma arquitectura, pero tienen checkpoints y rangos MIDI propios: voz 36–88, bajo 28–67, guitarra 40–82 y piano 21–108. La batería usa una CNN sobre Mel y clases de percusión General MIDI: bombo, caja, hi-hat cerrado y abierto, tom, crash, ride, clap/rim y otra percusión. Demucs es preentrenado y no se ajusta en estos datasets. El Slakh2100 original ocupa unos 105 GB; la [versión redux a 16 kHz](https://zenodo.org/records/7708270) ocupa unos 49 GB. `MAX_SLAKH_TRACKS` limita cuántos temas se procesan dentro de Kaggle.

**Alcance experimental:** BabySlakh y Vocadito son pequeños; permiten verificar entrenamiento e inferencia, no demostrar generalización de cinco instrumentos. Para los resultados finales habrá que usar una muestra amplia de Slakh2100-redux y ampliar los datos de voz, por ejemplo con [Annotated-VocalSet](https://zenodo.org/records/7061507). Los modelos aún no se entrenaron en Kaggle y no se reportan precisiones inventadas.

## Método

Para los cuatro stems con altura, la rama A propone una probabilidad por pitch y frame a partir de la fundamental, armónicos y subarmónicos. La rama B usa Mel de 256 bandas y croma para filtrar cada pitch. En el notebook de cinco stems se entrena la variante `gated` para voz, bajo, guitarra y piano. La batería se modela como detección multi-etiqueta de golpes; no se le aplica una compuerta de altura. Demucs se ejecuta **una vez por mezcla**, y después se cargan los cinco modelos pequeños para producir `five_stems.mid` y un MIDI por stem.

El experimento anterior de guitarra queda disponible en [train_guitarset.ipynb](kaggle/train_guitarset.ipynb): `baseline` no filtra, `paper_gate` aplica al **mismo checkpoint base** la máscara global fija al percentil 10 del paper y `gated` aprende una compuerta por pitch y frame. Sirve para estudiar el aporte del filtro antes de extender la comparación a otros stems; [train_five_stems.ipynb](kaggle/train_five_stems.ipynb) es ahora el flujo principal de cinco pistas.

## Entrenar en Kaggle

1. Adjuntar GuitarSet como **Add Input** en `/kaggle/input/guitarset` y abrir [train_five_stems.ipynb](kaggle/train_five_stems.ipynb). Primero hay que publicar este código en el remoto para que el notebook pueda clonarlo.
2. Indicar `SLAKH_ROOT` y `VOCADITO_ROOT` si los datasets ya están adjuntos. Si quedan vacíos, `AUTO_DOWNLOAD_AUX = True` descarga las versiones públicas de Vocadito (58 MB) y BabySlakh (~883 MB) desde Zenodo, verifica MD5 y las extrae. Para un experimento mayor, adjuntar Slakh2100-redux, fijar `SLAKH_ROOT`, desactivar la descarga auxiliar correspondiente y aumentar `MAX_SLAKH_TRACKS`.
3. Ejecutar el notebook con GPU e Internet. Escribe características en `/kaggle/working/five_cache`, checkpoints en `/kaggle/working/runs_five/<stem>/` y, cuando los cinco modelos completan el entrenamiento, prueba una mezcla y crea `five_stems.mid`.

Los comandos principales, dentro de Kaggle y desde la raíz del repositorio, son:

```bash
python -m polytranscribe.prepare --source-root /kaggle/input/guitarset --root /kaggle/working/guitarset
python -m polytranscribe.download_aux --output /kaggle/working/aux --dataset both
python -m polytranscribe.prepare_five --slakh-root /kaggle/working/aux/babyslakh --vocadito-root /kaggle/working/aux/vocadito --output /kaggle/working/five_cache
python -m polytranscribe.pipeline --audio /ruta/mezcla.wav --models-root /kaggle/working/runs_five --output-dir /kaggle/working/demo
```

El notebook ejecuta los cinco comandos de entrenamiento con el cache correspondiente: `gated` para voz, bajo, guitarra y piano; `baseline` para batería. La opción `--models-root` requiere los cinco checkpoints. Para la comparación antigua de una sola guitarra se conserva `--checkpoint`.

Cada entrenamiento guarda `<variante>_last.pt` **cada 25 lotes y al cerrar cada época**, con modelo, optimizador, estado aleatorio, época, lote siguiente y mejor modelo. `<variante>.pt` es el mejor checkpoint para inferencia. El notebook limita la duración de la sesión y permite reanudar desde una salida anterior fijando `PREVIOUS_OUTPUT`. Usar **Save Version** y adjuntar esa salida como **Notebook Output Files** en la próxima sesión; [Kaggle documenta su reutilización](https://www.kaggle.com/docs/notebooks). Si se cambia de BabySlakh a Slakh2100, activar `REBUILD_AUX_CACHE` y entrenar de nuevo los modelos afectados.

## Evaluación y límites

El entrenamiento informa precisión, recall y F1 por frame, en frames con dos o más notas, por nota con inicio y final, y por nota con inicio solamente. Se usa el emparejamiento de [mir_eval](https://mir-eval.readthedocs.io/stable/api/transcription.html): inicio a ±50 ms, pitch a 50 cents y final a ±máx(50 ms, 20 % de duración). Para batería, el F1 de notas evalúa **clase de golpe e inicio**. Los umbrales y la mejor época se eligen en validación; el test no se usa para ajustar.

El evaluador [eval_slakh.py](polytranscribe/eval_slakh.py) compara la transcripción de guitarra aislada con la obtenida tras Demucs sobre una mezcla Slakh y permite agregar SI-SDR como diagnóstico opcional. Slakh no aporta voz cantada, por lo que todavía no permite medir un F1 conjunto de las **cinco** pistas contra referencia en una sola mezcla. La variante de seis fuentes de Demucs también es experimental; sus autores advierten especialmente sobre la calidad del stem de piano. Esta limitación y la diferencia entre audio real y sintetizado deben informarse junto a cualquier resultado.

## Referencias

- Rashmi M. R. et al. (2026), *A modular approach to polyphonic music transcription using source separation and harmonic-gated pitch tracking*, Journal of Integrated Science and Technology 14(5), 1582. DOI: 10.62110/sciencein.jist.2026.v14.1582.
- Xi, Q. et al. (2018), *GuitarSet: A Dataset for Guitar Transcription*, ISMIR. Dataset: https://doi.org/10.5281/zenodo.3371780.
- Manilow, E. et al. (2019), *Cutting Music Source Separation Some Slakh*, WASPAA. Dataset: https://zenodo.org/records/4599666.
- Bittner, R. et al. (2021), *Vocadito: A dataset of solo vocals with f0, note, and lyric annotations*. Dataset: https://zenodo.org/records/5578807.
