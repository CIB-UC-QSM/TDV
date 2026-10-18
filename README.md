# wTDV-QSM — Weighted Total Deep Variation for Quantitative Susceptibility Mapping

Este repositorio **replica, acelera y extiende** la metodología del paper:

> *TDV Regularization for Improved Iterative QSM* (incluido en el repositorio como [PDF](./3%20-%20TDV%20Regularization%20for%20Improved%20Iterative%20QSM.pdf))

El pipeline resuelve el problema inverso de dipolo magnético para obtener mapas de susceptibilidad cuantitativa (QSM) usando el regularizador *Total Deep Variation* (TDV) dentro de un bucle **ADMM**. La implementación reemplaza la dependencia original de `optox` por operadores nativos de PyTorch, lo que elimina una barrera de instalación significativa y permite compilar el modelo con Triton para máximo rendimiento en GPU moderna.

---

## Índice

- [Contexto y contribuciones](#contexto-y-contribuciones)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Detalles de implementación](#detalles-de-implementación)
- [Resultados](#resultados)
- [Trabajo futuro](#trabajo-futuro)
- [Citación](#citación)

---

## Contexto y contribuciones

El regularizador **Total Deep Variation (TDV)** fue introducido en:

- E. Kobler, A. Effland, K. Kunisch, T. Pock — [*Total Deep Variation for Linear Inverse Problems*](https://arxiv.org/abs/2001.05005), CVPR 2020.
- E. Kobler, A. Effland, K. Kunisch, T. Pock — [*Total Deep Variation: A Stable Regularizer for Inverse Problems*](https://arxiv.org/abs/2006.08789), arXiv 2020.

Este repositorio no es una simple réplica: introduce las siguientes mejoras sobre el trabajo original:

| Contribución | Descripción |
|---|---|
| **Sin `optox`** | Los operadores de convolución se reimplementaron en PyTorch nativo, eliminando la dependencia de compilación de C++/CUDA personalizada |
| **`torch.compile` + Triton** | El modelo VNet se compila con `max-autotune`, aprovechando Tensor Cores sin reentrenamiento |
| **Skull-stripping automático** | Máscara de cerebro dinámica basada en magnitud que estabiliza el condicionamiento del ADMM (~12 iteraciones vs. >50 originales) |
| **Dynamic batching** | `BATCH_SIZE` adaptativo según VRAM disponible; extracción de tríos con `tensor.unfold` para evitar OOM |
| **Calibración dinámica de escala** | `alpha` calibrado en la iteración 0 para adaptar el rango QSM [−0.1, 0.1] al espacio de entrenamiento de la VNet sin reentrenar |
| **Regularización pseudo-3D** | Extensión del prior 2D a los tres ejes ortogonales (axial, coronal, sagital) con promediado de reconstrucciones |

---

## Estructura del repositorio

```
.
├── ddr/                        # Módulo principal de regularizadores
│   ├── __init__.py
│   ├── conv.py                 # Operadores de convolución forward/backward (PyTorch nativo)
│   ├── model.py                # Definición de VNet (red regularizadora)
│   ├── regularizer.py          # Interfaz base para regularizadores
│   ├── tdv.py                  # Implementación del regularizador TDV
│   ├── utils.py                # Utilidades: visualización 3D, métricas (RMSE)
│   ├── denoise.py              # Script de denoising en escala de grises / color
│   └── eigenfunctions.py       # Visualización de eigenfunciones del TDV
│
├── checkpoints/                # Pesos pre-entrenados de la VNet (.pth)
├── figures/                    # Figuras de diagnóstico generadas durante la ejecución
├── results/                    # Resultados de reconstrucción (.mat, .png)
│
├── wTDV_QSM_torch.py           # Pipeline 2D+: tríos de cortes axiales adyacentes
├── wTDV_QSM_torch_3d.py        # Extensión 3D: prior calculado en los tres ejes
│
├── params.mat                  # Datos de entrada: fase, kernel de dipolo, magnitud
├── chi_cosmos.mat              # Ground truth COSMOS para evaluación (RMSE)
│
├── pyproject.toml              # Gestión de dependencias (uv)
└── uv.lock                     # Lockfile reproducible
```

---

## Requisitos

- Python ≥ 3.13
- CUDA 12.x (recomendado; hay fallback a CPU)
- GPU con ≥ 6 GB VRAM (recomendado ≥ 12 GB para máximo rendimiento)

### Dependencias Python

| Paquete | Versión mínima | Uso |
|---|---|---|
| `torch` | 2.12.0 | Motor de cómputo GPU / ADMM / compilación Triton |
| `numpy` | 2.4.6 | Operaciones matriciales |
| `scipy` | — | Carga de `.mat`, morfología binaria (skull-stripping) |
| `nibabel` | 5.4.2 | Soporte NIfTI |
| `cupy-cuda12x` | 14.1.1 | Operaciones CUDA adicionales |
| `scikit-image` | 0.26.0 | Procesamiento de imagen |
| `imageio` | 2.37.3 | E/S de imágenes |
| `matplotlib` | 3.11.0 | Visualización |

> **Nota:** A diferencia del código base original, **no se requiere `optox`**. Los operadores de convolución están reimplementados en PyTorch nativo.

---

## Instalación

Se recomienda usar [`uv`](https://github.com/astral-sh/uv) para reproducibilidad exacta:

```bash
git clone <url-del-repo>
cd tdv

# Instalar dependencias desde el lockfile
uv sync
```

Alternativamente con pip:

```bash
pip install -e .
```

---

## Uso

### Preparar los datos

El pipeline espera dos archivos MATLAB en la raíz del repositorio:

| Archivo | Campo(s) | Descripción |
|---|---|---|
| `params.mat` | `phase_use`, `kernel`, `magn_use` | Fase, kernel de dipolo en Fourier, magnitud |
| `chi_cosmos.mat` | `chi_cosmos` | Susceptibilidad COSMOS (ground truth, opcional para RMSE) |

### Reconstrucción 2D+ (tríos axiales)

```bash
python wTDV_QSM_torch.py
```

Aplica TDV en tríos de cortes axiales adyacentes usando `tensor.unfold`. Equilibrio entre calidad y uso de VRAM. Usa el checkpoint oficial; salida en `results/checkpoints/tdv3-3-25-f32-color/`.

```bash
python wTDV_QSM_torch.py --checkpoint t2delta-100k-s1.pth
```

`--checkpoint` elige otro `.pth` de `checkpoints/`, de 3 canales; tiene que estar en `PARAMS`, al principio del script, con su `μ`, `target_std` y `tolUpdate`, y si no está el script lo dice y no corre. Para buscar parámetros de otro checkpoint: `python grid_search_wtdv.py --checkpoint <archivo>`. Los de Training-TDV:

| Checkpoint | `T` aprendido | `μ` | `target_std` | `tolUpdate` | Iteraciones | RMSE |
|---|---|---|---|---|---|---|
| `tdv3-3-25-f32-color.pth` (oficial) | 1.95·δ_Adam | 0.008 | 0.06 | 0.3 | 36 | **26.45%** |
| `t1p1delta-rapido-100k.pth` | 2.23·δ_Adam | 0.006 | 0.06 | 0.3 | 32 | 27.75% |
| `t2delta-100k-s1.pth` | 2.76·δ_Adam | 0.004 | 0.06 | 1.0 | 26 | 28.19% |
| `t8delta-100k-s1.pth` | 6.53·δ_Adam | 0.006 | 0.07 | 0.75 | 31 | 28.12% |
| `ciego2d-35k.pth` (ciego al ruido) | 7.91·δ_Adam | 0.006 | 0.04 | 1.5 | 13 | 29.11% |

Ninguno alcanza al oficial. Los parámetros salen de `grid_search_wtdv.py` con el solver enmascarado (`results/grid/`): de los cortes en que el `Update` baja de `tolUpdate` y se queda abajo, el de menos iteraciones a 0.05 pp o menos de la mejor RMSE. Por qué rinden peor como prior de QSM está medido en Training-TDV (`docs/mediciones/qsm-prior-4corridas.txt`).

Las salidas de cada corrida están en [`results/checkpoints/`](results/checkpoints/): una carpeta por checkpoint, con la QSM, las tres figuras de diagnóstico y el log de la corrida (`corrida.log`).

### Reconstrucción pseudo-3D

```bash
python wTDV_QSM_torch_3d.py
```

Calcula el prior TDV en los tres ejes ortogonales (axial, coronal, sagital) y promedia las reconstrucciones. Mejor RMSE al costo de ~4× el tiempo. Acepta el mismo `--checkpoint`, sólo de 3 canales porque arma tripletes en los tres ejes, con el `μ` y el `target_std` del 2D; su criterio de parada está ajustado con el oficial. Salida en `results/checkpoints/<checkpoint>-3d/`.

### Salidas comunes

Ambos scripts generan, en `results/checkpoints/<checkpoint>/` (el 3D en `<checkpoint>-3d/`), con el checkpoint, los parámetros, las iteraciones y el RMSE en el título:
- `.mat` — susceptibilidad reconstruida + tiempo de ejecución + número de iteraciones
- `.png` de la QSM — visualización 3D ortogonal de la susceptibilidad
- imágenes de diagnóstico de variables internas (`v`, `z1/zz`, `s1/sz`). `v` y `s1` se ven ruidosas por construcción: en el punto fijo del ADMM `s1 = v − TDV(v)`, lo que el denoiser le saca a `v`; la QSM es `x`, no `v`

Las imágenes de `results/` y `figures/` son las originales del repo y ningún script las escribe. Las carpetas `<checkpoint>-mascara-<ninguna|qsm|ambas>/` y `tdv3-3-25-f32-color-3d-sin-mascara/` son de la comparación del item 6 del bloque final del script.

---

## Detalles de implementación

### ADMM

El problema de optimización resuelto es:

```
min_x  ½ ‖W(Dx - φ)‖² + μ · R_TDV(x)
```

Resuelto con ADMM en tres bloques:

1. **Actualización de `x`** — Paso cuadrático en dominio de Fourier (solución cerrada vía FFT).
2. **Paso proximal `z1` — Prior TDV** — La VNet procesa tríos de cortes escalados con `alpha`; el escalado dinámico se calibra en la iteración 0 para que la std del input alcance `target_std = 0.06`. Su entrada y su salida se multiplican por la máscara del cerebro en cada iteración, como hacen LPCNN y MoDL-QSM con su red, y el mapa final también: afuera no hay datos (`W² = 0`) y el TDV amplifica. Proyectar también `x` tras la FFT, como WH-QSM de FANSI, dio peor (bloque final del script, item 6).
3. **Paso proximal `z2` — Data fidelity** — Resolución analítica con pesos de magnitud `W²`.

El bucle corta cuando el cambio relativo de `x` entre iteraciones (`Update`, en %) baja de `tolUpdate`. El `Update` se mide sólo dentro de la máscara del cerebro, que es donde la QSM se usa y se evalúa: afuera no hay datos (`W² = 0`) y el fondo nunca termina de asentarse. Equivale al update de FANSI sobre `mask.*x`, como en su variante WH-QSM, que proyecta χ en la máscara en cada iteración.

### Skull-stripping automático

1. Umbral dinámico: `0.05 × media_cerebro`
2. `binary_fill_holes` + `binary_closing` morfológico (kernel 3×3×3, 2 iteraciones)
3. Suavizado gaussiano → máscara dura (> 0.5)

Esto reduce el número de condición del ADMM, pasando de >50 iteraciones a convergencia en ~12–31 según tolerancia.

### Optimizaciones de hardware

| Técnica | Efecto |
|---|---|
| `torch.compile(mode="max-autotune")` | Fusión de kernels Triton, aprovecha Tensor Cores |
| `torch.autocast` (bf16/fp16) | Reduce VRAM a la mitad en inferencia |
| Dynamic batching | Adapta `BATCH_SIZE` a la VRAM disponible (16 / 64 / 256) |
| `tensor.unfold` | Extrae tríos de cortes sin copias de memoria (evita OOM) |
| Denominadores precalculados | `μ₂K² + μ` y `W² + μ₂` computados una sola vez fuera del bucle |

---

## Resultados

Evaluados sobre el dataset COSMOS incluido (`chi_cosmos.mat`) con parámetros `μ = 0.008`, `μ₂ = 1.0`, `target_std = 0.06`.

### Script 2D+ (`wTDV_QSM_torch.py`)

| Métrica | Valor |
|---|---|
| RMSE final | **26.45%** |
| Iteraciones hasta convergencia | 36 |
| Tiempo de ejecución (GPU ≥ 12 GB VRAM) | ~71 s |
| `tolUpdate` (Update dentro del cerebro) | 0.3 |

> Con la entrada y la salida del TDV enmascaradas en cada iteración y el mapa final enmascarado. Sin máscaras daba 27.38% en 31 iteraciones (~64 s). Con el `Update` sobre todo el volumen, el criterio anterior, `tolUpdate = 0.39` cortaba en el mismo punto (31–33 iteraciones, 27.34%).
>
> El pseudo-3D de abajo enmascara igual sus tres priors y queda por debajo del 2D (25.94% contra 26.45%).

### Script pseudo-3D (`wTDV_QSM_torch_3d.py`)

| Métrica | Valor |
|---|---|
| RMSE final | **25.94%** |
| Iteraciones | 100 (el criterio de parada no se dispara, ver nota) |
| Tiempo de ejecución (GPU ≥ 12 GB VRAM) | ~509 s |
| `tolUpdate` / `finUpdate` (Update sobre todo el volumen) | 1.15 / 1.30 |

> Con la entrada y la salida del TDV enmascaradas en los tres ejes y el mapa final enmascarado. Sin máscaras daba 26.89% en 26 iteraciones (~143 s). El corte (bajar de `tolUpdate` y después pasar `finUpdate`) se ajustó con sólo la salida enmascarada; con la entrada también enmascarada el `Update` baja sólo hasta 1.24 y se queda en ~1.39, así que corre las 100; la RMSE ya está en 25.95% en la iteración 40.

> La mejora de RMSE del 3D sobre el 2D (0.51 pp: 25.94% contra 26.45%) se logra incorporando coherencia isotrópica mediante el promediado de los priors axial, coronal y sagital.

> **Nota sobre los datos:** los experimentos actuales usan una fase de alta SNR (casi sin ruido). Los resultados deben interpretarse en ese contexto; la evaluación con fases ruidosas es parte del trabajo futuro.

---

## Trabajo futuro

- [ ] **Evaluación con ruido sintético:** agregar ruido gaussiano controlado a la fase para evaluar la robustez del regularizador TDV en condiciones realistas de adquisición.
- [ ] **Estrategias alternativas al promediado 3D:** explorar combinaciones ponderadas por confianza, fusión aprendida, o regularización isotrópica verdadera en lugar de promediar ejes ortogonales.
  - Ya probado el 2026-07-04 (rama `worktree-qsm-fusion-experiments`, integrada; el script `wTDV_QSM_torch_3d_experiments.py` quedó en la historia, `git show 92a25e6:wTDV_QSM_torch_3d_experiments.py`, y corre sobre el código con Optox anterior a `e7cdf19`). Con `μ = 0.02`, `target_std = 0.1`, sobre-relajación ADMM 1.6 y parada por meseta sin COSMOS: fusión espectral 27.37% en 12 iteraciones, promedio 27.65%, mediana 27.98%; reforzar el cono del ángulo mágico empeora (35–39%) y `μ` adaptativo diverge. Figuras en `figures/exp_{spectral,average,magic_angle,cosmos_gt}.png`. Falta portarlo al `wTDV_QSM_torch_3d.py` actual.
- [ ] **Reentrenamiento del prior:** explorar fine-tuning de la VNet sobre datos QSM para eliminar la necesidad del escalado dinámico `alpha`.
- [ ] **Comparación cuantitativa completa:** benchmark contra métodos clásicos (MEDI, iLSQR, STAR-QSM) sobre el mismo dataset.

---

## Citación

Si usas este código en tu investigación, por favor cita el trabajo original:

```bibtex
@InProceedings{KoEf20,
  Title     = {Total Deep Variation for Linear Inverse Problems},
  Author    = {Kobler, Erich and Effland, Alexander and Kunisch, Karl and Pock, Thomas},
  Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  Year      = {2020}
}
```

---

## Licencia

Distribuido bajo los términos de la [licencia incluida](./LICENSE).
