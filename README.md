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

Aplica TDV en tríos de cortes axiales adyacentes usando `tensor.unfold`. Equilibrio entre calidad y uso de VRAM. Salida en `results/wTDV_QSM_torch.mat`.

### Reconstrucción pseudo-3D

```bash
python wTDV_QSM_torch_3d.py
```

Calcula el prior TDV en los tres ejes ortogonales (axial, coronal, sagital) y promedia las reconstrucciones. Mejor RMSE al costo de ~2× el tiempo. Salida en `results/wTDV_QSM_torch_3d.mat`.

### Salidas comunes

Ambos scripts generan automáticamente:
- `results/*.mat` — susceptibilidad reconstruida + tiempo de ejecución + número de iteraciones
- `results/*.png` — visualización 3D ortogonal de la susceptibilidad
- `figures/` — imágenes de diagnóstico de variables internas (`v`, `z1/zz`, `s1/sz`)

---

## Detalles de implementación

### ADMM

El problema de optimización resuelto es:

```
min_x  ½ ‖W(Dx - φ)‖² + μ · R_TDV(x)
```

Resuelto con ADMM en tres bloques:

1. **Actualización de `x`** — Paso cuadrático en dominio de Fourier (solución cerrada vía FFT).
2. **Paso proximal `z1` — Prior TDV** — La VNet procesa tríos de cortes escalados con `alpha`; el escalado dinámico se calibra en la iteración 0 para que la std del input alcance `target_std = 0.06`.
3. **Paso proximal `z2` — Data fidelity** — Resolución analítica con pesos de magnitud `W²`.

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
| RMSE final | **27.34%** |
| Iteraciones hasta convergencia | 31 |
| Tiempo de ejecución (GPU ≥ 12 GB VRAM) | ~67 s |
| `tolUpdate` | 0.39 |

### Script pseudo-3D (`wTDV_QSM_torch_3d.py`)

| Métrica | Valor |
|---|---|
| RMSE final | **26.89%** |
| Iteraciones hasta convergencia | 25 |
| Tiempo de ejecución (GPU ≥ 12 GB VRAM) | ~144 s |
| `tolUpdate` | 1.05 |

> La mejora de RMSE del 3D (+0.45 pp) se logra incorporando coherencia isotrópica mediante el promediado de los priors axial, coronal y sagital, al costo de ~2× el tiempo de cómputo.

> **Nota sobre los datos:** los experimentos actuales usan una fase de alta SNR (casi sin ruido). Los resultados deben interpretarse en ese contexto; la evaluación con fases ruidosas es parte del trabajo futuro.

---

## Trabajo futuro

- [ ] **Evaluación con ruido sintético:** agregar ruido gaussiano controlado a la fase para evaluar la robustez del regularizador TDV en condiciones realistas de adquisición.
- [ ] **Estrategias alternativas al promediado 3D:** explorar combinaciones ponderadas por confianza, fusión aprendida, o regularización isotrópica verdadera en lugar de promediar ejes ortogonales.
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
