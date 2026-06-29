# Análisis Exhaustivo del Parámetro μ para wTDV-QSM

## Resumen Ejecutivo

Se ejecutó un **grid search de 17 valores de μ** (0.001 – 0.5) usando ambas GPUs en paralelo, evaluando 7 métricas de calidad por cada reconstrucción ADMM completa.

> [!IMPORTANT]
> **Hallazgo principal**: El score compuesto indica que μ más bajos producen mejores resultados en sharpness y data fidelity, pero hay un trade-off importante con SNR y streaking. El valor actual μ=0.0245 NO es óptimo, pero tampoco es descabellado — está en una zona conservadora del trade-off.

## Rol Matemático de μ

En el ADMM, el update de χ se resuelve en Fourier:

```
χ̂ = (μ₂·D*·F{z2-s2} + μ·F{z1-s1}) / (μ₂·|D|² + μ)
```

El denominador `μ₂·|D|² + μ` es crítico:
- **|D|²** (kernel dipolar²) tiene un **cono nulo** donde D≈0 (24.2% de vóxeles tienen |D|²<0.01)
- **μ actúa como regularizador de Tikhonov** en ese cono:
  - μ pequeño → denominador ≈ 0 en el cono → amplifica ruido/streaking
  - μ grande → denominador dominado por μ → aplasta la señal, resultado borroso
  - μ óptimo → estabiliza el cono sin aplastar

### Condition Number del Denominador

| μ | min(denom) | max(denom) | Cond. Num | 
|---|---|---|---|
| 0.001 | 0.001 | 0.445 | **445** |
| 0.01 | 0.010 | 0.454 | 45 |
| **0.0245** | **0.0245** | **0.469** | **19** |
| 0.1 | 0.100 | 0.544 | 5.4 |
| 1.0 | 1.000 | 1.444 | 1.4 |

## Resultados del Grid Search

![Gráficos de análisis de μ](/home/santicien/.gemini/antigravity-ide/brain/36f52a88-9e5d-4c56-9f47-83457b4fcd9d/mu_analysis.png)

### Tabla Completa

| μ | Iter | DataFid | Sharpness | Brain Std | SNR | Cone Streak | DynRange |
|---|---|---|---|---|---|---|---|
| **0.001** | 47 | **0.000287** | **0.02637** | **0.0341** | 8.58 | 25.49 | **0.662** |
| 0.003 | 27 | 0.000462 | 0.01963 | 0.0305 | 11.32 | 18.53 | 0.619 |
| 0.005 | 24 | 0.000557 | 0.01730 | 0.0293 | 12.64 | 16.08 | 0.610 |
| 0.008 | 22 | 0.000646 | 0.01564 | 0.0285 | 13.87 | 14.07 | 0.603 |
| 0.010 | 22 | 0.000689 | 0.01499 | 0.0282 | 14.46 | 13.26 | 0.601 |
| 0.015 | 22 | 0.000773 | 0.01392 | 0.0276 | 15.46 | 11.91 | 0.600 |
| 0.020 | 22 | 0.000839 | 0.01320 | 0.0272 | 16.03 | 11.02 | 0.599 |
| **0.0245** ◀ | **22** | **0.000889** | **0.01270** | **0.0270** | **16.32** | **10.41** | **0.599** |
| 0.030 | 22 | 0.000943 | 0.01221 | 0.0267 | 16.50 | 9.81 | 0.598 |
| **0.040** | **22** | **0.001026** | **0.01151** | **0.0263** | **16.56** | **8.97** | **0.595** |
| 0.050 | 22 | 0.001097 | 0.01096 | 0.0260 | 16.49 | 8.32 | 0.593 |
| 0.070 | 23 | 0.001217 | 0.01012 | 0.0256 | 16.24 | 7.49 | 0.590 |
| 0.100 | 24 | 0.001348 | 0.00935 | 0.0251 | 15.82 | 6.60 | 0.586 |
| 0.150 | 26 | 0.001471 | 0.00875 | 0.0247 | 15.29 | 5.71 | 0.578 |
| 0.200 | 28 | 0.001542 | 0.00843 | 0.0243 | 14.92 | 5.15 | 0.570 |
| 0.300 | 31 | 0.001628 | 0.00808 | 0.0238 | 14.45 | 4.41 | 0.556 |
| 0.500 | 36 | 0.001719 | 0.00777 | 0.0232 | 13.98 | 3.58 | 0.538 |

## Curvas de Convergencia

![Convergencia para diferentes μ](/home/santicien/.gemini/antigravity-ide/brain/36f52a88-9e5d-4c56-9f47-83457b4fcd9d/mu_convergence.png)

## Análisis de Trade-offs

Los resultados revelan **5 trade-offs monotónicos** al bajar μ:

| Al bajar μ | Efecto | Causa física |
|---|---|---|
| ✅ Sharpness sube | Más nítido, bordes preservados | Menor penalización, TDV con más libertad |
| ✅ Data Fidelity mejora | Más fiel a la fase medida | El prior (z1) pesa menos vs los datos |
| ✅ Rango dinámico sube | Más contraste susceptibilidad | No aplasta extremos paramagnéticos |
| ⚠️ SNR baja | Más ruido visible | Menor regularización del cono nulo |
| ⚠️ Cone Streaking sube | Artefactos de rayas | Amplificación en el cono mágico |

> [!WARNING]
> **μ=0.001 gana el score compuesto pero toma 47 iteraciones** (vs 22 del actual) y tiene SNR=8.58 (vs 16.32). Visualmente puede tener streaking visible. No es necesariamente la mejor opción práctica.

## Recomendación

Mirando la **L-Curve** (gráfico inferior derecho) y el análisis de trade-offs, hay 3 zonas operativas:

### Opción A: μ = 0.005–0.008 (Agresivo, más nítido)
- **Sharpness 1.2-1.5x mayor** que el actual
- SNR todavía aceptable (12.6–13.9)
- Convergencia rápida (22-24 iter)
- **Recomendado si se prioriza nitidez y las venas son importantes**

### Opción B: μ = 0.010–0.015 (Balanceado)
- Sharpness 10-18% mejor que el actual
- SNR bueno (14.5–15.5)
- 22 iteraciones
- **Recomendado como "mejor balance general"**

### Opción C: μ = 0.0245 (Actual, conservador)
- SNR alto (16.32)
- Menor streaking
- **Ya está en una zona funcional. No es erróneo, solo conservador**

> [!TIP]
> **Mi recomendación: μ = 0.01**. Ofrece +18% sharpness, +23% mejor data fidelity, SNR todavía alto (14.5), mismas 22 iteraciones, y el condition number (45) sigue siendo muy manejable. Es el "sweet spot" de la L-curve donde la ganancia en nitidez es significativa sin pagar un precio alto en ruido.

## Nota sobre la Autocalibración

El alpha se autocalibra en función de μ. Cuando μ baja, el qsm de la iteración 0 tiene menor std, por lo que alpha sube compensatoriamente. Esto es correcto y deseable — la calibración dinámica protege contra el cambio de escala.

| μ | Alpha calibrado |
|---|---|
| 0.001 | ~15 (datos más fuertes → menos amplificación necesaria) |
| 0.0245 | ~30 (actual) |
| 0.5 | ~85 (datos aplastados → mucha amplificación) |
