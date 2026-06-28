# Reentrenamiento de TDV-VNet para QSM: Teoría e Implementación

## 1. Contexto: ¿Qué se reentrenarría?

La VNet tiene dos componentes entrenables:

```
VNet
├── R: TDV (regularizador) ← pesos de las convoluciones (~1.5M params)
├── T: stopping time        ← 1 escalar aprendido
├── λ: balance R vs D       ← 1 escalar aprendido
└── D: Dataterm             ← NO tiene parámetros entrenables (es fijo)
```

**Lo que se reentrenar:** Los pesos de TDV (convoluciones K1, macro-blocks, KN), T y λ.  
**Lo que NO se reentrena:** El QSMDataterm — es el modelo físico fijo (convoulción dipolar).

La idea es que TDV aprenda qué **regularización** es óptima para el problema de QSM, en vez de la regularización genérica de denoising de imágenes naturales que tiene ahora.

---

## 2. Teoría: Variational Network Training

### 2.1 La VNet como optimización "unrolled"

VNet realiza S pasos de descenso por gradiente sobre:

```
min_x  R(x) + λ · D(x, z)
```

Cada paso (ver [model.py L174-L186](file:///home/santicien/Documents/tdv/model.py#L174-L186)):

```
Con prox:   x_{s+1} = prox_D(x_s - τ·∇R(x_s), z, λ/S)
Sin prox:   x_{s+1} = x_s - τ·∇R(x_s) - (λ/S)·∇D(x_s, z)
```

Donde:
- `∇R(x)` = gradiente del regularizador TDV (red neuronal)
- `∇D(x, z)` = gradiente del dataterm (física de QSM: `A^T W²(Ax-z)`)
- `prox_D` = operador proximal del dataterm

### 2.2 Entrenamiento end-to-end

Se entrena la red completa como un **unrolled optimization**:

```
                    S pasos de VNet
    z (fase) ───┐                      
                ├──► [paso 1] → [paso 2] → ... → [paso S] → x_S (predicción)
    x_0 (init) ─┘                                              │
                                                                 ↓
    χ_gt (ground truth) ──────────────────────────────────► Loss(x_S, χ_gt)
                                                                 │
                                                                 ↓
                                                          Backprop → actualizar R, T, λ
```

### 2.3 Loss function

La loss estándar para variational networks es:

```
L = Σ_{samples} ‖x_S - χ_gt‖² 
```

Donde `x_S` es la salida del último paso de VNet y `χ_gt` es el ground truth.

Opcionalmente se puede agregar un término de "deep supervision" que penaliza pasos intermedios:

```
L = Σ_s  w_s · ‖x_s - χ_gt‖²    (w_s creciente, más peso a pasos finales)
```

---

## 3. ¿Qué datos necesitas?

### 3.1 Pares de entrenamiento

Necesitas pares `(φ_medida, χ_gt)`:

| Dato | Descripción | Cómo obtenerlo |
|------|-------------|----------------|
| `χ_gt` | Mapa de susceptibilidad ground truth | COSMOS reconstruction, o phantoms simulados |
| `φ_medida` | Fase local medida | `φ = D * χ_gt + ruido` (simulado) o datos reales |
| `W` | Peso de magnitud | Magnitud de la imagen (datos reales) |
| `D` | Kernel dipolar | Calculado analíticamente según orientación B0 |

### 3.2 Opción A: Datos simulados (más fácil)

```python
# Generar datos de entrenamiento simulados
chi_gt = load_cosmos_or_phantom()          # ground truth susceptibility
D_kernel = compute_dipole_kernel(B0_dir)   # dipole kernel
phase = ifft(D_kernel * fft(chi_gt))       # forward model
noise = sigma * randn_like(phase)          # add noise
phase_noisy = phase + noise                # measured phase
```

**Ventaja:** Infinitos datos, control del nivel de ruido.  
**Desventaja:** Domain gap — la red podría no generalizar bien a datos reales.

### 3.3 Opción B: Datos reales con COSMOS como ground truth

Si tienes datos multi-orientación, puedes obtener χ_gt con COSMOS y usar cada orientación individual como φ_medida.

**Ventaja:** Sin domain gap.  
**Desventaja:** Pocos datos (cada sujeto da ~12 orientaciones → 12 pares).

### 3.4 Slicing para VNet 2D

Como VNet procesa slices 2D, los pares volumétricos se cortan en slices:

```python
for k in range(Nz):
    chi_slice = chi_gt[:, :, k]       # (Nx, Ny)
    phase_slice = phase[:, :, k]      # (Nx, Ny)
    # → un sample de entrenamiento
```

Con el modelo color (3 canales), 3 slices adyacentes forman un sample:

```python
for k in range(1, Nz-1):
    chi_3ch = chi_gt[:, :, k-1:k+2]     # (Nx, Ny, 3)
    phase_3ch = phase[:, :, k-1:k+2]    # (Nx, Ny, 3)
    # reshape a (1, 3, Nx, Ny)
```

---

## 4. Implementación del training loop

### 4.1 Estructura del código necesario

```
tdv/
├── train_qsm.py          # [NUEVO] script principal de entrenamiento
├── dataset_qsm.py        # [NUEVO] dataset class para cargar pares QSM
├── model.py               # [YA EXISTE] VNet + QSMDataterm (ya mejorado)
├── ddr/                   # [YA EXISTE] TDV regularizador (no se toca)
└── checkpoints/
    ├── tdv3-3-25-f32-color.pth    # checkpoint original (denoising)
    └── tdv-qsm-color.pth         # [NUEVO] checkpoint QSM reentrenado
```

### 4.2 Pseudocódigo del training loop

```python
# === Configuración ===
# Cargar checkpoint de denoising como inicialización (transfer learning)
checkpoint = torch.load('checkpoints/tdv3-3-25-f32-color.pth')
config = checkpoint['config']

# Cambiar dataterm a QSM
config['D']['type'] = 'qsm'
config['D']['config'] = {
    'use_prox': False,       # usar grad porque tenemos pesos W
    'dipole_kernel': D_2d,   # kernel dipolar 2D proyectado
    'weight': W_2d,          # magnitud weight (opcional)
}

# Crear VNet con QSMDataterm
vnet = VNet(config)
# Cargar SOLO los pesos del regularizador R (TDV) del checkpoint de denoising
# Los pesos de T y lambda se reinicializan
vnet.R.load_state_dict(checkpoint['model_R'])  # transfer learning

# === Training loop ===
optimizer = torch.optim.Adam(vnet.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for chi_gt, phase_measured in dataloader:
        # Forward: VNet procesa la fase y produce una estimación de χ
        x_0 = torch.zeros_like(chi_gt)  # inicialización en cero
        x_all = vnet(x_0, phase_measured)
        x_pred = x_all[-1]  # última iteración
        
        # Loss
        loss = F.mse_loss(x_pred, chi_gt)
        
        # Backward + update
        optimizer.zero_grad()
        loss.backward()
        
        # Proyectar parámetros (mantener constraints de TDV)
        for p in vnet.parameters():
            if hasattr(p, 'proj'):
                p.proj()
        
        optimizer.step()
    
    # Guardar checkpoint
    torch.save({
        'config': config,
        'model': vnet.state_dict(),
    }, 'checkpoints/tdv-qsm-color.pth')
```

### 4.3 Detalles importantes

| Aspecto | Detalle |
|---------|---------|
| **Transfer learning** | Inicializar R (TDV) desde pesos de denoising acelera convergencia |
| **Learning rate** | ~1e-4 a 1e-5 (fino, porque partimos de pesos pre-entrenados) |
| **Proyecciones** | TDV tiene constraints: zero-mean kernels, bounded norms. Hay que llamar `p.proj()` después de cada update (ver [conv.py L45-L58](file:///home/santicien/Documents/tdv/ddr/conv.py#L45-L58)) |
| **Batch size** | Depende de GPU RAM. Con slices 160×160 y 3 canales, ~8-16 por batch |
| **Épocas** | ~50-200 dependiendo del tamaño del dataset |
| **Data augmentation** | Flips, rotaciones de 90°, crops aleatorios |

---

## 5. Complejidad estimada

| Componente | Líneas de código | Dificultad | Dependencias |
|------------|:---:|:---:|---|
| `dataset_qsm.py` | ~80-120 | Media | Datos de entrenamiento (COSMOS o simulados) |
| `train_qsm.py` | ~150-200 | Media | dataset, model.py |
| Generación de datos simulados | ~50-80 | Baja | numpy, kernel dipolar |
| Modificar VNet para cargar pesos parciales | ~20-30 | Baja | model.py |
| **Total** | **~300-430** | **Media** | |

> [!IMPORTANT]
> **La barrera principal no es el código, sino los datos.** Sin ground truth de susceptibilidad (COSMOS, phantoms), no hay con qué entrenar. Si tienes datos COSMOS o acceso a simulaciones, el código de entrenamiento se puede escribir en 1-2 días.

---

## 6. Sobre modificar VNet para 3D

> "No podemos modificar VNet para que tome volúmenes?"

**Técnicamente sí, pero es un proyecto mayor:**
- Cambiar todos los `Conv2d` → `Conv3d` en [ddr/conv.py](file:///home/santicien/Documents/tdv/ddr/conv.py) y [ddr/tdv.py](file:///home/santicien/Documents/tdv/ddr/tdv.py)
- `optoth.pad.pad2d` → `pad3d`
- Los `ConvScale2d` (downsampling) → `ConvScale3d`
- **Todos los pesos pre-entrenados son incompatibles** → reentrenamiento obligatorio
- **Memoria GPU**: un volumen 160³ con 32 features = 160³×32×4 bytes ≈ **524 MB** por capa. Con múltiples escalas y macro-blocks, fácilmente 10-20 GB de VRAM solo para el forward pass
- Estimación: ~500-800 líneas de código + reentrenamiento + GPU potente

**Recomendación:** El enfoque 2D con slices (lo que tienes ahora) es pragmático y publicable. El 3D sería un proyecto de investigación en sí mismo.
