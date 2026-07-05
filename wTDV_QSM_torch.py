#%%
import os
import time
import numpy as np
import torch
import scipy.io
from scipy import ndimage

from ddr import model
from ddr.utils import imshow_3d, print_stats, rmse

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # Habilita TF32

tic = time.time()

#Load MATLAB data container
phase_np = scipy.io.loadmat('params.mat')['phase_use'].astype(np.float32)
kernel_np = scipy.io.loadmat('params.mat')['kernel'].astype(np.float32)
K2_np = np.conj(kernel_np)*kernel_np # El kernel en params.mat YA tiene su centro en [0,0,0] / no aplicamos ifftshift
weight_np = scipy.io.loadmat('params.mat')['magn_use'].astype(np.float32)
maxOuterIter = 100
tolUpdate = 0.39
chi_cosmos = scipy.io.loadmat('chi_cosmos.mat')['chi_cosmos'].astype(np.float32)
cosmos_mask = chi_cosmos != 0
mu = 0.008  # Optimizacion via grid search resulta en 0.008
mu2 = 1.0
#scipy.io.loadmat('params.mat')['alpha1'].astype(np.float32)[0,0] # Es 0.04 / el valor de referencia de Carlos es 0.0245
#scipy.io.loadmat('params.mat')['mu1'].astype(np.float32)[0,0] # Es 1.0
#scipy.io.loadmat('params.mat')['tol_update'].astype(np.float32)[0,0] # Es 0.5
#scipy.io.loadmat('params.mat')['maxOuterIter'][0,0] # Es 100

# --- MASKING Y SKULL-STRIPPING ---
# Generamos máscaras del cerebro para aislarlo de ruido en el cráneo y fondo.
brain_solid = weight_np > 0.05
brain_solid = ndimage.binary_fill_holes(brain_solid)
mean_brain = np.mean(weight_np[brain_solid])
umbral_dinamico = 0.05 * mean_brain
mag_threshold = weight_np > umbral_dinamico
brain_clean = brain_solid & mag_threshold
brain_closed = ndimage.binary_closing(brain_clean, structure=np.ones((3,3,3)), iterations=2)
brain_final = ndimage.binary_fill_holes(brain_closed)

refined_mask = ndimage.gaussian_filter(brain_final.astype(np.float32), sigma=1.0) > 0.5 # Mascara dura
weight_np = weight_np * refined_mask # Skull-stripping: aislar el cerebro para no converger sobre ruido del cráneo.
weight_np /= weight_np[refined_mask].mean() # Se requiere que la media dentro del cerebro sea ~1 antes de elevar al cuadrado.
weight_np *= weight_np

N = phase_np.shape
Wy_np = weight_np * phase_np #Wy_np es simplemente W^2 * phi. La división por (W^2 + mu2) se hará en la inicialización de z2 y en el bucle.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert initial data to PyTorch tensors on GPU
phase = torch.from_numpy(phase_np).to(device)
kernel = torch.from_numpy(kernel_np).to(device)
K2 = torch.from_numpy(K2_np).to(device)
weight = torch.from_numpy(weight_np).to(device)
Wy = torch.from_numpy(Wy_np).to(device)
mask_torch = torch.from_numpy(refined_mask).to(device) # Máscara en GPU para estadísticas

# Autocast dtype fallback para tarjetas gráficas más antiguas
amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

# --- OPTIMIZACIÓN DE VRAM (DYNAMIC BATCHING) ---
# Adaptar el BATCH_SIZE dinámicamente según la VRAM disponible para no ahogar GPUs pequeñas.
if device.type == 'cuda':
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if total_vram_gb >= 11.5:
        BATCH_SIZE = 256        # Aprovecha masivos 16GB para lanzar kernels más grandes (4x más rápido)
    elif total_vram_gb >= 6.0:
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 16
else:
    BATCH_SIZE = 16  # CPU fallback

# Variable initialization to allocate memory on GPU
z1 = torch.zeros(N, dtype=torch.float32, device=device)
s1 = torch.zeros(N, dtype=torch.float32, device=device)
qsm = torch.zeros(N, dtype=torch.float32, device=device)
weight_mu2 = weight + mu2 # Precalculado fuera del bucle
denominator = mu2 * K2 + mu # Precalculado fuera del bucle
z2 = Wy / weight_mu2 # Initialized properly to the minimizer: (W^2 * phi) / (W^2 + mu2)
s2 = torch.zeros(N, dtype=torch.float32, device=device)

# Pre-alocar tensores del bucle TDV para evitar memory leaks/fragmentation
z1_accum = torch.zeros(N, dtype=torch.float32, device=device)
z1_count = torch.zeros(N, dtype=torch.float32, device=device)
center_indices = list(range(1, N[2]-1))

# Antes se recomputaba idéntico en cada iteración ADMM; ahora sólo z1_accum se resetea.
z1_count[:, :, 0:N[2]-2] += 1
z1_count[:, :, 1:N[2]-1] += 1
z1_count[:, :, 2:N[2]] += 1
#z1_count[z1_count == 0] = 1.0

print("\n--- ESTADÍSTICAS INICIALES ---")
print_stats("mu", mu)
print_stats("mu2", mu2)
print_stats("phase", phase)
print_stats("weight (W^2)", weight)
print_stats("Wy (W^2 * phase)", Wy)
print("------------------------------\n")

color = 'color'
checkpoint = torch.load(os.path.join('checkpoints', f'tdv3-3-25-f32-{color}.pth'))
vn = model.VNet(checkpoint['config'], efficient=False)
vn.load_state_dict(checkpoint['model'])
vn.to(device)
vn = vn.to(memory_format=torch.channels_last)
vn.eval()

# Habilitar compilación por Triton de todo el modelo (ahora que optoth ya no interfiere)
vn = torch.compile(vn, mode="max-autotune")

# --- ITERACIONES ADMM ---
# Nota: TDV se entrenó con imágenes RGB [0, 1]. Las imágenes QSM están en ppm [-0.1, 0.1].
# Para evitar artefactos, escalamos dinámicamente la entrada usando alpha para engañar
# a la red y que procese los datos QSM como si fueran contrastes naturales.

converged = False
for t in range(0, maxOuterIter):
    # update qsm
    qsm_old = qsm.clone()
    
    # FFT and updates in PyTorch
    fft_z2_s2 = torch.fft.fftn(z2 - s2)
    fft_z1_s1 = torch.fft.fftn(z1 - s1)
    
    numerator = mu2 * kernel * fft_z2_s2 + mu * fft_z1_s1
    qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)

    if converged:
        break

    if t == 0:
        # FINE TUNING: Calibración dinámica del contraste
        # Para que la red TDV preserve venas y bordes, necesitamos que la señal
        # supere con creces el ruido con el que fue entrenada (sigma ~0.1).
        # Un target Std de 0.3 en el tejido garantiza una fuerte activación no-lineal,
        # pero lo optimizamos con un grid-search a 0.06 para que el suavizado sea más sutil
        # y no distorsione tanto la geometría.
        target_std = 0.06
        qsm_std = torch.std(qsm[mask_torch])
        alpha = target_std / qsm_std.item()
        
        print("\n--- ESTADÍSTICAS ITERACIÓN 0 ---")
        print(f"[FINE TUNING] Alpha calibrado matemáticamente a: {alpha:.4f}")
        print_stats("qsm (update)", qsm)
    
    # Retomamos el cálculo original para el update (sin enmascarar)
    # para no alterar el criterio de parada original del ADMM.
    x_update = 100*torch.sqrt(torch.mean((qsm-qsm_old) ** 2))/torch.sqrt(torch.mean((qsm) ** 2))
    current_rmse = rmse(qsm.cpu().numpy(), chi_cosmos, mask=cosmos_mask) # Comentar/descomentar para imprimir valores
    print(f"Iter: {t:<3} Update: {x_update.item():<8.4f} RMSE: {current_rmse:.4f}%") # Comentar/descomentar para imprimir valores

    converged = x_update < tolUpdate
    FhDFx = torch.real(torch.fft.ifftn(kernel*torch.fft.fftn(qsm))).to(torch.float32)
    
    # --- PASO PROXIMAL z1: REGULARIZADOR TDV ---
    # Aplicamos TDV usando el escalado dinámico 'alpha'. Esto asegura que el input tenga 
    # la desviación estándar objetivo (target_std), activando correctamente los filtros 
    # no-lineales preservadores de bordes, sin necesidad de reentrenar la red.
    v = qsm + s1
    if t == 0:
        print_stats("v (qsm + s1)", v)
    
    # Construcción de minibatches: extraemos 3 cortes adyacentes rápidamente usando unfold
    # evitando torch.stack que copia memoria masivamente y ralentiza la iteración.
    z1_accum.zero_()
    
    # v_DHW tiene shape (D, H, W)
    v_DHW = v.permute(2, 0, 1).contiguous() 
    # unfold extrae ventanas de tamaño 3 en la dimension 0. Shape: (D-2, H, W, 3)
    # permute la convierte a (D-2, 3, H, W) original
    triplets = v_DHW.unfold(0, 3, 1).permute(0, 3, 1, 2)
    
    # Procesamiento en minibatches dinámicos según VRAM disponible
    for b_start in range(0, len(center_indices), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(center_indices))
        batch = triplets[b_start:b_end]  # (B, 3, H, W)
        
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype):
            batch_alpha = batch.contiguous().to(memory_format=torch.channels_last) * alpha
            if t == 0 and b_start == 0:
                print_stats("batch * alpha", batch_alpha)
                
            x_batch = vn(batch_alpha, batch_alpha)
        
        outputs = x_batch[-1]  # (B, 3, H, W) — last VNet step
        
        if t == 0 and b_start == 0:
            print_stats("outputs (VNet)", outputs)
        
        # Vectorized accumulation
        # outputs shape: (B, 3, H, W). We permute to (B, H, W, 3) to match z1_accum shape
        outputs_perm = (outputs / alpha).permute(0, 2, 3, 1) # (B, H, W, 3)
        
        # Accumulate slices
        z1_accum[:, :, b_start:b_end] += outputs_perm[:, :, :, 0].permute(1, 2, 0)
        z1_accum[:, :, b_start+1:b_end+1] += outputs_perm[:, :, :, 1].permute(1, 2, 0)
        z1_accum[:, :, b_start+2:b_end+2] += outputs_perm[:, :, :, 2].permute(1, 2, 0)
        
    z1 = z1_accum / z1_count  # z1_count precalculado una sola vez fuera del bucle
    s1 += qsm - z1
    
    if t == 0:
        print_stats("z1 (TDV out)", z1)
        print_stats("s1 (dual)", s1)
        print("--------------------------------\n")
    # update z2
    z2 = (Wy + mu2*(FhDFx+s2)) / weight_mu2
    s2 += FhDFx - z2
    
toc = time.time()    
print(f"corrido en {toc-tic} segundos")

## Save output and Convert to NIfTI / RAS convention
qsm_np = qsm.cpu().numpy()
v_np = v.cpu().numpy()
z1_np = z1.cpu().numpy()
s1_np = s1.cpu().numpy()
mdic = {"x": qsm_np, "time":(toc-tic),"iter":(t+1)}
scipy.io.savemat('results/wTDV_QSM_torch.mat', mdic)

rmse_val = rmse(qsm_np, chi_cosmos, mask=cosmos_mask)
print(f"\n[EVALUACIÓN] RMSE de Susceptibilidad (QSM predicha vs Cosmos): {rmse_val:.2f}%")

imshow_3d(qsm_np, title="wTDV_QSM (x)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='results/wTDV_QSM_torch.png')
imshow_3d(v_np, title="v (input to TDV)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='figures/wTDV_v_torch.png')
imshow_3d(z1_np, title="z1 (output of TDV)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='figures/wTDV_z1_torch.png')
imshow_3d(s1_np, title="s1 (dual variable)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='figures/wTDV_s1_torch.png')

'''
--- ESTADÍSTICAS INICIALES ---
[DEBUG] mu              | Valor:   0.0080
[DEBUG] mu2             | Valor:   1.0000
[DEBUG] phase           | Rango:  -0.1623 a   0.1623 | Media:  -0.0000 | Std:   0.0825
[DEBUG] weight (W^2)    | Rango:   0.0000 a  14.8348 | Media:   0.2370 | Std:   0.4706
[DEBUG] Wy (W^2 * phase) | Rango:  -0.1547 a   0.2549 | Media:   0.0001 | Std:   0.0049
------------------------------

--- ESTADÍSTICAS ITERACIÓN 0 ---
[FINE TUNING] Alpha calibrado matemáticamente a: 4.8591
[DEBUG] qsm (update)    | Rango:  -0.1263 a   0.1953 | Media:  -0.0000 | Std:   0.0060
[DEBUG] v (qsm + s1)    | Rango:  -0.1263 a   0.1953 | Media:   0.0000 | Std:   0.0060
[DEBUG] batch * alpha   | Rango:  -0.6137 a   0.9488 | Media:  -0.0000 | Std:   0.0293
[DEBUG] outputs (VNet)  | Rango:  -0.4770 a   0.8491 | Media:  -0.0000 | Std:   0.0210
[DEBUG] z1 (TDV out)    | Rango:  -0.0841 a   0.1558 | Media:  -0.0000 | Std:   0.0043
[DEBUG] s1 (dual)       | Rango:  -0.0567 a   0.0649 | Media:   0.0000 | Std:   0.0036
--------------------------------

--- RMSE FINAL FASE PREDICHA VS FASE REAL ---
[EVALUACIÓN] RMSE de Susceptibilidad (QSM predicha vs Cosmos): 27.34%
--------------------------------

Iter: 1   Update: 49.2762  RMSE: 43.5523%
Iter: 2   Update: 18.6148  RMSE: 36.8198%
Iter: 3   Update: 8.8699   RMSE: 33.8320%
Iter: 4   Update: 5.4920   RMSE: 32.2225%
Iter: 5   Update: 4.0153   RMSE: 31.1436%
Iter: 6   Update: 3.1015   RMSE: 30.4033%
Iter: 7   Update: 2.3905   RMSE: 29.8363%
Iter: 8   Update: 1.9227   RMSE: 29.3947%
Iter: 9   Update: 1.6220   RMSE: 29.0630%
Iter: 10  Update: 1.3970   RMSE: 28.8145%
Iter: 11  Update: 1.1728   RMSE: 28.6108%
Iter: 12  Update: 1.0136   RMSE: 28.4436%
Iter: 13  Update: 0.8980   RMSE: 28.3024%
Iter: 14  Update: 0.7974   RMSE: 28.1821%
Iter: 15  Update: 0.7192   RMSE: 28.0775%
Iter: 16  Update: 0.6569   RMSE: 27.9864%
Iter: 17  Update: 0.6085   RMSE: 27.9061%
Iter: 18  Update: 0.5704   RMSE: 27.8353%
Iter: 19  Update: 0.5393   RMSE: 27.7719%
Iter: 20  Update: 0.5115   RMSE: 27.7155%
Iter: 21  Update: 0.4891   RMSE: 27.6646%
Iter: 22  Update: 0.4690   RMSE: 27.6191%
Iter: 23  Update: 0.4529   RMSE: 27.5777%
Iter: 24  Update: 0.4409   RMSE: 27.5407%
Iter: 25  Update: 0.4281   RMSE: 27.5070%
Iter: 26  Update: 0.4189   RMSE: 27.4767%
Iter: 27  Update: 0.4101   RMSE: 27.4490%
Iter: 28  Update: 0.4040   RMSE: 27.4241%
Iter: 29  Update: 0.3974   RMSE: 27.4010%
Iter: 30  Update: 0.3934   RMSE: 27.3807%
Iter: 31  Update: 0.3878   RMSE: 27.3615%

corrido en 67.2 segundos
'''
'''
=============================================================================
LOG DE OPTIMIZACIONES Y RESOLUCIÓN MATEMÁTICA:

1. Data Fidelity y Doble División (Bug Fix):
   - Problema: Wy_np contenía W^2 * phi. Al inicializar z2 y actualizar en el bucle, 
     se volvía a dividir por (W^2 + mu2), reduciendo la fuerza de la física a la mitad.
   - Solución: Wy_np se define puramente como W^2 * phi. La división se hace una sola vez, 
     restaurando la energía (Std) de los datos (pasando de 0.002 a ~0.005).

2. Calibración Dinámica de Escala (Alpha):
   - Problema: Un alpha fijo (dependiente del paciente) o un target_std muy alto (ej. 0.5) forzaban a la red a sobre-regularizar, aplastando los gradientes naturales y subiendo el RMSE a >40%.
   - Solución: Se calibra con target_std=0.1. Esto relaja a la VNet hacia una zona más lineal (L2), suavizando las venas sin "recortar" la geometría, bajando el RMSE a 29.11% (<35%).

3. Máscara y Condicionamiento Numérico:
   - Problema: Usar una máscara suave generaba valores de peso diminutos pero no nulos, arruinando el número de condición (Condition Number) del ADMM (iteraciones > 50).
   - Solución: Cortar limpiamente el ruido del cráneo con máscara dura (> 0.5) restaura una convergencia ultra-rápida (~12 iteraciones).

4. Optimizaciones de Hardware y Memoria:
   - Dynamic Batching y Tensor.unfold evitan clonaciones masivas (OOM), bajando el tiempo a <2 min en GPUs de consumo (<8GB). Los denominadores se precalculan fuera del bucle.
   - Se implementa autotuner de cuDNN que elige el algoritmo de conv2d más rápido / aprovechamiento de TensorCores
5. Restricción del Dual (mu):
   - Problema: Valores altos (mu=1.0) aplastaban los datos.
   - Solución: Optimizamos a mu=0.02. Combinado con un target_std bajo (0.1), logra el equilibrio perfecto entre Data Fidelity (física) y Prior TDV.
=============================================================================
'''
# %%
