#%%
import argparse
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

# --checkpoint elige el modelo TDV, de 3 canales: el volumen entra como tripletes de cortes adyacentes (color)
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='tdv3-3-25-f32-color.pth', help='.pth de checkpoints/ que esté en PARAMS')
args = parser.parse_known_args()[0]  # parse_known_args para no chocar con los argumentos de Jupyter

# Parámetros asociados a cada checkpoint, de grid_search_wtdv.py; tolUpdate es sobre el Update dentro del cerebro
PARAMS = {
    'tdv3-3-25-f32-color.pth': {'mu': 0.008, 'target_std': 0.06, 'tolUpdate': 0.4},
    't1p1delta-rapido-100k.pth': {'mu': 0.006, 'target_std': 0.06, 'tolUpdate': 0.3},
    't2delta-100k-s1.pth': {'mu': 0.004, 'target_std': 0.06, 'tolUpdate': 1.0},
    't8delta-100k-s1.pth': {'mu': 0.006, 'target_std': 0.07, 'tolUpdate': 0.75},
    'ciego2d-35k.pth': {'mu': 0.006, 'target_std': 0.04, 'tolUpdate': 1.5},
}
if args.checkpoint not in PARAMS:
    raise SystemExit(f"{args.checkpoint} no está en PARAMS: buscarle mu, target_std y tolUpdate con grid_search_wtdv.py")
PARAMS = {'checkpoint': args.checkpoint, **PARAMS[args.checkpoint]}
checkpoint = torch.load(os.path.join('checkpoints', args.checkpoint))
C = 3  # tripletes de cortes adyacentes como R, G, B

# La salida va a results/checkpoints/<checkpoint>/; las imágenes de results/ y figures/ son las originales del repo
SALIDA = os.path.join('results', 'checkpoints', args.checkpoint.removesuffix('.pth'))

tic = time.time()

#Load MATLAB data container
phase_np = scipy.io.loadmat('data_mat/params.mat')['phase_use'].astype(np.float32)
kernel_np = scipy.io.loadmat('data_mat/params.mat')['kernel'].astype(np.float32)
K2_np = np.conj(kernel_np)*kernel_np # El kernel en params.mat YA tiene su centro en [0,0,0] / no aplicamos ifftshift
weight_np = scipy.io.loadmat('data_mat/params.mat')['magn_use'].astype(np.float32)
maxOuterIter = 100
tolUpdate = PARAMS['tolUpdate']
chi_cosmos = scipy.io.loadmat('data_mat/chi_cosmos.mat')['chi_cosmos'].astype(np.float32)
cosmos_mask = chi_cosmos != 0
mu = PARAMS['mu']  # Optimizacion via grid search: 0.008 para 3 canales
mu2 = 1.0
#scipy.io.loadmat('data_mat/params.mat')['alpha1'].astype(np.float32)[0,0] # Es 0.04 / el valor de referencia de Carlos es 0.0245
#scipy.io.loadmat('data_mat/params.mat')['mu1'].astype(np.float32)[0,0] # Es 1.0
#scipy.io.loadmat('data_mat/params.mat')['tol_update'].astype(np.float32)[0,0] # Es 0.5
#scipy.io.loadmat('data_mat/params.mat')['maxOuterIter'][0,0] # Es 100

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
mask_f = mask_torch.float()  # para enmascarar z1 en cada iteración

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
num_windows = N[2] - C + 1  # ventanas de C cortes adyacentes en el eje z

# Antes se recomputaba idéntico en cada iteración ADMM; ahora sólo z1_accum se resetea.
# Cuántas ventanas cubren cada corte: 1, 2, 3, ..., 3, 2, 1
for c in range(C):
    z1_count[:, :, c:num_windows+c] += 1
#z1_count[z1_count == 0] = 1.0

print("\n--- ESTADÍSTICAS INICIALES ---")
print(f"Modelo de {C} canal(es): {PARAMS['checkpoint']}")
print_stats("mu", mu)
print_stats("mu2", mu2)
print_stats("phase", phase)
print_stats("weight (W^2)", weight)
print_stats("Wy (W^2 * phase)", Wy)
print("------------------------------\n")

vn = model.VNet(checkpoint['config'], efficient=False)
vn.load_state_dict(checkpoint['model'])
vn.to(device)
vn = vn.to(memory_format=torch.channels_last)
vn.eval()

# Habilitar compilación por Triton de todo el modelo (ahora que optoth ya no interfiere)
vn = torch.compile(vn, mode="max-autotune")

# --- ITERACIONES ADMM ---
# Nota: TDV se entrenó con imágenes RGB en [0, 1]. Las imágenes QSM están en ppm [-0.1, 0.1].
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
        # pero lo optimizamos con un grid-search (0.06 con 3 canales) para que el suavizado
        # sea más sutil y no distorsione tanto la geometría.
        target_std = PARAMS['target_std']
        qsm_std = torch.std(qsm[mask_torch])
        alpha = target_std / qsm_std.item()
        
        print("\n--- ESTADÍSTICAS ITERACIÓN 0 ---")
        print(f"[FINE TUNING] Alpha calibrado matemáticamente a: {alpha:.4f}")
        print_stats("qsm (update)", qsm)
    
    # Update sólo dentro del cerebro, donde la QSM se usa y se evalúa: afuera no hay datos (W^2 = 0) y el fondo
    # nunca termina de asentarse. Es el update de FANSI sobre mask.*x, como en su WH-QSM.
    x_update = 100*torch.norm((qsm-qsm_old)[mask_torch])/torch.norm(qsm[mask_torch])
    current_rmse = rmse(qsm.cpu().numpy(), chi_cosmos, mask=cosmos_mask) # Comentar/descomentar para imprimir valores
    print(f"Iter: {t:<3} Update: {x_update.item():<8.4f} RMSE: {current_rmse:.4f}%") # Comentar/descomentar para imprimir valores

    converged = x_update < tolUpdate
    FhDFx = torch.real(torch.fft.ifftn(kernel*torch.fft.fftn(qsm))).to(torch.float32)
    
    # --- PASO PROXIMAL z1: REGULARIZADOR TDV ---
    # Aplicamos TDV usando el escalado dinámico 'alpha'. Esto asegura que el input tenga 
    # la desviación estándar objetivo (target_std), activando correctamente los filtros 
    # no-lineales preservadores de bordes, sin necesidad de reentrenar la red.
    v = (qsm + s1) * mask_f  # la entrada del TDV también: con z1 = 0 afuera, el prox no depende de v afuera
    if t == 0:
        print_stats("v (qsm + s1)", v)
    
    # Construcción de minibatches: extraemos C cortes adyacentes rápidamente usando unfold
    # evitando torch.stack que copia memoria masivamente y ralentiza la iteración.
    z1_accum.zero_()

    # v_DHW tiene shape (D, H, W)
    v_DHW = v.permute(2, 0, 1).contiguous()
    # unfold extrae ventanas de tamaño C en la dimension 0. Shape: (D-C+1, H, W, C)
    # permute la convierte a (D-C+1, C, H, W)
    windows = v_DHW.unfold(0, C, 1).permute(0, 3, 1, 2)

    # Procesamiento en minibatches dinámicos según VRAM disponible
    for b_start in range(0, num_windows, BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, num_windows)
        batch = windows[b_start:b_end]  # (B, C, H, W)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype):
            batch_alpha = batch.contiguous().to(memory_format=torch.channels_last) * alpha
            if t == 0 and b_start == 0:
                print_stats("batch * alpha", batch_alpha)
                
            x_batch = vn(batch_alpha, batch_alpha)
        
        outputs = x_batch[-1]  # (B, C, H, W) — last VNet step

        if t == 0 and b_start == 0:
            print_stats("outputs (VNet)", outputs)

        # Vectorized accumulation
        # outputs shape: (B, C, H, W). We permute to (B, H, W, C) to match z1_accum shape
        outputs_perm = (outputs / alpha).permute(0, 2, 3, 1) # (B, H, W, C)

        # Accumulate slices: el canal c de la ventana b es el corte b+c
        for c in range(C):
            z1_accum[:, :, b_start+c:b_end+c] += outputs_perm[:, :, :, c].permute(1, 2, 0)

    z1 = z1_accum / z1_count  # z1_count precalculado una sola vez fuera del bucle
    z1 = z1 * mask_f  # fuera del cerebro el prior es 0: ahí no hay datos y el TDV amplifica (bloque final, item 6)
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
qsm_np = (qsm * mask_f).cpu().numpy()  # mapa final por la máscara (consenso QSM, MRM 2024)
v_np = v.cpu().numpy()
z1_np = z1.cpu().numpy()
s1_np = s1.cpu().numpy()
mdic = {"x": qsm_np, "time":(toc-tic),"iter":(t+1)}
rmse_val = rmse(qsm_np, chi_cosmos, mask=cosmos_mask)
print(f"\n[EVALUACIÓN] RMSE de Susceptibilidad (QSM predicha vs Cosmos): {rmse_val:.2f}%")

os.makedirs(SALIDA, exist_ok=True)
titulo = f"wTDV_QSM (x) · {os.path.basename(SALIDA)}\nμ={mu}, std={PARAMS['target_std']} · {t+1} iter · RMSE {rmse_val:.2f}%"
scipy.io.savemat(os.path.join(SALIDA, 'wTDV_QSM_torch.mat'), mdic)

imshow_3d(qsm_np, title=titulo, rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath=os.path.join(SALIDA, 'wTDV_QSM_torch.png'))
imshow_3d(v_np, title="v (input to TDV)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath=os.path.join(SALIDA, 'wTDV_v_torch.png'))
imshow_3d(z1_np, title="z1 (output of TDV)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath=os.path.join(SALIDA, 'wTDV_z1_torch.png'))
imshow_3d(s1_np, title="s1 (dual variable)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath=os.path.join(SALIDA, 'wTDV_s1_torch.png'))

'''
--- ESTADÍSTICAS INICIALES ---
Modelo de 3 canal(es): tdv3-3-25-f32-color.pth
[DEBUG] mu              | Valor:   0.0080
[DEBUG] mu2             | Valor:   1.0000
[DEBUG] phase           | Rango:  -0.1623 a   0.1623 | Media:  -0.0000 | Std:   0.0825
[DEBUG] weight (W^2)    | Rango:   0.0000 a  14.8348 | Media:   0.2370 | Std:   0.4706
[DEBUG] Wy (W^2 * phase) | Rango:  -0.1547 a   0.2549 | Media:   0.0001 | Std:   0.0049
------------------------------


--- ESTADÍSTICAS ITERACIÓN 0 ---
[FINE TUNING] Alpha calibrado matemáticamente a: 4.8591
[DEBUG] qsm (update)    | Rango:  -0.1263 a   0.1953 | Media:  -0.0000 | Std:   0.0060
Iter: 0   Update: 100.0000 RMSE: 65.8201%
[DEBUG] v (qsm + s1)    | Rango:  -0.1263 a   0.1953 | Media:   0.0000 | Std:   0.0060
[DEBUG] batch * alpha   | Rango:  -0.6137 a   0.9488 | Media:  -0.0000 | Std:   0.0293
[DEBUG] outputs (VNet)  | Rango:  -0.4776 a   0.8494 | Media:  -0.0000 | Std:   0.0210
[DEBUG] z1 (TDV out)    | Rango:  -0.0841 a   0.1558 | Media:   0.0000 | Std:   0.0042
[DEBUG] s1 (dual)       | Rango:  -0.0567 a   0.0649 | Media:  -0.0000 | Std:   0.0037
--------------------------------

Iter: 1   Update: 49.5766  RMSE: 43.1759%
Iter: 2   Update: 17.6966  RMSE: 36.3167%
Iter: 3   Update: 8.1999   RMSE: 33.2949%
Iter: 4   Update: 4.9554   RMSE: 31.6339%
Iter: 5   Update: 3.6879   RMSE: 30.4854%
Iter: 6   Update: 2.9200   RMSE: 29.6648%
Iter: 7   Update: 2.2939   RMSE: 29.0199%
Iter: 8   Update: 1.8836   RMSE: 28.5080%
Iter: 9   Update: 1.6183   RMSE: 28.1158%
Iter: 10  Update: 1.4211   RMSE: 27.8172%
Iter: 11  Update: 1.2114   RMSE: 27.5728%
Iter: 12  Update: 1.0627   RMSE: 27.3755%
Iter: 13  Update: 0.9421   RMSE: 27.2144%
Iter: 14  Update: 0.8327   RMSE: 27.0840%
Iter: 15  Update: 0.7380   RMSE: 26.9785%
Iter: 16  Update: 0.6588   RMSE: 26.8935%
Iter: 17  Update: 0.5919   RMSE: 26.8248%
Iter: 18  Update: 0.5394   RMSE: 26.7686%
Iter: 19  Update: 0.4995   RMSE: 26.7225%
Iter: 20  Update: 0.4657   RMSE: 26.6830%
Iter: 21  Update: 0.4407   RMSE: 26.6498%
Iter: 22  Update: 0.4128   RMSE: 26.6209%
Iter: 23  Update: 0.3920   RMSE: 26.5959%
corrido en 52.4 segundos

[EVALUACIÓN] RMSE de Susceptibilidad (QSM predicha vs Cosmos): 26.57%
Saved figure to: results/checkpoints/tdv3-3-25-f32-color/wTDV_QSM_torch.png
Saved figure to: results/checkpoints/tdv3-3-25-f32-color/wTDV_v_torch.png
Saved figure to: results/checkpoints/tdv3-3-25-f32-color/wTDV_z1_torch.png
Saved figure to: results/checkpoints/tdv3-3-25-f32-color/wTDV_s1_torch.png
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

6. Máscara en la salida del TDV (z1):
   - Problema: fuera del cerebro (77% del volumen) no hay datos (W^2 = 0) y el TDV de los checkpoints de Training-TDV
     amplifica ahí las perturbaciones: el fondo queda texturado y el ADMM no se asienta.
   - Solución: z1 = z1 * mask_f en cada iteración, como LPCNN y MoDL-QSM con la salida de su red. Oficial 27.38% ->
     26.57% (31 -> 27 iteraciones), 2δ 28.82% -> 28.18%, 8δ 28.37% -> 27.83%. Proyectar también qsm dio peor.
=============================================================================
'''