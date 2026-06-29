#%%
import os
import time
import numpy as np
import torch
import scipy.io
from scipy import ndimage

import model
from utils import imshow_3d


tic = time.time()
#Load MATLAB data container
phase_np = scipy.io.loadmat('params.mat')['phase_use'].astype(np.float32)
kernel_np = scipy.io.loadmat('params.mat')['kernel'].astype(np.float32)
K2_np = np.conj(kernel_np)*kernel_np

weight_np = scipy.io.loadmat('params.mat')['magn_use'].astype(np.float32)

# --- MASKING MORFOLÓGICO ---
# Generamos un contorno suave para evitar artefactos de estrella en la inversión QSM.
brain_solid = weight_np > 0.05
brain_solid = ndimage.binary_fill_holes(brain_solid)

mean_brain = np.mean(weight_np[brain_solid])
umbral_dinamico = 0.05 * mean_brain
mag_threshold = weight_np > umbral_dinamico
brain_clean = brain_solid & mag_threshold

brain_closed = ndimage.binary_closing(brain_clean, structure=np.ones((3,3,3)), iterations=2)
brain_final = ndimage.binary_fill_holes(brain_closed)
refined_mask = ndimage.gaussian_filter(brain_final.astype(np.float32), sigma=1.0) > 0.5

# Aplicar la máscara suave para limpiar el fondo
weight_np = weight_np * refined_mask

# --- NORMALIZACIÓN DE MAGNITUD ---
# Se requiere que la media dentro del cerebro sea ~1 antes de elevar al cuadrado (requerimiento FANSI).
weight_np /= weight_np[refined_mask].mean()
weight_np *= weight_np

# --- PARÁMETROS DE REGULARIZACIÓN ---
# Usamos un alpha empírico. En el pasado, ajustábamos alpha a ~0.04 para forzar 
# a la red TDV a un régimen lineal, intentando mitigar el sobre-suavizado y los artefactos.
# Concluimos que esto era insuficiente: el modelo TDV pre-entrenado con imágenes RGB 
# naturales falla fundamentalmente con datos QSM, requiriendo ser reentrenado.
alpha = 0.2903
mu = 0.0245

maxOuterIter = scipy.io.loadmat('params.mat')['maxOuterIter'][0,0]
tolUpdate = scipy.io.loadmat('params.mat')['tol_update'].astype(np.float32)[0,0]

mu2 = 1.0

N = phase_np.shape
Wy_np = weight_np*phase_np/(weight_np+mu2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert initial data to PyTorch tensors on GPU
phase = torch.from_numpy(phase_np).to(device)
kernel = torch.from_numpy(kernel_np).to(device)
K2 = torch.from_numpy(K2_np).to(device)
weight = torch.from_numpy(weight_np).to(device)
Wy = torch.from_numpy(Wy_np).to(device)

## Variable initialization to allocate memory on GPU
z1 = torch.zeros(N, dtype=torch.float32, device=device)
s1 = torch.zeros(N, dtype=torch.float32, device=device)
qsm = torch.zeros(N, dtype=torch.float32, device=device)
z2 = Wy.clone() # Initialized with weighted data
s2 = torch.zeros(N, dtype=torch.float32, device=device)

color = 'color'
checkpoint = torch.load(os.path.join('checkpoints', f'tdv3-3-25-f32-{color}.pth'))
vn = model.VNet(checkpoint['config'], efficient=False)
vn.load_state_dict(checkpoint['model'])
vn.to(device)
vn.eval()

# --- ITERACIONES ADMM ---
# Nota histórica: TDV se entrenó con imágenes naturales en rango [0, 1].
# Las imágenes QSM tienen rango ~[-0.1, 0.1]. Intentamos varias estrategias de 
# normalización y escalado, pero la discrepancia de dominios causó artefactos severos.

for t in range(0, maxOuterIter):
    # update qsm
    qsm_old = qsm.clone()
    
    # FFT and updates in PyTorch
    fft_z2_s2 = torch.fft.fftn(z2 - s2)
    fft_z1_s1 = torch.fft.fftn(z1 - s1)
    
    numerator = mu2 * torch.conj(kernel) * fft_z2_s2 + mu * fft_z1_s1
    denominator = mu2 * K2 + mu
    qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)
    
    x_update = 100*torch.sqrt(torch.mean((qsm-qsm_old) ** 2))/torch.sqrt(torch.mean((qsm) ** 2))
    print('Iter: '+str(t)+'   Update: '+str(x_update.item()))
    
    if x_update < tolUpdate:
        break
    FhDFx = torch.real(torch.fft.ifftn(kernel*torch.fft.fftn(qsm))).to(torch.float32)
    
    # --- PASO PROXIMAL z1: REGULARIZADOR TDV ---
    # Aplicamos TDV usando un escalado 'alpha'. Como se mencionó, intentamos usar alphas
    # muy pequeños para mantener las activaciones en un régimen lineal (comportamiento tipo Tikhonov).
    # Eventualmente concluimos que este truco no sustituye un reentrenamiento real del modelo.
    v = qsm + s1
    
    # Construcción de minibatches: agrupamos 3 cortes adyacentes para simular 
    # canales RGB (3, H, W) requeridos por la arquitectura original de TDV.
    z1_accum = torch.zeros(N, dtype=torch.float32, device=device)
    z1_count = torch.zeros(N, dtype=torch.float32, device=device)
    
    center_indices = list(range(1, N[2]-1))  # stride 1 covers all slices safely
    triplets = torch.stack([
        v[:, :, i-1:i+2].permute(2, 0, 1)  # (3, H, W) in original scale
        for i in center_indices
    ])  # (num_triplets, 3, H, W)
    
    # Procesamiento en minibatches para evitar Out Of Memory en la GPU.
    BATCH_SIZE = 158
    
    for b_start in range(0, len(center_indices), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(center_indices))
        batch = triplets[b_start:b_end]  # (B, 3, H, W)
        
        with torch.no_grad():
            x_batch = vn(batch.contiguous() * alpha, batch.contiguous() * alpha)
        
        outputs = x_batch[-1]  # (B, 3, H, W) — last VNet step
        
        for j, i in enumerate(center_indices[b_start:b_end]):
            x_S_scaled = outputs[j].permute(1, 2, 0)  # (H, W, 3)
            x_S = x_S_scaled / alpha
            
            z1_accum[:, :, i-1:i+2] += x_S
            z1_count[:, :, i-1:i+2] += 1
            
    z1_count[z1_count == 0] = 1.0
    z1 = z1_accum / z1_count
    
    s1 += qsm - z1
    # update z2
    z2 = Wy + mu2*(FhDFx+s2)/(weight + mu2)
    s2 += FhDFx - z2
          
toc = time.time()    
print(f"corrido en {toc-tic} segundos")

## Save output    
mdic = {"x": qsm.cpu().numpy(), "time":(toc-tic),"iter":(t+1)}
scipy.io.savemat('result_wTDV_torch.mat', mdic)
imshow_3d(qsm.cpu().numpy(), title="wTDV_QSM (x)", rango=(-0.1, 0.1), savepath='figures/wTDV_QSM_torch.png')
imshow_3d(v.cpu().numpy(), title="v (input to TDV)", rango=(-0.1, 0.1), savepath='figures/wTDV_v_torch.png')
imshow_3d(z1.cpu().numpy(), title="z1 (output of TDV)", rango=(-0.1, 0.1), savepath='figures/wTDV_z1_torch.png')
imshow_3d(s1.cpu().numpy(), title="s1 (dual variable)", rango=(-0.1, 0.1), savepath='figures/wTDV_s1_torch.png')

# %%
