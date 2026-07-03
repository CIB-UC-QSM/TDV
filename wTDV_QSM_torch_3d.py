#%%
#Proof of concept hecho con IA replicando el 2D desarrollado.
import os
import time
import numpy as np
import torch
import scipy.io
from scipy import ndimage

from ddr import model
from ddr.utils import imshow_3d, print_stats, rmse, process_axis

tic = time.time()

#Load MATLAB data container
phase_np = scipy.io.loadmat('params.mat')['phase_use'].astype(np.float32)
kernel_np = scipy.io.loadmat('params.mat')['kernel'].astype(np.float32)
K2_np = np.conj(kernel_np)*kernel_np # El kernel en params.mat YA tiene su centro en [0,0,0] / no aplicamos ifftshift
weight_np = scipy.io.loadmat('params.mat')['magn_use'].astype(np.float32)
maxOuterIter = scipy.io.loadmat('params.mat')['maxOuterIter'][0,0]
tolUpdate = scipy.io.loadmat('params.mat')['tol_update'].astype(np.float32)[0,0]
chi_cosmos = scipy.io.loadmat('chi_cosmos.mat')['chi_cosmos'].astype(np.float32)
cosmos_mask = chi_cosmos != 0
mu = 0.02  # Optimizacion via grid search resulta en 0.02
mu2 = 1.0
#scipy.io.loadmat('params.mat')['alpha1'].astype(np.float32)[0,0] # Es 0.04
#scipy.io.loadmat('params.mat')['mu1'].astype(np.float32)[0,0] # Es 1.0

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
weight_np /= weight_np[refined_mask].mean() # Se requiere que la media dentro del cerebro sea ~1 antes de elevating al cuadrado.
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
mask_torch = torch.from_numpy(refined_mask).to(device)

# --- OPTIMIZACIÓN DE VRAM (DYNAMIC BATCHING) ---
if device.type == 'cuda':
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if total_vram_gb >= 11.5:
        BATCH_SIZE = 128        # Aprovecha masivos 16GB para lanzar kernels más grandes (4x más rápido)
    elif total_vram_gb >= 6.0:
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 16
else:
    BATCH_SIZE = 16

# Si procesamos por las dimensiones X e Y, N[0] y N[1] pueden ser más grandes que N[2].
# Aseguramos que el batch size no reviente la RAM. BATCH_SIZE es sobre el número de slices.
# Podemos usar un max_batch_size en base a la VRAM.
max_batch_size = BATCH_SIZE

# Variable initialization to allocate memory on GPU
zx = torch.zeros(N, dtype=torch.float32, device=device)
sx = torch.zeros(N, dtype=torch.float32, device=device)
zy = torch.zeros(N, dtype=torch.float32, device=device)
sy = torch.zeros(N, dtype=torch.float32, device=device)
zz = torch.zeros(N, dtype=torch.float32, device=device)
sz = torch.zeros(N, dtype=torch.float32, device=device)

qsm = torch.zeros(N, dtype=torch.float32, device=device)
weight_mu2 = weight + mu2
denominator = mu2 * K2 + mu
z2 = Wy / weight_mu2
s2 = torch.zeros(N, dtype=torch.float32, device=device)

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
vn.eval()

# --- ITERACIONES ADMM ---
# Nota: TDV se entrenó con imágenes RGB [0, 1]. Las imágenes QSM están en ppm [-0.1, 0.1].
# Para evitar artefactos, escalamos dinámicamente la entrada usando alpha para engañar
# a la red y que procese los datos QSM como si fueran contrastes naturales.

for t in range(0, maxOuterIter):
    qsm_old = qsm.clone()
    
    fft_z2_s2 = torch.fft.fftn(z2 - s2)
    fft_zx_sx = torch.fft.fftn(zx - sx)
    fft_zy_sy = torch.fft.fftn(zy - sy)
    fft_zz_sz = torch.fft.fftn(zz - sz)
    
    numerator = mu2 * kernel * fft_z2_s2 + (mu / 3.0) * (fft_zx_sx + fft_zy_sy + fft_zz_sz)
    qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)
    
    if t == 0:
        target_std = 0.1
        qsm_std = torch.std(qsm[mask_torch])
        alpha = target_std / qsm_std.item() if qsm_std.item() > 0 else 1.0
    
    # Retomamos el cálculo original para el update (sin enmascarar)
    # para no alterar el criterio de parada original del ADMM.
    x_update = 100*torch.sqrt(torch.mean((qsm-qsm_old) ** 2))/torch.sqrt(torch.mean((qsm) ** 2))
    #print('Iter: '+str(t)+'   Update: '+str(x_update.item())) # Descomentar para ver el print de las iteraciones
    
    if x_update < tolUpdate:
        break
    FhDFx = torch.real(torch.fft.ifftn(kernel*torch.fft.fftn(qsm))).to(torch.float32)
    
    # --- PASO PROXIMAL: REGULARIZADOR TDV MULTI-VARIABLE ---
    vx = qsm + sx
    vy = qsm + sy
    vz = qsm + sz
    
    # Axial (Z-axis, permute: 2, 0, 1)
    zz = process_axis(vz, (2, 0, 1), vn, alpha, max_batch_size)
    
    # Coronal (Y-axis, permute: 1, 0, 2)
    zy = process_axis(vy, (1, 0, 2), vn, alpha, max_batch_size)
    
    # Sagital (X-axis, permute: 0, 1, 2)
    zx = process_axis(vx, (0, 1, 2), vn, alpha, max_batch_size)
    
    sx += qsm - zx
    sy += qsm - zy
    sz += qsm - zz
    
    if t == 0:
        print(f"\n--- ESTADÍSTICAS ITERACIÓN 0 ---")
        print(f"[FINE TUNING] Alpha calibrado matemáticamente a: {alpha:.4f}")
        print_stats("qsm (update)", qsm)
        print_stats("zz (TDV out Z)", zz)
        print_stats("sz (dual Z)", sz)
        print("--------------------------------\n")
        
    # update z2
    z2 = (Wy + mu2*(FhDFx+s2)) / weight_mu2
    s2 += FhDFx - z2
    
    # Progress Print
    if (t + 1) % 5 == 0 or t == 0:
        toc_iter = time.time()
        qsm_iter_np = qsm.cpu().numpy()
        rmse_iter = rmse(qsm_iter_np, chi_cosmos, mask=cosmos_mask)
        print(f"Iteración {t+1:02d}/50 | Tiempo transcurrido: {toc_iter-tic:.1f}s | RMSE: {rmse_iter:.2f}%")

          
toc = time.time()    
print(f"corrido en {toc-tic} segundos")

## Save output and Convert to NIfTI / RAS convention
qsm_np = qsm.cpu().numpy()
vz_np = vz.cpu().numpy()
zz_np = zz.cpu().numpy()
sz_np = sz.cpu().numpy()

mdic = {"x": qsm_np, "time":(toc-tic),"iter":(t+1)}
scipy.io.savemat('results/wTDV_QSM_torch_3d.mat', mdic)

# --- EVALUACIÓN RMSE (Susceptibilidad) ---
rmse_val = rmse(qsm_np, chi_cosmos, mask=cosmos_mask)
print(f"\n--- RMSE FINAL SUSCEPTIBILIDAD ---")
print(f"[EVALUACIÓN] RMSE de Susceptibilidad (QSM predicha vs Cosmos): {rmse_val:.2f}%")
print(f"--------------------------------")

imshow_3d(qsm_np, title="wTDV_QSM_3D (x)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='results/wTDV_QSM_torch_3d.png')
imshow_3d(vz_np, title="vz (input to TDV Z)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='figures/wTDV_vz_torch_3d.png')
imshow_3d(zz_np, title="zz (output of TDV Z)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='figures/wTDV_zz_torch_3d.png')
imshow_3d(sz_np, title="sz (dual variable Z)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='figures/wTDV_sz_torch_3d.png')

'''
--- ESTADÍSTICAS INICIALES ---
[DEBUG] mu              | Valor:   0.0200
[DEBUG] mu2             | Valor:   1.0000
[DEBUG] phase           | Rango:  -0.1623 a   0.1623 | Media:  -0.0000 | Std:   0.0825
[DEBUG] weight (W^2)    | Rango:   0.0000 a  14.8348 | Media:   0.2370 | Std:   0.4706
[DEBUG] Wy (W^2 * phase) | Rango:  -0.1547 a   0.2549 | Media:   0.0001 | Std:   0.0049
------------------------------

--- ESTADÍSTICAS ITERACIÓN 0 ---
[FINE TUNING] Alpha calibrado matemáticamente a: 9.5334
[DEBUG] qsm (update)    | Rango:  -0.1023 a   0.1601 | Media:   0.0000 | Std:   0.0051
[DEBUG] v (qsm + s1)    | Rango:  -0.1023 a   0.1601 | Media:   0.0000 | Std:   0.0051
[DEBUG] z1 (TDV out 3D) | Rango:  -0.0683 a   0.1323 | Media:   0.0000 | Std:   0.0041
[DEBUG] s1 (dual)       | Rango:  -0.0385 a   0.0375 | Media:  -0.0000 | Std:   0.0024
--------------------------------

--- RMSE FINAL SUSCEPTIBILIDAD ---
[EVALUACIÓN] RMSE de Susceptibilidad (QSM predicha vs Cosmos): 28.19%
--------------------------------
'''

'''
=============================================================================
ADICIÓN 3D:
Se extendió el regularizador 2D a un entorno pseudo-3D calculando el 
prior en los tres ejes ortogonales (Axial, Coronal y Sagital) mediante 
permutaciones del tensor y promediando las reconstrucciones.
'''
# %%
