#%%
#Proof of concept hecho con IA replicando el 2D desarrollado.
import os
import time
import numpy as np
import torch
import scipy.io
from scipy import ndimage

from ddr import model
from ddr.utils import imshow_3d

def print_stats(name, tensor):
    if isinstance(tensor, (int, float)):
        print(f"[DEBUG] {name:<15} | Valor: {tensor:>8.4f}")
    elif torch.is_tensor(tensor) and tensor.numel() == 1:
        print(f"[DEBUG] {name:<15} | Valor: {tensor.item():>8.4f}")
    else:
        print(f"[DEBUG] {name:<15} | Rango: {tensor.min().item():>8.4f} a {tensor.max().item():>8.4f} | Media: {tensor.mean().item():>8.4f} | Std: {tensor.std().item():>8.4f}")

def process_axis(v_in, axis_perm, vn, alpha, BATCH_SIZE):
    """
    Applies the TDV network along a specific axis to achieve pseudo-3D regularization.
    """
    v_perm = v_in.permute(*axis_perm)
    D = v_perm.shape[0]
    
    accum = torch.zeros_like(v_perm)
    count = torch.zeros_like(v_perm)
    
    triplets = v_perm.unfold(0, 3, 1).permute(0, 3, 1, 2)
    center_indices = list(range(1, D - 1))
    
    for b_start in range(0, len(center_indices), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(center_indices))
        batch = triplets[b_start:b_end]
        
        with torch.no_grad():
            batch_alpha = batch.contiguous() * alpha
            x_batch = vn(batch_alpha, batch_alpha)
            
        outputs = x_batch[-1] # (B, 3, H, W)
        
        for j, i in enumerate(center_indices[b_start:b_end]):
            accum[i-1:i+2, :, :] += outputs[j] / alpha
            count[i-1:i+2, :, :] += 1
            
        del batch, batch_alpha, x_batch, outputs
        torch.cuda.empty_cache()
            
    inv_perm = [0, 1, 2]
    for i, p in enumerate(axis_perm):
        inv_perm[p] = i
        
    return accum.permute(*inv_perm), count.permute(*inv_perm)


tic = time.time()
#Load MATLAB data container
phase_np = scipy.io.loadmat('params.mat')['phase_use'].astype(np.float32)
kernel_np = scipy.io.loadmat('params.mat')['kernel'].astype(np.float32)
K2_np = np.conj(kernel_np)*kernel_np

weight_np = scipy.io.loadmat('params.mat')['magn_use'].astype(np.float32)

# --- MASKING PARA ESTADÍSTICAS Y SKULL-STRIPPING ---
brain_solid = weight_np > 0.05
brain_solid = ndimage.binary_fill_holes(brain_solid)

mean_brain = np.mean(weight_np[brain_solid])
umbral_dinamico = 0.05 * mean_brain
mag_threshold = weight_np > umbral_dinamico
brain_clean = brain_solid & mag_threshold

brain_closed = ndimage.binary_closing(brain_clean, structure=np.ones((3,3,3)), iterations=2)
brain_final = ndimage.binary_fill_holes(brain_closed)

refined_mask = ndimage.gaussian_filter(brain_final.astype(np.float32), sigma=1.0) > 0.5

# Skull-stripping: aislar el cerebro para no converger sobre ruido del cráneo.
weight_np = weight_np * refined_mask

# --- NORMALIZACIÓN DE MAGNITUD ---
weight_np /= weight_np[refined_mask].mean()
weight_np *= weight_np

mu = 0.0245

maxOuterIter = scipy.io.loadmat('params.mat')['maxOuterIter'][0,0]
tolUpdate = scipy.io.loadmat('params.mat')['tol_update'].astype(np.float32)[0,0]

mu2 = 1.0

N = phase_np.shape
Wy_np = weight_np * phase_np

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
        BATCH_SIZE = 8
    elif total_vram_gb >= 6.0:
        BATCH_SIZE = 4
    else:
        BATCH_SIZE = 2
else:
    BATCH_SIZE = 2

# Si procesamos por las dimensiones X e Y, N[0] y N[1] pueden ser más grandes que N[2].
# Aseguramos que el batch size no reviente la RAM. BATCH_SIZE es sobre el número de slices.
# Podemos usar un max_batch_size en base a la VRAM.
max_batch_size = BATCH_SIZE

## Variable initialization to allocate memory on GPU
z1 = torch.zeros(N, dtype=torch.float32, device=device)
s1 = torch.zeros(N, dtype=torch.float32, device=device)
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

for t in range(0, maxOuterIter):
    qsm_old = qsm.clone()
    
    fft_z2_s2 = torch.fft.fftn(z2 - s2)
    fft_z1_s1 = torch.fft.fftn(z1 - s1)
    
    numerator = mu2 * kernel * fft_z2_s2 + mu * fft_z1_s1
    qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)
    
    if t == 0:
        target_std = 0.3
        qsm_std = torch.std(qsm[mask_torch])
        alpha = target_std / qsm_std.item()
        
        print("\n--- ESTADÍSTICAS ITERACIÓN 0 ---")
        print(f"[FINE TUNING] Alpha calibrado matemáticamente a: {alpha:.4f}")
        print_stats("qsm (update)", qsm)
    
    # Retomamos el cálculo original para el update (sin enmascarar)
    # para no alterar el criterio de parada original del ADMM.
    x_update = 100*torch.sqrt(torch.mean((qsm-qsm_old) ** 2))/torch.sqrt(torch.mean((qsm) ** 2))
    print('Iter: '+str(t)+'   Update: '+str(x_update.item()))
    
    if x_update < tolUpdate:
        break
    FhDFx = torch.real(torch.fft.ifftn(kernel*torch.fft.fftn(qsm))).to(torch.float32)
    
    # --- PASO PROXIMAL z1: REGULARIZADOR TDV EN 3D ---
    v = qsm + s1
    if t == 0:
        print_stats("v (qsm + s1)", v)
    
    # Aplicar la red 2D sobre los tres planos ortogonales (Axial, Coronal, Sagital)
    # Axial (Z-axis, permute: 2, 0, 1)
    accum_z, count_z = process_axis(v, (2, 0, 1), vn, alpha, max_batch_size)
    # Coronal (Y-axis, permute: 1, 0, 2)
    accum_y, count_y = process_axis(v, (1, 0, 2), vn, alpha, max_batch_size)
    # Sagital (X-axis, permute: 0, 1, 2)
    accum_x, count_x = process_axis(v, (0, 1, 2), vn, alpha, max_batch_size)
    
    z1_accum_total = accum_x + accum_y + accum_z
    z1_count_total = count_x + count_y + count_z
    
    z1_count_total[z1_count_total == 0] = 1.0
    z1 = z1_accum_total / z1_count_total
    
    s1 += qsm - z1
    
    if t == 0:
        print_stats("z1 (TDV out 3D)", z1)
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
scipy.io.savemat('results/wTDV_QSM_torch_3d.mat', mdic)

# --- EVALUACIÓN RMSE (Data Fidelity Ponderado) EN GPU ---
# Al no tener Ground Truth de susceptibilidad, la métrica física estándar es el Data Fidelity RMSE 
# ponderado por la magnitud (W). Lo calculamos 100% en la GPU usando los tensores de PyTorch.
W_torch = torch.sqrt(weight)  # weight ya está elevado al cuadrado
diff = (FhDFx - phase) * W_torch
ref = phase * W_torch
rmse_val_gpu = (torch.norm(diff[mask_torch]) / torch.norm(ref[mask_torch])) * 100.0
print(f"\n--- RMSE FINAL FASE PREDICHA VS FASE REAL ---")
print(f"[EVALUACIÓN] RMSE de Data Fidelity Ponderado (Fase predicha vs Fase real): {rmse_val_gpu.item():.2f}%")
print(f"--------------------------------")

imshow_3d(qsm_np, title="wTDV_QSM_3D (x)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='results/wTDV_QSM_torch_3d.png')
imshow_3d(v_np, title="v (input to TDV)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='figures/wTDV_v_torch_3d.png')
imshow_3d(z1_np, title="z1 (output of TDV 3D)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='figures/wTDV_z1_torch_3d.png')
imshow_3d(s1_np, title="s1 (dual variable)", rango=(-0.1, 0.1), angles=(-90, -90, 90), savepath='figures/wTDV_s1_torch_3d.png')

'''
=============================================================================
ADICIÓN 3D:
Se extendió el regularizador 2D a un entorno pseudo-3D calculando el 
prior en los tres ejes ortogonales (Axial, Coronal y Sagital) mediante 
permutaciones del tensor y promediando las reconstrucciones.
=============================================================================
'''
# %%
