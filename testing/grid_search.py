import os
import time
import numpy as np
import torch
import scipy.io
from scipy import ndimage
from ddr import model
from ddr.utils import rmse

#Load MATLAB data container
phase_np = scipy.io.loadmat('params.mat')['phase_use'].astype(np.float32)
kernel_np = scipy.io.loadmat('params.mat')['kernel'].astype(np.float32)
K2_np = np.conj(kernel_np)*kernel_np
weight_np = scipy.io.loadmat('params.mat')['magn_use'].astype(np.float32)
maxOuterIter = scipy.io.loadmat('params.mat')['maxOuterIter'][0,0]
tolUpdate = scipy.io.loadmat('params.mat')['tol_update'].astype(np.float32)[0,0]
chi_cosmos = scipy.io.loadmat('chi_cosmos.mat')['chi_cosmos'].astype(np.float32)
cosmos_mask = chi_cosmos != 0

brain_solid = weight_np > 0.05
brain_solid = ndimage.binary_fill_holes(brain_solid)
mean_brain = np.mean(weight_np[brain_solid])
umbral_dinamico = 0.05 * mean_brain
mag_threshold = weight_np > umbral_dinamico
brain_clean = brain_solid & mag_threshold
brain_closed = ndimage.binary_closing(brain_clean, structure=np.ones((3,3,3)), iterations=2)
brain_final = ndimage.binary_fill_holes(brain_closed)

refined_mask = ndimage.gaussian_filter(brain_final.astype(np.float32), sigma=1.0) > 0.5
weight_np = weight_np * refined_mask
weight_np /= weight_np[refined_mask].mean()
weight_np *= weight_np

N = phase_np.shape
Wy_np = weight_np * phase_np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

phase = torch.from_numpy(phase_np).to(device)
kernel = torch.from_numpy(kernel_np).to(device)
K2 = torch.from_numpy(K2_np).to(device)
weight = torch.from_numpy(weight_np).to(device)
Wy = torch.from_numpy(Wy_np).to(device)
mask_torch = torch.from_numpy(refined_mask).to(device)

if device.type == 'cuda':
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if total_vram_gb >= 11.5:
        BATCH_SIZE = 16
    elif total_vram_gb >= 6.0:
        BATCH_SIZE = 8
    else:
        BATCH_SIZE = 4
else:
    BATCH_SIZE = 4

color = 'color'
checkpoint = torch.load(os.path.join('checkpoints', f'tdv3-3-25-f32-{color}.pth'))
vn = model.VNet(checkpoint['config'], efficient=False)
vn.load_state_dict(checkpoint['model'])
vn.to(device)
vn.eval()

def run_qsm(mu, target_std):
    mu2 = 1.0
    z1 = torch.zeros(N, dtype=torch.float32, device=device)
    s1 = torch.zeros(N, dtype=torch.float32, device=device)
    qsm = torch.zeros(N, dtype=torch.float32, device=device)
    weight_mu2 = weight + mu2
    denominator = mu2 * K2 + mu
    z2 = Wy / weight_mu2
    s2 = torch.zeros(N, dtype=torch.float32, device=device)
    
    for t in range(0, maxOuterIter):
        qsm_old = qsm.clone()
        fft_z2_s2 = torch.fft.fftn(z2 - s2)
        fft_z1_s1 = torch.fft.fftn(z1 - s1)
        numerator = mu2 * kernel * fft_z2_s2 + mu * fft_z1_s1
        qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)
        
        if t == 0:
            qsm_std = torch.std(qsm[mask_torch])
            alpha = target_std / qsm_std.item()

        x_update = 100*torch.sqrt(torch.mean((qsm-qsm_old) ** 2))/torch.sqrt(torch.mean((qsm) ** 2))
        if x_update < tolUpdate:
            break
            
        FhDFx = torch.real(torch.fft.ifftn(kernel*torch.fft.fftn(qsm))).to(torch.float32)
        v = qsm + s1
        
        D = v.shape[2]
        accum = torch.zeros_like(v)
        count = torch.zeros_like(v)
        triplets = v.unfold(2, 3, 1).permute(2, 3, 0, 1)
        center_indices = list(range(1, D - 1))
        
        for b_start in range(0, len(center_indices), BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, len(center_indices))
            batch = triplets[b_start:b_end]
            with torch.no_grad():
                batch_alpha = batch.contiguous() * alpha
                x_batch = vn(batch_alpha, batch_alpha)
            outputs = x_batch[-1]
            for j, i in enumerate(center_indices[b_start:b_end]):
                accum[:, :, i-1:i+2] += outputs[j].permute(1, 2, 0) / alpha
                count[:, :, i-1:i+2] += 1
            del batch, batch_alpha, x_batch, outputs
            torch.cuda.empty_cache()
            
        count[count == 0] = 1.0
        z1 = accum / count
        s1 += qsm - z1
        z2 = (Wy + mu2*(FhDFx+s2)) / weight_mu2
        s2 += FhDFx - z2
        
    rmse_val = rmse(qsm.cpu().numpy(), chi_cosmos, mask=cosmos_mask)
    return rmse_val

best_rmse = 100
best_params = None

for mu in [0.02, 0.0245, 0.03]:
    for target_std in [0.05, 0.1, 0.2]:
        try:
            r = run_qsm(mu, target_std)
            print(f"mu={mu}, target_std={target_std} -> RMSE={r:.2f}%")
            if r < best_rmse:
                best_rmse = r
                best_params = (mu, target_std)
        except Exception as e:
            print(f"Failed for mu={mu}, target_std={target_std}: {e}")

print(f"BEST: mu={best_params[0]}, target_std={best_params[1]} with RMSE={best_rmse:.2f}%")

