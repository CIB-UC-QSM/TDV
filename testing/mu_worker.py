"""
Worker script: ejecuta ADMM para una lista de valores de mu en una GPU específica.
Uso: python mu_worker.py <gpu_id> <output_file> <mu1,mu2,...>
"""
import sys
import os
import time
import json
import numpy as np
import torch
import scipy.io
from scipy import ndimage

import sys
import os

# Add project root to sys.path to allow importing from ddr package
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from ddr import model

def main():
    gpu_id = int(sys.argv[1])
    output_file = sys.argv[2]
    mu_values = [float(x) for x in sys.argv[3].split(',')]
    
    device = torch.device(f'cuda:{gpu_id}')
    print(f"[GPU {gpu_id}] Cargando datos... (μ values: {mu_values})")
    
    # Carga de datos
    phase_np = scipy.io.loadmat('params.mat')['phase_use'].astype(np.float32)
    kernel_np = scipy.io.loadmat('params.mat')['kernel'].astype(np.float32)
    K2_np = np.conj(kernel_np) * kernel_np
    weight_np = scipy.io.loadmat('params.mat')['magn_use'].astype(np.float32)
    
    # Máscara
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
    
    Wy_np = weight_np * phase_np
    maxOuterIter = int(scipy.io.loadmat('params.mat')['maxOuterIter'][0,0])
    tolUpdate = scipy.io.loadmat('params.mat')['tol_update'].astype(np.float32)[0,0]
    mu2 = 1.0
    N = phase_np.shape
    
    # Tensores a GPU
    phase = torch.from_numpy(phase_np).to(device)
    kernel = torch.from_numpy(kernel_np).to(device)
    K2 = torch.from_numpy(K2_np).to(device)
    weight = torch.from_numpy(weight_np).to(device)
    Wy = torch.from_numpy(Wy_np).to(device)
    mask_torch = torch.from_numpy(refined_mask).to(device)
    
    # Batching
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if total_vram_gb >= 11.5:
        BATCH_SIZE = N[2]
    elif total_vram_gb >= 6.0:
        BATCH_SIZE = N[2] // 2 + 1
    else:
        BATCH_SIZE = N[2] // 4 + 1
    
    # Cargar modelo TDV
    checkpoint = torch.load(os.path.join('checkpoints', 'tdv3-3-25-f32-color.pth'), map_location=device)
    vn = model.VNet(checkpoint['config'], efficient=False)
    vn.load_state_dict(checkpoint['model'])
    vn.to(device)
    vn.eval()
    
    center_indices = list(range(1, N[2]-1))
    
    # Máscara erosionada para sharpness
    mask_eroded = torch.from_numpy(
        ndimage.binary_erosion(refined_mask, iterations=1)
    ).to(device)
    
    print(f"[GPU {gpu_id}] Modelo cargado. Iniciando grid search de {len(mu_values)} valores...")
    
    results = {}
    
    for mu_idx, mu_val in enumerate(mu_values):
        tic = time.time()
        
        weight_mu2 = weight + mu2
        denominator = mu2 * K2 + mu_val
        
        z1 = torch.zeros(N, dtype=torch.float32, device=device)
        s1 = torch.zeros(N, dtype=torch.float32, device=device)
        qsm = torch.zeros(N, dtype=torch.float32, device=device)
        z2 = Wy / weight_mu2
        s2 = torch.zeros(N, dtype=torch.float32, device=device)
        z1_accum = torch.zeros(N, dtype=torch.float32, device=device)
        z1_count = torch.zeros(N, dtype=torch.float32, device=device)
        
        alpha = None
        updates = []
        
        for t in range(maxOuterIter):
            qsm_old = qsm.clone()
            
            fft_z2_s2 = torch.fft.fftn(z2 - s2)
            fft_z1_s1 = torch.fft.fftn(z1 - s1)
            numerator = mu2 * kernel * fft_z2_s2 + mu_val * fft_z1_s1
            qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)
            
            if t == 0:
                target_std = 0.3
                qsm_std = torch.std(qsm[mask_torch])
                alpha = target_std / qsm_std.item()
            
            x_update = 100*torch.sqrt(torch.mean((qsm-qsm_old)**2))/torch.sqrt(torch.mean(qsm**2))
            updates.append(x_update.item())
            
            if x_update < tolUpdate:
                break
            
            FhDFx = torch.real(torch.fft.ifftn(kernel * torch.fft.fftn(qsm))).to(torch.float32)
            
            # z1 update (TDV)
            v = qsm + s1
            z1_accum.zero_()
            z1_count.zero_()
            
            v_DHW = v.permute(2, 0, 1)
            triplets = v_DHW.unfold(0, 3, 1).permute(0, 3, 1, 2)
            
            for b_start in range(0, len(center_indices), BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, len(center_indices))
                batch = triplets[b_start:b_end]
                with torch.no_grad():
                    batch_alpha = batch.contiguous() * alpha
                    x_batch = vn(batch_alpha, batch_alpha)
                outputs = x_batch[-1]
                
                for j, i in enumerate(center_indices[b_start:b_end]):
                    x_S_scaled = outputs[j].permute(1, 2, 0)
                    x_S = x_S_scaled / alpha
                    z1_accum[:, :, i-1:i+2] += x_S
                    z1_count[:, :, i-1:i+2] += 1
            
            z1_count[z1_count == 0] = 1.0
            z1 = z1_accum / z1_count
            s1 += qsm - z1
            z2 = (Wy + mu2 * (FhDFx + s2)) / weight_mu2
            s2 += FhDFx - z2
        
        toc = time.time()
        n_iter = t + 1
        
        # --- MÉTRICAS ---
        brain = qsm[mask_torch]
        dynamic_range = brain.max().item() - brain.min().item()
        brain_std = brain.std().item()
        
        # Sharpness
        gx = qsm[1:,:,:] - qsm[:-1,:,:]
        gy = qsm[:,1:,:] - qsm[:,:-1,:]
        gz = qsm[:,:,1:] - qsm[:,:,:-1]
        sharpness_x = torch.abs(gx[mask_eroded[1:,:,:]]).mean().item()
        sharpness_y = torch.abs(gy[mask_eroded[:,1:,:]]).mean().item()
        sharpness_z = torch.abs(gz[mask_eroded[:,:,1:]]).mean().item()
        sharpness = (sharpness_x + sharpness_y + sharpness_z) / 3
        
        # HF energy
        qsm_np = qsm.cpu().numpy()
        laplacian = ndimage.laplace(qsm_np)
        hf_energy = np.sqrt(np.mean(laplacian[refined_mask]**2))
        
        # Cone streaking
        K2_small = K2.cpu().numpy() < 0.005
        qsm_fft = torch.fft.fftn(qsm)
        cone_energy = torch.abs(qsm_fft).cpu().numpy()
        cone_streak = np.std(cone_energy[K2_small])
        
        # SNR
        outside = qsm[~mask_torch]
        noise_std = outside.std().item()
        snr = brain_std / noise_std if noise_std > 0 else float('inf')
        
        # Data fidelity
        FDx = torch.real(torch.fft.ifftn(kernel * torch.fft.fftn(qsm))).to(torch.float32)
        residual = weight * (FDx - phase)
        data_fidelity = torch.sqrt(torch.mean(residual[mask_torch]**2)).item()
        
        results[str(mu_val)] = {
            'n_iter': n_iter,
            'time': toc - tic,
            'alpha': alpha,
            'dynamic_range': dynamic_range,
            'brain_std': brain_std,
            'sharpness': sharpness,
            'hf_energy': float(hf_energy),
            'cone_streak': float(cone_streak),
            'snr': snr,
            'data_fidelity': data_fidelity,
            'final_update': updates[-1] if updates else float('nan'),
            'updates': updates,
        }
        
        print(f"[GPU {gpu_id}] μ={mu_val:.4f} ({mu_idx+1}/{len(mu_values)}): "
              f"Iter={n_iter:>3d} Time={toc-tic:.1f}s Sharp={sharpness:.6f} "
              f"DataFid={data_fidelity:.6f} SNR={snr:.2f}")
        
        # Guardar QSM para comparación visual del actual y potencial óptimo
        if mu_val in [0.0245, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
            scipy.io.savemat(f'figures/qsm_mu_{mu_val:.4f}.mat', {'qsm': qsm_np})
        
        del z1, s1, qsm, z2, s2, z1_accum, z1_count
        torch.cuda.empty_cache()
    
    # Guardar resultados
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[GPU {gpu_id}] ✅ Completado. Resultados en {output_file}")

if __name__ == '__main__':
    main()
