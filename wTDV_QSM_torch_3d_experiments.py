import os
import time
import argparse
import numpy as np
import torch
import scipy.io
from scipy import ndimage

from ddr import model
from ddr.utils import print_stats, rmse, process_axis

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion', type=str, default='average', choices=['average', 'median', 'adaptive_mu', 'spectral'])
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"=== Corriendo estrategia de fusión: {args.fusion.upper()} ===")
    
    tic = time.time()

    # Load MATLAB data container
    phase_np = scipy.io.loadmat('params.mat')['phase_use'].astype(np.float32)
    kernel_np = scipy.io.loadmat('params.mat')['kernel'].astype(np.float32)
    K2_np = np.conj(kernel_np)*kernel_np
    weight_np = scipy.io.loadmat('params.mat')['magn_use'].astype(np.float32)
    maxOuterIter = scipy.io.loadmat('params.mat')['maxOuterIter'][0,0]
    tolUpdate = scipy.io.loadmat('params.mat')['tol_update'].astype(np.float32)[0,0]
    chi_cosmos = scipy.io.loadmat('chi_cosmos.mat')['chi_cosmos'].astype(np.float32)
    cosmos_mask = chi_cosmos != 0
    mu = 0.02
    mu2 = 1.0

    # Masking y Skull-stripping
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

    # Convert initial data to PyTorch tensors on GPU
    phase = torch.from_numpy(phase_np).to(device)
    kernel = torch.from_numpy(kernel_np).to(device)
    K2 = torch.from_numpy(K2_np).to(device)
    weight = torch.from_numpy(weight_np).to(device)
    Wy = torch.from_numpy(Wy_np).to(device)
    mask_torch = torch.from_numpy(refined_mask).to(device)

    # VRAM Optimization
    if device.type == 'cuda':
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if total_vram_gb >= 11.5:
            BATCH_SIZE = 128
        elif total_vram_gb >= 6.0:
            BATCH_SIZE = 64
        else:
            BATCH_SIZE = 16
    else:
        BATCH_SIZE = 16
    max_batch_size = BATCH_SIZE

    # Variable initialization
    zx = torch.zeros(N, dtype=torch.float32, device=device)
    sx = torch.zeros(N, dtype=torch.float32, device=device)
    zy = torch.zeros(N, dtype=torch.float32, device=device)
    sy = torch.zeros(N, dtype=torch.float32, device=device)
    zz = torch.zeros(N, dtype=torch.float32, device=device)
    sz = torch.zeros(N, dtype=torch.float32, device=device)
    
    zx_old = torch.zeros_like(zx)
    zy_old = torch.zeros_like(zy)
    zz_old = torch.zeros_like(zz)

    qsm = torch.zeros(N, dtype=torch.float32, device=device)
    weight_mu2 = weight + mu2
    z2 = Wy / weight_mu2
    s2 = torch.zeros(N, dtype=torch.float32, device=device)

    # Initialize variables for specific fusion methods
    if args.fusion == 'adaptive_mu':
        mu_x = torch.tensor(mu, device=device)
        mu_y = torch.tensor(mu, device=device)
        mu_z = torch.tensor(mu, device=device)
    elif args.fusion == 'spectral':
        kx = torch.fft.fftfreq(N[0], d=1.0).to(device)
        ky = torch.fft.fftfreq(N[1], d=1.0).to(device)
        kz = torch.fft.fftfreq(N[2], d=1.0).to(device)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        K_sq = KX**2 + KY**2 + KZ**2
        K_sq[0, 0, 0] = 1e-6 # prevent division by zero
        W_x = (KY**2 + KZ**2) / (2 * K_sq)
        W_y = (KX**2 + KZ**2) / (2 * K_sq)
        W_z = (KX**2 + KY**2) / (2 * K_sq)
        denominator = mu2 * K2 + mu

    color = 'color'
    checkpoint = torch.load(os.path.join('checkpoints', f'tdv3-3-25-f32-{color}.pth'))
    vn = model.VNet(checkpoint['config'], efficient=False)
    vn.load_state_dict(checkpoint['model'])
    vn.to(device)
    vn.eval()

    # ADMM Loop
    for t in range(0, maxOuterIter):
        qsm_old = qsm.clone()
        
        fft_z2_s2 = torch.fft.fftn(z2 - s2)
        
        if args.fusion == 'average':
            fft_zx_sx = torch.fft.fftn(zx - sx)
            fft_zy_sy = torch.fft.fftn(zy - sy)
            fft_zz_sz = torch.fft.fftn(zz - sz)
            numerator = mu2 * kernel * fft_z2_s2 + (mu / 3.0) * (fft_zx_sx + fft_zy_sy + fft_zz_sz)
            denominator = mu2 * K2 + mu
            qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)
            
        elif args.fusion == 'median':
            vx_eff = zx - sx
            vy_eff = zy - sy
            vz_eff = zz - sz
            stack = torch.stack([vx_eff, vy_eff, vz_eff], dim=0)
            consensus = torch.median(stack, dim=0).values
            fft_consensus = torch.fft.fftn(consensus)
            numerator = mu2 * kernel * fft_z2_s2 + mu * fft_consensus
            denominator = mu2 * K2 + mu 
            qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)
            
        elif args.fusion == 'adaptive_mu':
            fft_zx_sx = torch.fft.fftn(zx - sx)
            fft_zy_sy = torch.fft.fftn(zy - sy)
            fft_zz_sz = torch.fft.fftn(zz - sz)
            numerator = mu2 * kernel * fft_z2_s2 + mu_x * fft_zx_sx + mu_y * fft_zy_sy + mu_z * fft_zz_sz
            denominator_adaptive = mu2 * K2 + (mu_x + mu_y + mu_z)
            qsm = torch.real(torch.fft.ifftn(numerator / denominator_adaptive)).to(torch.float32)
            
        elif args.fusion == 'spectral':
            fft_zx_sx = torch.fft.fftn(zx - sx)
            fft_zy_sy = torch.fft.fftn(zy - sy)
            fft_zz_sz = torch.fft.fftn(zz - sz)
            numerator = mu2 * kernel * fft_z2_s2 + mu * (W_x * fft_zx_sx + W_y * fft_zy_sy + W_z * fft_zz_sz)
            qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)

        if t == 0:
            target_std = 0.1
            qsm_std = torch.std(qsm[mask_torch])
            alpha = target_std / qsm_std.item() if qsm_std.item() > 0 else 1.0

        x_update = 100*torch.sqrt(torch.mean((qsm-qsm_old) ** 2))/torch.sqrt(torch.mean((qsm) ** 2))
        
        if x_update < tolUpdate:
            break
            
        FhDFx = torch.real(torch.fft.ifftn(kernel*torch.fft.fftn(qsm))).to(torch.float32)
        
        if args.fusion == 'adaptive_mu':
            zx_old.copy_(zx)
            zy_old.copy_(zy)
            zz_old.copy_(zz)
        
        # Proximal step: TDV Multi-variable
        vx = qsm + sx
        vy = qsm + sy
        vz = qsm + sz
        
        zz = process_axis(vz, (2, 0, 1), vn, alpha, max_batch_size)
        zy = process_axis(vy, (1, 0, 2), vn, alpha, max_batch_size)
        zx = process_axis(vx, (0, 1, 2), vn, alpha, max_batch_size)
        
        sx += qsm - zx
        sy += qsm - zy
        sz += qsm - zz
        
        if args.fusion == 'adaptive_mu':
            # Actualizar mu_x, mu_y, mu_z basado en balance primal-dual
            with torch.no_grad():
                r_x = torch.norm(qsm - zx)
                s_dual_x = mu_x * torch.norm(zx - zx_old)
                if r_x > 10 * s_dual_x:
                    mu_x_old = mu_x.clone()
                    mu_x *= 1.5
                    sx *= (mu_x_old / mu_x)
                elif s_dual_x > 10 * r_x:
                    mu_x_old = mu_x.clone()
                    mu_x /= 1.5
                    sx *= (mu_x_old / mu_x)
                
                r_y = torch.norm(qsm - zy)
                s_dual_y = mu_y * torch.norm(zy - zy_old)
                if r_y > 10 * s_dual_y:
                    mu_y_old = mu_y.clone()
                    mu_y *= 1.5
                    sy *= (mu_y_old / mu_y)
                elif s_dual_y > 10 * r_y:
                    mu_y_old = mu_y.clone()
                    mu_y /= 1.5
                    sy *= (mu_y_old / mu_y)
                
                r_z = torch.norm(qsm - zz)
                s_dual_z = mu_z * torch.norm(zz - zz_old)
                if r_z > 10 * s_dual_z:
                    mu_z_old = mu_z.clone()
                    mu_z *= 1.5
                    sz *= (mu_z_old / mu_z)
                elif s_dual_z > 10 * r_z:
                    mu_z_old = mu_z.clone()
                    mu_z /= 1.5
                    sz *= (mu_z_old / mu_z)
            
        # update z2
        z2 = (Wy + mu2*(FhDFx+s2)) / weight_mu2
        s2 += FhDFx - z2
        
        if (t + 1) % 5 == 0 or t == 0:
            toc_iter = time.time()
            qsm_iter_np = qsm.cpu().numpy()
            rmse_iter = rmse(qsm_iter_np, chi_cosmos, mask=cosmos_mask)
            print(f"Iteración {t+1:02d}/{int(maxOuterIter)} | RMSE: {rmse_iter:.2f}% | x_update: {x_update.item():.3f}% | Tiempo: {toc_iter-tic:.1f}s")
            if args.fusion == 'adaptive_mu':
                print(f"  mu_x: {mu_x.item():.4f}, mu_y: {mu_y.item():.4f}, mu_z: {mu_z.item():.4f}")
              
    toc = time.time()    
    print(f"corrido en {toc-tic:.1f} segundos")

    qsm_np = qsm.cpu().numpy()
    rmse_val = rmse(qsm_np, chi_cosmos, mask=cosmos_mask)
    print(f"[{args.fusion.upper()}] RMSE FINAL: {rmse_val:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()
