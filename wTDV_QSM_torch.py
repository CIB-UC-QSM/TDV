#%%
import os
import time
import numpy as np
import torch
import scipy.io

import model
from utils import imshow_3d


tic = time.time()
#Load MATLAB data container
phase_np = scipy.io.loadmat('params.mat')['phase_use'].astype(np.float32)
kernel_np = scipy.io.loadmat('params.mat')['kernel'].astype(np.float32)
K2_np = np.conj(kernel_np)*kernel_np
weight_np = scipy.io.loadmat('params.mat')['magn_use'].astype(np.float32)
weight_np *= weight_np
alpha = scipy.io.loadmat('params.mat')['alpha1'].astype(np.float32)[0,0]
mu = scipy.io.loadmat('params.mat')['mu1'].astype(np.float32)[0,0]

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

# define the application of the VN
def apply_vn(x_0, z):
    # tranform to reference noise level
    x = vn(x_0 * alpha, z * alpha)
    # convert back to original scale
    x = [j/alpha for j in x]
    return x

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
    
    # update z1 - Proximal step with TDV
    for i in range(1, N[2]-1, 2):
        z = qsm[:,:,i-1:i+2]+s1[:,:,i-1:i+2]
        # Normalizar y escalar a 256 niveles (signed int8 simétrico de -128 a 127)
        z_scaled = torch.clamp(torch.round(z * 1280.0), -128, 127)
        z_int8 = z_scaled.to(torch.int8)
        
        # De-cuantizar de inmediato antes de pasar a la red neuronal
        # para que la red reciba la fase en el rango correcto [-0.1, 0.1]
        z_quantized = z_int8.to(torch.float32) / 1280.0
        z_th = z_quantized.permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            x_th = apply_vn(z_th.contiguous(), z_th.contiguous())
        x_S = x_th[-1][0].permute(1, 2, 0)      # already on GPU
        
        z1[:,:,(i-1)] = (x_S[:,:,0]+z1[:,:,(i-1)])/2
        z1[:,:,i] = x_S[:,:,1]
        z1[:,:,i+1] = x_S[:,:,2]
        
    s1 += qsm-z1
    # update z2
    z2 = Wy + mu2*(FhDFx+s2)/(weight + mu2)
    s2 += FhDFx - z2
          
toc = time.time()    
print(f"corrido en {toc-tic} segundos")

## Save output    
mdic = {"x": qsm.cpu().numpy(), "time":(toc-tic),"iter":(t+1)}
scipy.io.savemat('result_wTDV_torch.mat', mdic)
imshow_3d(qsm.cpu().numpy(), title="wTDV_QSM", rango=(-0.1, 0.1)) #angles=(-90, -90, 90))
imshow_3d(z_th.squeeze().cpu().numpy(), title="z_th", rango=(-0.1, 0.1)) #angles=(-90, -90, 90))
#imshow_3d(x_th[-1].squeeze().cpu().numpy(), title="x_th", rango=(-0.1, 0.1))
# %%
