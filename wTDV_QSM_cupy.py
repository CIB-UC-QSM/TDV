#%%
import os
import time
import numpy as np
import cupy as cp
import torch
import scipy.io

import model
from utils import imshow_3d

tic = time.time()
#Load MATLAB data container
#groundtruth = scipy.io.loadmat('chi_cosmos.mat')['chi_cosmos']
phase = cp.array(scipy.io.loadmat('params.mat')['phase_use'].astype(np.float32))
kernel = cp.array(scipy.io.loadmat('params.mat')['kernel'].astype(np.float32))
K2 = cp.conj(kernel)*kernel
weight = cp.array(scipy.io.loadmat('params.mat')['magn_use'].astype(np.float32))
weight *= weight
alpha = scipy.io.loadmat('params.mat')['alpha1'].astype(np.float32)[0,0]
mu = scipy.io.loadmat('params.mat')['mu1'].astype(np.float32)[0,0]

maxOuterIter = scipy.io.loadmat('params.mat')['maxOuterIter'][0,0]
tolUpdate = scipy.io.loadmat('params.mat')['tol_update'].astype(np.float32)[0,0]

mu2 = 1.0

N = phase.shape
Wy = weight*phase/(weight+mu2)

## Variable initialization to allocate memory on GPU
z1 = cp.zeros(N, cp.float32)
s1 = cp.zeros(N, cp.float32)
qsm = cp.zeros(N, cp.float32)
z2 = Wy.copy() # Initialized with weighted data
s2 = cp.zeros(N, cp.float32)


color = 'color'
checkpoint = torch.load(os.path.join('checkpoints', f'tdv3-3-25-f32-{color}.pth'))
vn = model.VNet(checkpoint['config'], efficient=False)
vn.load_state_dict(checkpoint['model'])
vn.cuda()

# define the application oxxf the VN
def apply_vn(x_0, z):
    # tranform to reference noise level
    x = vn(x_0 * alpha, z * alpha)
    # convert back to original scale
    x = [j/alpha for j in x]
    return x

for t in range(0, maxOuterIter):
    # update qsm
    qsm_old = qsm.copy().astype(cp.float32)
    qsm = cp.real(cp.fft.ifftn( (mu2*cp.conj(kernel)*cp.fft.fftn(z2-s2)+mu*cp.fft.fftn(z1-s1))/(mu2*K2+mu) )).astype(cp.float32)
    x_update = 100*cp.sqrt(cp.mean((qsm-qsm_old) ** 2))/cp.sqrt(cp.mean((qsm) ** 2))
    print('Iter: '+str(t)+'   Update: '+str(x_update.item()))
    
    if x_update < tolUpdate:
        break
    FhDFx = cp.real(cp.fft.ifftn(kernel*cp.fft.fftn(qsm))).astype(cp.float32)
    # update z1 - Proximal step with TDV
    for i in range(1, N[2]-1, 2):
        z = qsm[:,:,i-1:i+2]+s1[:,:,i-1:i+2]
        # Normalizar y escalar a 256 niveles (signed int8 simétrico de -128 a 127) dinámicamente
        z_min = float(z.min())
        z_max = float(z.max())
        z_range = z_max - z_min
        
        if z_range > 1e-8:
            z_norm = (z - z_min) / z_range
            z_scaled = z_norm * 255.0 - 128.0
            z_int8 = cp.clip(cp.round(z_scaled), -128, 127).astype(cp.int8)
            # De-cuantizar de inmediato antes de pasar a la red neuronal
            z_quantized = ((z_int8.astype(cp.float32) + 128.0) / 255.0) * z_range + z_min
        else:
            z_quantized = z
            
        z_trans = cp.transpose(z_quantized, (2,0,1))[None]
        z_th = torch.as_tensor(z_trans, device='cuda')
        with torch.no_grad():
            x_th = apply_vn(z_th.contiguous(), z_th.contiguous())
        x_S_th = x_th[-1][0].permute(1,2,0)
        x_S = cp.asarray(x_S_th)
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
mdic = {"x": cp.asnumpy(qsm), "time":(toc-tic),"iter":(t+1)}
scipy.io.savemat('result_wTDV_cupy.mat', mdic)
imshow_3d(cp.asnumpy(qsm), title="wTDV_QSM", rango=(-0.1, 0.1))
imshow_3d(z_th.squeeze().cpu().numpy(), title="z_th", rango=(-0.1, 0.1)) #angles=(-90, -90, 90))
#imshow_3d(x_th[-1].squeeze().cpu().numpy(), title="x_th", rango=(-0.1, 0.1))

# %%
