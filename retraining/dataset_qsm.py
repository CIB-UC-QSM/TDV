"""
Dataset para el reentrenamiento de TDV-VNet en QSM.
Optimizado para Alto Rendimiento: Carga volúmenes completos y realiza la simulación física (FFT 3D) 
solo UNA vez por volumen. Los slices 2.5D se extraen en el DataLoader/Loop.
"""

import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here if os.path.exists(os.path.join(_here, 'model.py')) else os.path.dirname(_here)
for _p in (_here, _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ddr.model import QSMDataterm

def _load_matlike(path, keys):
    out = {}
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            available = {k.lower(): k for k in f.keys()}
            for logical, candidates in keys.items():
                for cand in candidates:
                    if cand.lower() in available:
                        arr = np.asarray(f[available[cand.lower()]])
                        out[logical] = arr.T if arr.ndim >= 2 else arr
                        break
        return out
    import scipy.io
    mat = scipy.io.loadmat(path)
    available = {k.lower(): k for k in mat.keys() if not k.startswith('__')}
    for logical, candidates in keys.items():
        for cand in candidates:
            if cand.lower() in available:
                out[logical] = np.asarray(mat[available[cand.lower()]])
                break
    return out

def _to_complex_kernel(arr):
    if arr.dtype.names is not None and 'real' in arr.dtype.names:
        return arr['real'] + 1j * arr['imag']
    return arr


class QSMDataset(Dataset):
    def __init__(
        self, root, pattern='*.mat', 
        phase_key=('phase', 'phase_use', 'phi', 'local_field'),
        chi_key=('chi', 'chi_gt', 'susc', 'cosmos', 'x_gt', 'chi_cosmos'),
        weight_key=('weight', 'magn', 'magn_use', 'mask', 'W'),
        kernel_key=('kernel', 'dipole', 'D'),
        kernel_method='mean', scale=1.0, limit=None
    ):
        super().__init__()
        if isinstance(root, (list, tuple)):
            self.files = list(root)
        else:
            self.files = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
        if limit is not None:
            self.files = self.files[:limit]
        if len(self.files) == 0:
            raise FileNotFoundError(f'No se encontraron volumenes en {root!r} con patron {pattern!r}.')

        self.keys = {'phase': phase_key, 'chi': chi_key, 'weight': weight_key, 'kernel': kernel_key}
        self.kernel_method = kernel_method
        self.scale = float(scale)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        
        # --- Simulación "On-the-fly" altamente optimizada ---
        if path.endswith('.nii.gz') or path.endswith('.nii'):
            import nibabel as nib
            img = nib.load(path)
            chi = np.asarray(img.dataobj, dtype=np.float32)
            
            Nx, Ny, Nz = chi.shape
            zooms = img.header.get_zooms()
            dx, dy, dz = zooms[:3]
            
            kx = np.fft.fftfreq(Nx, d=dx)
            ky = np.fft.fftfreq(Ny, d=dy)
            kz = np.fft.fftfreq(Nz, d=dz)
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
            K2 = KX**2 + KY**2 + KZ**2
            K2[K2 == 0] = 1e-6
            D = 1/3 - (KZ)**2 / K2  
            D[0, 0, 0] = 0
            
            phase = np.fft.ifftn(D * np.fft.fftn(chi)).real.astype(np.float32)
            weight = np.ones_like(chi, dtype=np.float32)
            mask = np.ones_like(chi, dtype=np.float32)
            
            kernel_3d = np.ascontiguousarray(D, dtype=np.float32)
            
        else:
            raw = _load_matlike(path, self.keys)
            phase = np.asarray(raw['phase'], dtype=np.float32)
            chi = np.asarray(raw['chi'], dtype=np.float32)
            weight = np.asarray(raw['weight'], dtype=np.float32)
            mask = np.ones_like(chi, dtype=np.float32)
            
            kernel = _to_complex_kernel(np.asarray(raw['kernel']))
            kernel_3d = np.real(kernel).astype(np.float32)
            kernel_3d = np.ascontiguousarray(kernel_3d, dtype=np.float32)

        # Transponer de (X, Y, Z) a (Z, X, Y) para extracción rápida de tripletes en el bucle principal
        # ascontiguousarray es CRÍTICO aquí para máxima velocidad de transferencia PCIe a GPU
        phase = np.ascontiguousarray(phase.transpose(2, 0, 1))
        chi = np.ascontiguousarray(chi.transpose(2, 0, 1))
        weight = np.ascontiguousarray(weight.transpose(2, 0, 1))
        mask = np.ascontiguousarray(mask.transpose(2, 0, 1))
        if kernel_3d.ndim == 3:
            kernel_3d = np.ascontiguousarray(kernel_3d.transpose(2, 0, 1))

        return {
            'phase': torch.from_numpy(phase * self.scale),
            'chi_gt': torch.from_numpy(chi * self.scale),
            'weight': torch.from_numpy(weight),
            'mask': torch.from_numpy(mask),
            'kernel': torch.from_numpy(kernel_3d)
        }
