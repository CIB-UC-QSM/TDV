"""
Dataset para el reentrenamiento de TDV-VNet en QSM (modo 2.5D / 3 canales).

Cada muestra de entrenamiento es un triplete de slices axiales contiguos
(z-1, z, z+1) apilados como canales (3, H, W), de modo que sea compatible con
el regularizador TDV `in_channels = 3` del checkpoint de denoising a color.

Un volumen de entrenamiento debe contener:
    - phase  : fase local medida (input del problema inverso)   φ
    - chi    : susceptibilidad de referencia (ground truth)      χ_gt
    - weight : pesos de magnitud (data fidelity weighting)       W
    - kernel : kernel dipolar en k-space (3D global o por volumen) D

Convencion de k-space: el dataterm aplica  F^-1{ D . F{x} }  con `torch.fft`
SIN fftshift (componente DC en la esquina [0,0]), igual que `wTDV_QSM_torch.py`.
El kernel guardado debe seguir esa misma convencion (no fftshifted).

Se aceptan archivos ``.mat`` (v7 vía scipy, v7.3/HDF5 vía h5py) y ``.h5``.

La salida de ``__getitem__`` es un diccionario de tensores ``float32``:
    {
        'phase' : (3, H, W),   # fase de los slices (z-1, z, z+1)
        'chi_gt': (3, H, W),   # susceptibilidad ground truth
        'weight': (3, H, W),   # pesos de magnitud W (no W²)
        'kernel': (H, W),      # kernel dipolar 2D proyectado en k-space
    }
"""

import os
import sys
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

# Permite que el archivo viva en un subdirectorio (e.g. retraining/) y se ejecute
# desde cualquier CWD: agrega al sys.path el directorio que contiene model.py.
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here if os.path.exists(os.path.join(_here, 'model.py')) else os.path.dirname(_here)
for _p in (_here, _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import QSMDataterm


# --------------------------------------------------------------------------- #
#  Utilidades de carga (.mat v7 / v7.3 / .h5)
# --------------------------------------------------------------------------- #
def _load_matlike(path, keys):
    """Carga un subconjunto de variables de un archivo .mat o .h5.

    Devuelve un dict {nombre_logico: np.ndarray}. Soporta MATLAB v7 (scipy)
    y v7.3/HDF5 (h5py). Las claves se buscan de forma case-insensitive.
    """
    out = {}
    import h5py

    # HDF5 cubre tanto .h5 nativo como MATLAB v7.3; scipy.io solo lee <= v7.
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            available = {k.lower(): k for k in f.keys()}
            for logical, candidates in keys.items():
                real = _match_key(candidates, available)
                if real is not None:
                    arr = np.asarray(f[real])
                    # h5py lee los .mat v7.3 transpuestos respecto a MATLAB
                    out[logical] = arr.T if arr.ndim >= 2 else arr
        return out

    import scipy.io

    mat = scipy.io.loadmat(path)
    available = {k.lower(): k for k in mat.keys() if not k.startswith('__')}
    for logical, candidates in keys.items():
        real = _match_key(candidates, available)
        if real is not None:
            out[logical] = np.asarray(mat[real])
    return out


def _match_key(candidates, available_lower):
    """Devuelve la clave real (con su casing original) que matchea alguno de
    los nombres candidatos, o None si ninguno existe."""
    for cand in candidates:
        if cand.lower() in available_lower:
            return available_lower[cand.lower()]
    return None


def _to_complex_kernel(arr):
    """h5py devuelve complejos como structured arrays con campos 'real'/'imag'."""
    if arr.dtype.names is not None and 'real' in arr.dtype.names:
        return arr['real'] + 1j * arr['imag']
    return arr


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class QSMDataset(Dataset):
    """Dataset de pares (fase, susceptibilidad) para entrenar la VNet en QSM.

    Args:
        root: carpeta con los volumenes, o lista explicita de rutas de archivo.
        pattern: glob para descubrir archivos dentro de ``root`` (e.g. '*.mat').
        phase_key/chi_key/weight_key/kernel_key: nombres (o listas de nombres
            alternativos) de las variables dentro de cada archivo.
        kernel_method: 'mean' (proyeccion integrando en kz, robusta) o 'central'
            (slice central de kz) para construir el kernel dipolar 2D.
        scale: factor lineal aplicado a fase y chi_gt para llevar los valores al
            rango en que opera el regularizador TDV (las activaciones Student-t
            son no lineales). El forward model es lineal, asi que escalar fase y
            chi por el mismo factor es consistente. W queda sin escalar.
        skip_border: si True, descarta slices sin vecinos validos (extremos del
            eje axial), evitando padding artificial en los tripletes.
        cache: mantiene los volumenes ya leidos en RAM para acelerar las epocas.
    """

    def __init__(
        self,
        root,
        pattern='*.mat',
        phase_key=('phase', 'phase_use', 'phi', 'local_field'),
        chi_key=('chi', 'chi_gt', 'susc', 'cosmos', 'x_gt', 'chi_cosmos'),
        weight_key=('weight', 'magn', 'magn_use', 'mask', 'W'),
        kernel_key=('kernel', 'dipole', 'D'),
        kernel_method='mean',
        scale=1.0,
        skip_border=True,
        cache=True,
    ):
        super().__init__()

        if isinstance(root, (list, tuple)):
            self.files = list(root)
        else:
            self.files = sorted(glob.glob(os.path.join(root, pattern)))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f'No se encontraron volumenes en {root!r} con patron {pattern!r}.'
            )

        self.keys = {
            'phase': _as_tuple(phase_key),
            'chi': _as_tuple(chi_key),
            'weight': _as_tuple(weight_key),
            'kernel': _as_tuple(kernel_key),
        }
        self.kernel_method = kernel_method
        self.scale = float(scale)
        self.skip_border = skip_border
        self.cache = cache

        self._volume_cache = {}

        # Construir el indice (file_idx, center_slice) recorriendo cada volumen.
        self.index = []
        for fi, path in enumerate(self.files):
            nz = self._get_volume(fi)['phase'].shape[2]
            if self.skip_border:
                centers = range(1, nz - 1)        # tripletes (z-1, z, z+1) validos
            else:
                centers = range(0, nz)
            for c in centers:
                self.index.append((fi, c))

        if len(self.index) == 0:
            raise RuntimeError('El dataset quedo vacio (revisa el eje axial / skip_border).')

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.index)

    def _get_volume(self, file_idx, path=None):
        """Carga (y opcionalmente cachea) un volumen ya pre-procesado a float32."""
        if file_idx in self._volume_cache:
            return self._volume_cache[file_idx]

        path = path if path is not None else self.files[file_idx]
        raw = _load_matlike(path, self.keys)

        for req in ('phase', 'chi', 'weight', 'kernel'):
            if req not in raw:
                raise KeyError(
                    f"El archivo {os.path.basename(path)} no contiene la variable "
                    f"'{req}' (busque entre {self.keys[req]})."
                )

        phase = np.asarray(raw['phase'], dtype=np.float32)
        chi = np.asarray(raw['chi'], dtype=np.float32)
        weight = np.asarray(raw['weight'], dtype=np.float32)

        if not (phase.shape == chi.shape == weight.shape):
            raise ValueError(
                f"Dimensiones inconsistentes en {os.path.basename(path)}: "
                f"phase={phase.shape}, chi={chi.shape}, weight={weight.shape}."
            )
        if phase.ndim != 3:
            raise ValueError(
                f"Se esperan volumenes 3D (H, W, Z); {os.path.basename(path)} "
                f"tiene phase.ndim={phase.ndim}."
            )

        # Kernel dipolar: aceptar 3D (k-space) y proyectar a 2D, o 2D directo.
        kernel = _to_complex_kernel(np.asarray(raw['kernel']))
        if kernel.ndim == 3:
            kernel_2d = QSMDataterm.project_kernel_2d(kernel, method=self.kernel_method)
        elif kernel.ndim == 2:
            kernel_2d = np.real(kernel).astype(np.float32)
        else:
            raise ValueError(
                f"Kernel con ndim={kernel.ndim} no soportado en {os.path.basename(path)}."
            )
        kernel_2d = np.ascontiguousarray(kernel_2d, dtype=np.float32)

        vol = {
            'phase': phase * self.scale,
            'chi': chi * self.scale,
            'weight': weight,            # W no se escala (multiplica el residuo)
            'kernel': kernel_2d,         # (H, W) en k-space
        }
        if self.cache:
            self._volume_cache[file_idx] = vol
        return vol

    # ------------------------------------------------------------------ #
    def _triplet(self, vol_arr, center):
        """Extrae los slices (z-1, z, z+1) como canales (3, H, W).

        Si ``skip_border`` es False y el centro esta en un borde, se replica el
        slice de borde (padding por borde) para mantener 3 canales.
        """
        nz = vol_arr.shape[2]
        idxs = [max(0, center - 1), center, min(nz - 1, center + 1)]
        # (H, W, 3) -> (3, H, W)
        return np.transpose(vol_arr[:, :, idxs], (2, 0, 1)).copy()

    def __getitem__(self, i):
        file_idx, center = self.index[i]
        vol = self._get_volume(file_idx)

        phase = self._triplet(vol['phase'], center)
        chi = self._triplet(vol['chi'], center)
        weight = self._triplet(vol['weight'], center)
        kernel = vol['kernel']  # (H, W)

        return {
            'phase': torch.from_numpy(phase),                  # (3, H, W)
            'chi_gt': torch.from_numpy(chi),                   # (3, H, W)
            'weight': torch.from_numpy(weight),                # (3, H, W)
            'kernel': torch.from_numpy(kernel),                # (H, W)
        }


def _as_tuple(x):
    return (x,) if isinstance(x, str) else tuple(x)
