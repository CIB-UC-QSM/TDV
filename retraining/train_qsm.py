"""
Reentrenamiento end-to-end (unrolled) de la TDV-VNet para QSM.

La VNet realiza S pasos de descenso por gradiente sobre

        min_x   R(x) + lambda * D(x, z)

donde R es el regularizador TDV (entrenable) y D es el dataterm fisico de QSM
    D(x, z) = 1/2 || W (A x - z) ||^2 ,   A x = F^-1{ D_k . F{x} }
que NO tiene parametros entrenables. Reentrenamos R, T (stopping time) y lambda
para que el regularizador aprenda los artefactos especificos del mapeo de
susceptibilidad (e.g. streaking) en lugar del denoising generico de imagenes
naturales con el que viene pre-entrenado.

Transfer learning: se inicializa R desde el checkpoint de denoising
(`tdv3-3-25-f32-color.pth`) y se reinicializan los escalares T y lambda a
valores estables (no se cargan del checkpoint) para no quedar atascados en el
minimo local de denoising.

Tras cada paso del optimizador se aplican las restricciones del TDV llamando
`p.proj()` sobre los parametros que lo definen (filtros zero-mean / norma
espectral acotada y escalares positivos), segun `ddr/conv.py` y `model.py`.

Uso tipico (funciona desde cualquier CWD, aunque el script viva en retraining/):
    python retraining/train_qsm.py \
        --train-data data/train --val-data data/val \
        --out-ckpt checkpoints/tdv-qsm-color.pth \
        --epochs 100 --batch-size 8 --lr 1e-4 --efficient

Inferencia con el checkpoint resultante: el kernel dipolar NO se guarda (es
dependiente de los datos), asi que hay que reconstruir el config e inyectarlo:
    ck = torch.load('checkpoints/tdv-qsm-color.pth')          # weights_only=True OK
    cfg = ck['config']
    cfg['D']['config']['dipole_kernel'] = D_2d                # kernel 2D real
    vn = model.VNet(cfg); vn.load_state_dict(ck['model'], strict=False)
"""

import os
import sys
import copy
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Permite que el archivo viva en un subdirectorio (e.g. retraining/) y se ejecute
# desde cualquier CWD: agrega al sys.path el directorio que contiene model.py.
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here if os.path.exists(os.path.join(_here, 'model.py')) else os.path.dirname(_here)
for _p in (_here, _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ddr import model
# pyrefly: ignore [missing-import]
from dataset_qsm import QSMDataset



# --------------------------------------------------------------------------- #
#  Configuracion de la VNet para QSM (a partir del checkpoint de denoising)
# --------------------------------------------------------------------------- #
def build_qsm_config(base_config, args):
    """Toma el config del checkpoint de denoising y lo adapta a QSM.

    - Cambia el dataterm a 'qsm' con use_prox=False (porque usamos pesos W, que
      rompen la solucion cerrada en Fourier => usamos descenso por gradiente).
    - Reinicializa los escalares T y lambda a valores estables.
    El kernel y los pesos del dataterm se inyectan por-batch en el loop, por lo
    que aqui se pasa un kernel placeholder (sera sobreescrito).
    """
    config = dict(base_config)  # copia superficial; mutamos subdiccionarios abajo

    # Dataterm fisico de QSM. dipole_kernel placeholder: lo inyectamos por batch.
    config['D'] = {
        'type': 'qsm',
        'config': {
            'use_prox': False,
            'dipole_kernel': np.zeros((1, 1), dtype=np.float32),  # placeholder
            'weight': None,                                       # inyectado por batch
        },
    }

    # Escalares: aprendibles, reinicializados a valores estables (NO del checkpoint).
    config['T_mode'] = 'learned'
    config['T'] = {'init': args.t_init, 'min': 0.0, 'max': 1000.0}
    config['lambda_mode'] = 'learned'
    config['lambda'] = {'init': args.lambda_init, 'min': 0.0, 'max': 1000.0}

    return config


def load_regularizer_weights(vn, base_ckpt):
    """Transfer learning: carga UNICAMENTE los pesos del regularizador R (TDV).

    Extrae del state_dict del checkpoint las claves con prefijo 'R.' y las carga
    en vn.R. Los escalares T y lambda quedan en sus valores reinicializados.
    """
    full_sd = base_ckpt['model']
    r_sd = {k[len('R.'):]: v for k, v in full_sd.items() if k.startswith('R.')}
    missing, unexpected = vn.R.load_state_dict(r_sd, strict=False)
    if unexpected:
        raise RuntimeError(f'Claves inesperadas al cargar R: {unexpected}')
    # 'mask' son buffers fijos; cualquier missing real seria un problema.
    real_missing = [m for m in missing if not m.endswith('mask')]
    if real_missing:
        raise RuntimeError(f'Faltan pesos del regularizador R: {real_missing}')
    return len(r_sd)


# --------------------------------------------------------------------------- #
#  Inyeccion del kernel/pesos por batch + forward de la VNet
# --------------------------------------------------------------------------- #
def set_dataterm(vn, kernel, weight):
    """Inyecta el kernel dipolar y los pesos de magnitud del batch en el
    QSMDataterm, reemplazando sus buffers fijos.

    Args:
        kernel: (B, H, W) kernel dipolar 2D en k-space por muestra.
        weight: (B, C, H, W) pesos de magnitud W (no W^2), o None.

    El forward del dataterm hace  F^-1{ D_k . F{x} }  con fft2 sobre los dos
    ultimos ejes; D_k de forma (B, 1, H, W) hace broadcast sobre los C canales.
    W2 de forma (B, C, H, W) multiplica el residuo por-canal/por-muestra.
    """
    D = kernel.unsqueeze(1)            # (B, 1, H, W)
    vn.D.D = D
    vn.D.D2 = D * D
    vn.D.W2 = (weight * weight) if weight is not None else None


def vnet_predict(vn, phase, kernel, weight):
    """Corre la VNet y devuelve la estimacion final de chi: (B, C, H, W)."""
    set_dataterm(vn, kernel, weight)
    x0 = torch.zeros_like(phase)          # inicializacion en cero
    x_all = vn(x0, phase)                 # (S+1, B, C, H, W)
    return x_all[-1]                       # ultimo paso de la VNet


def apply_tdv_projections(vn):
    """Restricciones del TDV: filtros zero-mean / norma acotada y escalares
    positivos. Se ejecuta tras cada optimizer.step()."""
    with torch.no_grad():
        for p in vn.parameters():
            if hasattr(p, 'proj'):
                p.proj()


# --------------------------------------------------------------------------- #
#  Loss
# --------------------------------------------------------------------------- #
def qsm_loss(x_pred, chi_gt, center_only=True):
    """MSE entre la salida de la VNet y el ground truth de susceptibilidad.

    En modo 2.5D los canales son slices (z-1, z, z+1); por defecto la perdida se
    concentra en el slice central (canal 1), que es el unico con contexto de
    vecindad completo en ambos lados.
    """
    if center_only:
        c = x_pred.shape[1] // 2
        return torch.nn.functional.mse_loss(x_pred[:, c], chi_gt[:, c])
    return torch.nn.functional.mse_loss(x_pred, chi_gt)


# --------------------------------------------------------------------------- #
#  Loops de entrenamiento / validacion
# --------------------------------------------------------------------------- #
def run_epoch(vn, loader, device, optimizer=None, center_only=True):
    """Una epoca. Si optimizer es None, corre en modo evaluacion (sin grad)."""
    train = optimizer is not None
    vn.train(train)
    total, n = 0.0, 0

    for batch in loader:
        phase = batch['phase'].to(device, non_blocking=True)
        chi_gt = batch['chi_gt'].to(device, non_blocking=True)
        weight = batch['weight'].to(device, non_blocking=True)
        kernel = batch['kernel'].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            x_pred = vnet_predict(vn, phase, kernel, weight)
            loss = qsm_loss(x_pred, chi_gt, center_only=center_only)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            apply_tdv_projections(vn)       # restricciones del TDV tras el update

        bs = phase.shape[0]
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)


def save_checkpoint(path, vn, config, epoch, val_loss):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # Filtramos los buffers del dataterm (prefijo 'D.'): contienen el kernel/W
    # del ultimo batch (shape (B,1,H,W)) y se reinyectan en inferencia. Guardar
    # solo R.* y los escalares T/lambda mantiene el checkpoint limpio y cargable.
    state = {k: v for k, v in vn.state_dict().items() if not k.startswith('D.')}
    # Config numpy-free: el placeholder del kernel es un np.ndarray y rompe
    # torch.load(weights_only=True) (default en PyTorch >=2.6). Se anula aqui;
    # en inferencia hay que setear config['D']['config']['dipole_kernel'] con el
    # kernel real (2D) antes de construir la VNet.
    clean_config = copy.deepcopy(config)
    clean_config['D']['config']['dipole_kernel'] = None
    clean_config['D']['config']['weight'] = None
    torch.save(
        {
            'config': clean_config,
            'model': state,
            'epoch': epoch,
            'val_loss': val_loss,
        },
        path,
    )


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description='Reentrenamiento de TDV-VNet para QSM.')
    # Datos
    p.add_argument('--train-data', required=True,
                   help='Carpeta (o archivo) con volumenes de entrenamiento.')
    p.add_argument('--val-data', default=None,
                   help='Carpeta con volumenes de validacion. Si se omite, se '
                        'separa --val-split del set de entrenamiento.')
    p.add_argument('--val-split', type=float, default=0.1,
                   help='Fraccion de validacion si no se da --val-data.')
    p.add_argument('--pattern', default='*.mat', help='Glob de archivos.')
    p.add_argument('--scale', type=float, default=1.0,
                   help='Factor lineal aplicado a fase y chi_gt (normalizacion).')
    p.add_argument('--kernel-method', default='mean', choices=['mean', 'central'],
                   help="Proyeccion 3D->2D del kernel dipolar.")
    # Checkpoints (por defecto relativos al root del proyecto, no al CWD)
    p.add_argument('--base-ckpt',
                   default=os.path.join(_root, 'checkpoints', 'tdv3-3-25-f32-color.pth'),
                   help='Checkpoint de denoising para transfer learning de R.')
    p.add_argument('--out-ckpt',
                   default=os.path.join(_root, 'checkpoints', 'tdv-qsm-color.pth'),
                   help='Ruta de salida del checkpoint reentrenado (mejor val).')
    # Optimizacion
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--t-init', type=float, default=0.01,
                   help='Valor inicial estable del stopping time T.')
    p.add_argument('--lambda-init', type=float, default=1.0,
                   help='Valor inicial estable del balance lambda (R vs D).')
    p.add_argument('--center-only', action='store_true', default=True,
                   help='Calcular la perdida solo en el slice central.')
    p.add_argument('--all-slices', dest='center_only', action='store_false',
                   help='Calcular la perdida en los 3 canales.')
    # Sistema
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--no-cache', dest='cache', action='store_false', default=True,
                   help='No cachear volumenes en RAM (util si el dataset es grande '
                        'o se usan muchos workers).')
    p.add_argument('--efficient', action='store_true',
                   help='Gradient checkpointing en R.grad para ahorrar VRAM.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # --- Datasets -------------------------------------------------------- #
    full_train = QSMDataset(
        args.train_data, pattern=args.pattern, scale=args.scale,
        kernel_method=args.kernel_method, cache=args.cache,
    )
    if args.val_data is not None:
        train_ds = full_train
        val_ds = QSMDataset(
            args.val_data, pattern=args.pattern, scale=args.scale,
            kernel_method=args.kernel_method, cache=args.cache,
        )
    else:
        n_val = max(1, int(round(len(full_train) * args.val_split)))
        n_train = len(full_train) - n_val
        gen = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == 'cuda'), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == 'cuda'),
    )
    print(f'Train samples: {len(train_ds)} | Val samples: {len(val_ds)}')

    # --- Modelo + transfer learning ------------------------------------- #
    base_ckpt = torch.load(args.base_ckpt, map_location='cpu')
    config = build_qsm_config(base_ckpt['config'], args)

    vn = model.VNet(config, efficient=args.efficient)
    n_loaded = load_regularizer_weights(vn, base_ckpt)
    vn.to(device)
    print(f'Transfer learning: {n_loaded} tensores de R (TDV) cargados desde '
          f'{os.path.basename(args.base_ckpt)}. T={args.t_init}, lambda={args.lambda_init}.')

    optimizer = torch.optim.Adam(
        vn.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # --- Loop de epocas -------------------------------------------------- #
    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(vn, train_loader, device, optimizer,
                               center_only=args.center_only)
        val_loss = run_epoch(vn, val_loader, device, optimizer=None,
                             center_only=args.center_only)
        dt = time.time() - t0

        improved = val_loss < best_val
        flag = '  *mejor*' if improved else ''
        print(f'[{epoch:3d}/{args.epochs}] train={train_loss:.6e} '
              f'val={val_loss:.6e} ({dt:.1f}s)'
              f'  T={float(vn.T.detach()):.4g} lambda={float(vn.lmbda.detach()):.4g}{flag}')

        if improved:
            best_val = val_loss
            save_checkpoint(args.out_ckpt, vn, config, epoch, val_loss)

    print(f'Listo. Mejor val_loss={best_val:.6e}. Checkpoint en {args.out_ckpt}.')


if __name__ == '__main__':
    main()
