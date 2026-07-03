"""
Reentrenamiento end-to-end de la TDV-VNet para QSM.
Optimizaciones aplicadas:
- DDP (DistributedDataParallel) para eliminar cuello de botella PCIe.
- Prefetch agresivo en RAM y sinconización perezosa.
- torch.compile para fusion de kernels Triton (Ada Lovelace).
- AMP (Automatic Mixed Precision) para doble velocidad en TF32/FP16.
"""

import os
# Limitar subprocesos de CPU para evitar colapso de "thread contention" durante las FFT 3D en workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import copy
import argparse
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here if os.path.exists(os.path.join(_here, 'model.py')) else os.path.dirname(_here)
for _p in (_here, _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ddr import model
# pyrefly: ignore [missing-import]
from dataset_qsm import QSMDataset

def build_qsm_config(base_config, args):
    config = dict(base_config)
    config['D'] = {
        'type': 'qsm',
        'config': {
            'use_prox': False,
            'dipole_kernel': np.zeros((1, 1), dtype=np.float32),
            'weight': None,
        },
    }
    config['T_mode'] = 'learned'
    config['T'] = {'init': args.t_init, 'min': 0.0, 'max': 1000.0}
    config['lambda_mode'] = 'learned'
    config['lambda'] = {'init': args.lambda_init, 'min': 0.0, 'max': 1000.0}
    return config

def load_regularizer_weights(vn, base_ckpt):
    full_sd = base_ckpt['model']
    r_sd = {k[len('R.'):]: v for k, v in full_sd.items() if k.startswith('R.')}
    missing, unexpected = vn.R.load_state_dict(r_sd, strict=False)
    return len(r_sd)

class VNetWrapper(torch.nn.Module):
    """Envuelve la VNet para que devuelva solo el último paso (B, C, H, W)."""
    def __init__(self, vn):
        super().__init__()
        self.vn = vn
        
    def forward(self, phase, kernel, weight):
        x0 = torch.zeros_like(phase)
        x_all = self.vn(x0, phase, kernel=kernel, weight=weight)
        return x_all[-1]

def apply_tdv_projections(raw_vn):
    with torch.no_grad():
        for p in raw_vn.parameters():
            if hasattr(p, 'proj'):
                p.proj()

def qsm_loss(x_pred, chi_gt, center_only=True, mask=None):
    if mask is not None:
        if center_only:
            c = x_pred.shape[1] // 2
            diff = x_pred[:, c] - chi_gt[:, c]
            m = mask[:, c]
            return torch.sum((diff ** 2) * m) / torch.clamp(torch.sum(m), min=1.0)
        else:
            diff = x_pred - chi_gt
            return torch.sum((diff ** 2) * mask) / torch.clamp(torch.sum(mask), min=1.0)

    if center_only:
        c = x_pred.shape[1] // 2
        return torch.nn.functional.mse_loss(x_pred[:, c], chi_gt[:, c])
    return torch.nn.functional.mse_loss(x_pred, chi_gt)

def run_epoch(wrapped_vn, raw_vn, loader, device, optimizer=None, scaler=None, center_only=True, mini_batch_size=8, slices_per_volume=100, is_main=True):
    train = optimizer is not None
    wrapped_vn.train(train)
    
    # Acumuladores en GPU para evitar cuellos de botella CPU-GPU por loss.item()
    total_loss = torch.tensor(0.0, device=device)
    total_n = torch.tensor(0.0, device=device)

    for batch_idx, batch in enumerate(loader):
        t_vol = time.time()
        B, Nz, Nx, Ny = batch['phase'].shape
        if Nz < 3: continue
            
        v_phase = batch['phase'].to(device, non_blocking=True)
        v_chi = batch['chi_gt'].to(device, non_blocking=True)
        v_weight = batch['weight'].to(device, non_blocking=True)
        v_mask = batch['mask'].to(device, non_blocking=True)
        v_kernel = batch['kernel'].to(device, non_blocking=True)
            
        # Add channel dimension to all tensors (B, 1, Z, H, W)
        v_phase = v_phase.unsqueeze(1)
        v_chi = v_chi.unsqueeze(1)
        v_weight = v_weight.unsqueeze(1)
        v_mask = v_mask.unsqueeze(1)
        v_kernel = v_kernel.unsqueeze(1)
        
        # 1. Filtrar los cortes extremos (fondo) verificando si hay cerebro en la mascara
        brain_pixels = v_mask[:, 0].sum(dim=(2, 3)) # sum over H, W for each Z
        valid_indices = torch.where(brain_pixels[0] > 1000)[0]
        
        if len(valid_indices) == 0:
            valid_indices = torch.arange(Nz, device=device)
            
        num_valid = len(valid_indices)
        
        # 2. [CRITICAL DDP FIX]: Acordar la cantidad de cortes validos mas pequeña entre ambas GPUs
        local_trips = torch.tensor([num_valid], device=device, dtype=torch.long)
        dist.all_reduce(local_trips, op=dist.ReduceOp.MIN)
        safe_num_slices = local_trips.item()
        
        # 3. Limitar a N cortes maximo por volumen para acelerar el entrenamiento (ej. 100 cortes)
        if safe_num_slices > slices_per_volume:
            safe_num_slices = slices_per_volume
            
        if train:
            # En entrenamiento, elegimos un bloque continuo de cortes para preservar la fisica 3D
            max_start = valid_indices[-1] - safe_num_slices + 1
            min_start = valid_indices[0]
            if max_start > min_start:
                start_z = torch.randint(min_start.item(), max_start.item() + 1, (1,)).item()
            else:
                start_z = min_start.item()
        else:
            # En validacion, elegimos cortes secuenciales del centro del cerebro
            start_z = valid_indices[0].item() + (num_valid - safe_num_slices) // 2
            
        end_z = start_z + safe_num_slices
        
        phase = v_phase[:, :, start_z:end_z].contiguous()
        chi_gt = v_chi[:, :, start_z:end_z].contiguous()
        weight = v_weight[:, :, start_z:end_z].contiguous()
        mask = v_mask[:, :, start_z:end_z].contiguous()
        kernel = v_kernel[:, :, start_z:end_z].contiguous()

        # AMP reactivado, optoth protegido con fp32 internamente
        with torch.amp.autocast('cuda', enabled=scaler.is_enabled() if scaler is not None else False):
            x_pred = wrapped_vn(phase, kernel, weight)
            # Para 3D no usamos center_only porque procesamos el volumen entero
            loss = qsm_loss(x_pred, chi_gt, center_only=False, mask=mask)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            apply_tdv_projections(raw_vn)

        bs = phase.shape[0] * phase.shape[2] # Batch * Z slices
        total_loss += loss.detach().float() * bs
        total_n += bs

        if is_main:
            print(f'  [Vol {batch_idx}] procesado en {time.time() - t_vol:.2f}s', flush=True)

    # Reducir metricas cruzando todas las GPUs de DDP
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_n, op=dist.ReduceOp.SUM)

    return (total_loss / torch.clamp(total_n, min=1.0)).item()

def save_checkpoint(path, vn, config, epoch, val_loss):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    model_to_save = vn.module if hasattr(vn, 'module') else vn
    state = {k: v for k, v in model_to_save.state_dict().items() if not k.startswith('D.')}
    clean_config = copy.deepcopy(config)
    clean_config['D']['config']['dipole_kernel'] = None
    clean_config['D']['config']['weight'] = None
    torch.save({'config': clean_config, 'model': state, 'epoch': epoch, 'val_loss': val_loss}, path)

def parse_args():
    p = argparse.ArgumentParser(description='Reentrenamiento optimizado de TDV-VNet QSM con DDP + AMP + Compile.')
    p.add_argument('--train-data', required=True)
    p.add_argument('--val-data', default=None)
    p.add_argument('--val-split', type=float, default=0.1)
    p.add_argument('--pattern', default='**/*_T1w.nii.gz')
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--kernel-method', default='mean', choices=['mean', 'central'])
    p.add_argument('--limit', type=int, default=500, help='Limitar dataset a 500 sujetos (50k cortes)')
    p.add_argument('--slices-per-volume', type=int, default=100, help='Cortes aleatorios por volumen')
    p.add_argument('--base-ckpt', default=os.path.join(_root, 'checkpoints', 'tdv3-3-25-f32-color.pth'))
    p.add_argument('--out-ckpt', default=os.path.join(_root, 'checkpoints', 'tdv-qsm-color.pth'))
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--volume-batch', type=int, default=1, help='Volumenes cacheados por worker')
    p.add_argument('--batch-size', type=int, default=16, help='Mini-batch de slices 2.5D')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--t-init', type=float, default=0.01)
    p.add_argument('--lambda-init', type=float, default=1.0)
    p.add_argument('--center-only', action='store_true', default=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--efficient', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    is_main = (local_rank == 0)

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    # Optimizacion masiva para la GPU RTX 5070 Ti (Ada Lovelace) y CUDNN
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    full_train = QSMDataset(args.train_data, pattern=args.pattern, scale=args.scale, kernel_method=args.kernel_method, limit=args.limit)
    if args.val_data is not None:
        train_ds = full_train
        val_ds = QSMDataset(args.val_data, pattern=args.pattern, scale=args.scale, kernel_method=args.kernel_method, limit=args.limit)
    else:
        n_val = max(1, int(round(len(full_train) * args.val_split)))
        train_ds, val_ds = random_split(full_train, [len(full_train) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    num_workers_per_gpu = max(1, args.workers // world_size)
    prefetch_factor = 4 if num_workers_per_gpu > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=args.volume_batch, sampler=train_sampler,
        num_workers=num_workers_per_gpu, pin_memory=True, drop_last=False,
        prefetch_factor=prefetch_factor, persistent_workers=True if num_workers_per_gpu > 0 else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.volume_batch, sampler=val_sampler,
        num_workers=num_workers_per_gpu, pin_memory=True, drop_last=False,
        prefetch_factor=prefetch_factor, persistent_workers=True if num_workers_per_gpu > 0 else False
    )
    
    if is_main:
        print(f'Configuracion DDP lista! GPUs: {world_size} | Workers/GPU: {num_workers_per_gpu} | Prefetch: {prefetch_factor}', flush=True)

    base_ckpt = torch.load(args.base_ckpt, map_location='cpu')
    config = build_qsm_config(base_ckpt['config'], args)

    raw_vn = model.VNet(config, efficient=args.efficient)
    load_regularizer_weights(raw_vn, base_ckpt)
    raw_vn.to(device)
    
    wrapped_vn = VNetWrapper(raw_vn)
    wrapped_vn.to(device)
    # [REMOVIDO] torch.compile choca con extensiones C++ opacas como optoth (genera endless recompilation)
    
    wrapped_vn = torch.nn.parallel.DistributedDataParallel(wrapped_vn, device_ids=[local_rank])
    
    optimizer = torch.optim.Adam(raw_vn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # [NUEVO] Scaler para Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_val = float('inf')
    local_batch_size = max(1, args.batch_size // world_size)
    
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        t0 = time.time()
        
        train_loss = run_epoch(wrapped_vn, raw_vn, train_loader, device, optimizer, scaler=scaler, center_only=args.center_only, mini_batch_size=local_batch_size, is_main=is_main)
        val_loss = run_epoch(wrapped_vn, raw_vn, val_loader, device, optimizer=None, scaler=scaler, center_only=args.center_only, mini_batch_size=local_batch_size, is_main=is_main)
        
        torch.cuda.synchronize()
        dt = time.time() - t0

        if is_main:
            improved = val_loss < best_val
            flag = '  *mejor*' if improved else ''
            print(f'[{epoch:3d}/{args.epochs}] train={train_loss:.6e} val={val_loss:.6e} ({dt:.1f}s)  '
                  f'T={float(raw_vn.T.detach()):.4g} lambda={float(raw_vn.lmbda.detach()):.4g}{flag}', flush=True)

            if improved:
                best_val = val_loss
                save_checkpoint(args.out_ckpt, raw_vn, config, epoch, val_loss)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
