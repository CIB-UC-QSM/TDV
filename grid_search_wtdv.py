"""
Grid search de mu y target_std para wTDV_QSM_torch.py. Corre el mismo ADMM con cada combinación y
guarda las curvas de Update y RMSE por iteración en un JSON; el tolUpdate se elige después sobre
esas curvas, simulando dónde cortaría el script (--resumen). El Update que decide es el de dentro
del cerebro, como en el script; --criterio total usa el de todo el volumen (el criterio anterior).
Como el script, enmascara la entrada y la salida del TDV en cada iteración y mide la RMSE sobre el mapa
enmascarado. Los checkpoints son de 3 canales: tripletes de cortes adyacentes como RGB.

    uv run grid_search_wtdv.py --checkpoint t2delta-100k-s1.pth --out results/grid/t2delta.json
    CUDA_VISIBLE_DEVICES=1 uv run grid_search_wtdv.py --mus 0.03 0.06 --out results/grid/oficial_b.json
    uv run grid_search_wtdv.py --resumen results/grid/t2delta.json
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.io
import torch
from scipy import ndimage

from ddr import model
from ddr.utils import rmse

MU2 = 1.0
TOLS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.35, 0.39, 0.45, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0)


def cargar_datos(device):
    """Mismos datos, máscara y pesos W^2 que wTDV_QSM_torch.py."""
    params = scipy.io.loadmat('data_mat/params.mat')
    phase = params['phase_use'].astype(np.float32)
    kernel = params['kernel'].astype(np.float32)
    weight = params['magn_use'].astype(np.float32)
    chi = scipy.io.loadmat('data_mat/chi_cosmos.mat')['chi_cosmos'].astype(np.float32)

    brain_solid = ndimage.binary_fill_holes(weight > 0.05)
    brain_clean = brain_solid & (weight > 0.05 * np.mean(weight[brain_solid]))
    brain_closed = ndimage.binary_closing(brain_clean, structure=np.ones((3, 3, 3)), iterations=2)
    brain_final = ndimage.binary_fill_holes(brain_closed)
    mask = ndimage.gaussian_filter(brain_final.astype(np.float32), sigma=1.0) > 0.5
    weight = weight * mask
    weight /= weight[mask].mean()
    weight *= weight

    gpu = lambda a: torch.from_numpy(a).to(device)
    return {'kernel': gpu(kernel), 'K2': gpu(kernel * kernel), 'weight': gpu(weight),
            'Wy': gpu(weight * phase), 'mask': gpu(mask), 'chi': chi, 'chi_mask': chi != 0}


def cargar_modelo(nombre, device, compilar, steps=None, t_scale=1.0):
    ckpt = torch.load(os.path.join('checkpoints', nombre))
    vn = model.VNet(ckpt['config'], efficient=False)
    vn.load_state_dict(ckpt['model'])
    if steps:
        vn.set_end(steps)  # mismo T y lambda en más pasos: tau = T/S más chico
    with torch.no_grad():
        vn.T.mul_(t_scale)  # T de la VNet escalado al inferir
    vn = vn.to(device).to(memory_format=torch.channels_last).eval()
    C = ckpt['config']['R']['config']['in_channels']
    return (torch.compile(vn, mode='max-autotune') if compilar else vn), C


@torch.no_grad()
def admm(vn, d, C, mu, target_std, max_iter, stop_update, abort_update, batch_size, amp_dtype,
         clave='update_cerebro'):
    """El bucle de wTDV_QSM_torch.py sin prints; corta una iteración después de que el Update de `clave`
    baje de stop_update."""
    kernel, K2, weight, Wy, mask = d['kernel'], d['K2'], d['weight'], d['Wy'], d['mask']
    mask_f = mask.float()
    N = Wy.shape
    num_windows = N[2] - C + 1
    z1_count = torch.zeros_like(Wy)
    for c in range(C):
        z1_count[:, :, c:num_windows + c] += 1

    weight_mu2 = weight + MU2
    denominator = MU2 * K2 + mu
    z1, s1, qsm, s2 = (torch.zeros_like(Wy) for _ in range(4))
    z2 = Wy / weight_mu2
    updates, updates_c, rmses = [], [], []
    for t in range(max_iter):
        qsm_old = qsm
        numerator = MU2 * kernel * torch.fft.fftn(z2 - s2) + mu * torch.fft.fftn(z1 - s1)
        qsm = torch.real(torch.fft.ifftn(numerator / denominator)).to(torch.float32)
        if t == 0:
            alpha = target_std / torch.std(qsm[mask]).item()

        update = 100 * torch.sqrt(torch.mean((qsm - qsm_old) ** 2)) / torch.sqrt(torch.mean(qsm ** 2))
        updates.append(update.item())
        updates_c.append((100 * (qsm - qsm_old)[mask].norm() / qsm[mask].norm()).item())  # sólo el cerebro
        rmses.append(float(rmse((qsm * mask_f).cpu().numpy(), d['chi'], mask=d['chi_mask'])))  # mapa final enmascarado
        criterio = updates_c if clave == 'update_cerebro' else updates
        if (t >= 1 and criterio[-2] < stop_update) or not np.isfinite(updates[-1]):
            break
        if t >= 40 and min(criterio[-10:]) > abort_update:  # no va a llegar a ningún tol de TOLS antes de max_iter
            break

        FhDFx = torch.real(torch.fft.ifftn(kernel * torch.fft.fftn(qsm))).to(torch.float32)
        windows = ((qsm + s1) * mask_f).permute(2, 0, 1).contiguous().unfold(0, C, 1).permute(0, 3, 1, 2)
        z1_accum = torch.zeros_like(Wy)
        for b_start in range(0, num_windows, batch_size):
            b_end = min(b_start + batch_size, num_windows)
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                x = windows[b_start:b_end].contiguous(memory_format=torch.channels_last) * alpha
                outputs = vn(x, x)[-1]
            outputs = (outputs / alpha).permute(0, 2, 3, 1)
            for c in range(C):
                z1_accum[:, :, b_start + c:b_end + c] += outputs[:, :, :, c].permute(1, 2, 0)
        z1 = z1_accum / z1_count * mask_f
        s1 += qsm - z1
        z2 = (Wy + MU2 * (FhDFx + s2)) / weight_mu2
        s2 += FhDFx - z2

    return {'mu': mu, 'target_std': target_std, 'alpha': alpha, 'max_iter': max_iter,
            'update': updates, 'update_cerebro': updates_c, 'rmse': rmses}


def parada(r, tol, clave='update'):
    """(iteraciones, RMSE final) que daría wTDV_QSM_torch.py con este tolUpdate medido sobre `clave`, o None."""
    for t, u in enumerate(r[clave]):
        if u < tol:
            return (t + 2, r['rmse'][t + 1]) if t + 1 < len(r['rmse']) else None
    return (r['max_iter'], r['rmse'][-1]) if len(r['rmse']) == r['max_iter'] else None


def resumen(paths, clave='update'):
    runs = [r for p in paths for r in json.load(open(p))['runs'] if clave in r]
    filas = []
    for r in runs:
        cortes = [(tol, *parada(r, tol, clave)) for tol in TOLS if parada(r, tol, clave) is not None]
        # mejor corte real (el Update bajó de tol); si no hay ninguno, lo que da llegar al tope de iteraciones
        reales = [c for c in cortes if min(r[clave]) < c[0]]
        tope = [(np.inf, *c[1:]) for c in cortes if min(r[clave]) >= c[0]][:1]
        tol, iters, final = min(reales or tope, key=lambda x: x[2]) if cortes else (np.nan, 0, np.inf)
        t_min = int(np.argmin(r['rmse']))
        filas.append((final, r['mu'], r['target_std'], tol, iters, r['rmse'][t_min], t_min,
                      r[clave][-1], len(r[clave])))
    print(f"{'mu':>7} {'std':>6} | {'tol':>5} {'iters':>5} {'RMSE fin':>9} | {'RMSE min':>9} {'@it':>4} | "
          f"{'último Update':>13} {'corridas':>8}")
    for final, mu, std, tol, iters, rmin, tmin, ulast, n in sorted(filas):
        fin = f"{final:>8.2f}%" if np.isfinite(final) else f"{'no conv.':>9}"
        tol = f"{tol:>5.2f}" if np.isfinite(tol) else f"{'tope' if np.isinf(tol) else '-':>5}"
        print(f"{mu:>7.4g} {std:>6.3g} | {tol} {iters:>5d} {fin} | {rmin:>8.2f}% {tmin:>4d} | "
              f"{ulast:>13.3f} {n:>8d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='tdv3-3-25-f32-color.pth', help='.pth de checkpoints/, de 3 canales')
    p.add_argument('--steps', type=int, default=None, help='pasos S de la VNet (default: los del checkpoint)')
    p.add_argument('--t-scale', type=float, default=1.0, help='factor sobre el T aprendido de la VNet')
    p.add_argument('--mus', type=float, nargs='+', default=[0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.12])
    p.add_argument('--stds', type=float, nargs='+', default=[0.02, 0.04, 0.06, 0.1, 0.15, 0.25, 0.4])
    p.add_argument('--max-iter', type=int, default=100)
    p.add_argument('--stop-update', type=float, default=0.1)
    p.add_argument('--abort-update', type=float, default=2.0, help="'inf' corre siempre hasta --max-iter")
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--out', default=None)
    p.add_argument('--resumen', nargs='+', metavar='JSON')
    p.add_argument('--criterio', choices=('cerebro', 'total'), default='cerebro',
                   help='Update que decide el corte: sólo dentro del cerebro (el del script) o todo el volumen')
    args = p.parse_args()
    clave = 'update_cerebro' if args.criterio == 'cerebro' else 'update'
    if args.resumen:
        return resumen(args.resumen, clave)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    batch_size = 256 if torch.cuda.get_device_properties(device).total_memory / 1024**3 >= 11.5 else 64

    d = cargar_datos(device)
    checkpoint = args.checkpoint
    vn, C = cargar_modelo(checkpoint, device, not args.no_compile, args.steps, args.t_scale)
    out = args.out or f'results/grid_search_{checkpoint.removesuffix(".pth")}.json'
    runs = []
    for mu in args.mus:
        for std in args.stds:
            tic = time.time()
            r = admm(vn, d, C, mu, std, args.max_iter, args.stop_update,
                     args.abort_update, batch_size, amp_dtype, clave)
            r['seconds'] = time.time() - tic
            runs.append(r)
            json.dump({'channels': C, 'checkpoint': checkpoint, 'steps': args.steps,
                       't_scale': args.t_scale, 'runs': runs},
                      open(out, 'w'))
            print(f"mu={mu:<7g} std={std:<6g} alpha={r['alpha']:<8.3f} iters={len(r['update']):<4} "
                  f"Update final={r['update'][-1]:<8.3f} RMSE final={r['rmse'][-1]:.2f}% "
                  f"min={min(r['rmse']):.2f}% ({time.time() - tic:.0f} s)", flush=True)


if __name__ == '__main__':
    main()
