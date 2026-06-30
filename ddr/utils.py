import numpy as np
import numpy.typing as npt
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import os
import torch

def rotation_by_permutarion(im: npt.NDArray, angle: int) -> npt.NDArray:
    if angle == 90:
        im = im.T
    elif angle == -90:
        im = im.T
        im = im[::-1, :]
        return im
    elif angle == 180:
        im = im[:, :-1]
    return im

def imshow_3d(
    image: npt.NDArray,
    title: str | None = None,
    cmap: str = "gray",
    rango: tuple[float, float] | None = None,
    angles: tuple[int, int, int] | None = None,
    savepath: str | None = None,
):
    """Display or save 3 orthogonal slices of a 3D volume.
    
    Args:
        savepath: If provided, save figure to this path instead of plt.show().
                  Supports .png, .pdf, .svg. Useful for SSH without X11.
    """
    # Use non-interactive backend if saving to file
    if savepath is not None:
        matplotlib.use('Agg')
    
    if rango is None:
        rango = (image.min(), image.max())
    if angles is None:
        angles = (0, 0, 0)

    D, H, W = image.shape

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(
        rotation_by_permutarion(image[D // 2, :, :], angles[0]),
        cmap=cmap,
        aspect="equal",
        vmin=rango[0],
        vmax=rango[1],
    )
    ax1.axis("off")
    ax2.imshow(
        rotation_by_permutarion(image[:, H // 2, :], angles[1]),
        cmap=cmap,
        aspect="equal",
        vmin=rango[0],
        vmax=rango[1],
    )
    ax2.axis("off")
    ax3.imshow(
        rotation_by_permutarion(image[:, :, W // 2], angles[2]),
        cmap=cmap,
        aspect="equal",
        vmin=rango[0],
        vmax=rango[1],
    )
    ax3.axis("off")

    if title:
        fig.suptitle(title, fontsize=32)

    plt.tight_layout()
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath) if os.path.dirname(savepath) else '.', exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure to: {savepath}")
        plt.show()

def print_stats(name, tensor):
    if isinstance(tensor, (int, float)):
        print(f"[DEBUG] {name:<15} | Valor: {tensor:>8.4f}")
    elif torch.is_tensor(tensor) and tensor.numel() == 1:
        print(f"[DEBUG] {name:<15} | Valor: {tensor.item():>8.4f}")
    else:
        print(f"[DEBUG] {name:<15} | Rango: {tensor.min().item():>8.4f} a {tensor.max().item():>8.4f} | Media: {tensor.mean().item():>8.4f} | Std: {tensor.std().item():>8.4f}")

def rmse(
    pred: npt.NDArray[np.floating[Any]],
    gt: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    rmse_val = np.linalg.norm(pred[mask] - gt[mask]) / np.linalg.norm(gt[mask])
    return 100 * rmse_val

def process_axis(v_in, axis_perm, vn, alpha, BATCH_SIZE):
    """
    Applies the TDV network along a specific axis to achieve pseudo-3D regularization.
    """
    v_perm = v_in.permute(*axis_perm)
    D = v_perm.shape[0]
    
    accum = torch.zeros_like(v_perm)
    count = torch.zeros_like(v_perm)
    
    triplets = v_perm.unfold(0, 3, 1).permute(0, 3, 1, 2)
    center_indices = list(range(1, D - 1))
    
    for b_start in range(0, len(center_indices), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(center_indices))
        batch = triplets[b_start:b_end]
        
        with torch.no_grad():
            batch_alpha = batch.contiguous() * alpha
            x_batch = vn(batch_alpha, batch_alpha)
            
        outputs = x_batch[-1] # (B, 3, H, W)
        
        for j, i in enumerate(center_indices[b_start:b_end]):
            accum[i-1:i+2, :, :] += outputs[j] / alpha
            count[i-1:i+2, :, :] += 1
            
        del batch, batch_alpha, x_batch, outputs
        torch.cuda.empty_cache()
            
    inv_perm = [0, 1, 2]
    for i, p in enumerate(axis_perm):
        inv_perm[p] = i
        
    return accum.permute(*inv_perm), count.permute(*inv_perm)
