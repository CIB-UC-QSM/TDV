import numpy.typing as npt
import matplotlib.pyplot as plt

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
):
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
    plt.show()
