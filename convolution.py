"""2D image convolution from scratch, used for edge detection and blurring."""

import numpy as np
import matplotlib.pyplot as plt


def pad_with_zeros(image: np.ndarray, pad: int) -> np.ndarray:
    """Wrap the image in a border of zero-valued pixels.

    Without a border the kernel can't be centred on the edge pixels, so the
    output would come out smaller than the input. Padding keeps the sizes equal.
    """
    height, width = image.shape
    padded = np.zeros((height + 2 * pad, width + 2 * pad))
    padded[pad:pad + height, pad:pad + width] = image
    return padded


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide a square kernel over a grayscale image and return the result.

    For each pixel the kernel is lined up on the surrounding neighbourhood,
    multiplied element-wise, and summed into a single output value.
    """
    kernel = np.flipud(np.fliplr(kernel))   # a true convolution flips the kernel
    k = kernel.shape[0]
    pad = k // 2
    padded = pad_with_zeros(image, pad)

    output = np.zeros_like(image, dtype=float)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y + k, x:x + k]
            output[y, x] = np.sum(region * kernel)
    return output


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a (size x size) Gaussian blur kernel that sums to 1."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def main() -> None:
    # A simple test image: a bright square sitting on a dark background.
    image = np.zeros((100, 100))
    image[30:70, 30:70] = 1.0

    # Sobel kernels respond to intensity changes in x and y.
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = sobel_x.T

    edges_x = convolve(image, sobel_x)
    edges_y = convolve(image, sobel_y)
    edges = np.sqrt(edges_x ** 2 + edges_y ** 2)   # combined edge strength

    blurred = convolve(image, gaussian_kernel(size=5, sigma=1.0))

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, data, title in zip(
        axes, [image, edges, blurred], ["Original", "Edges (Sobel)", "Blurred (Gaussian)"]
    ):
        ax.imshow(data, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("convolution_demo.png", dpi=120)
    print("Edges range:", round(edges.min(), 2), "to", round(edges.max(), 2))
    print("Saved demo image.")


if __name__ == "__main__":
    main()