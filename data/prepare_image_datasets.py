#!/usr/bin/env python3
"""
Prepare Real Image Datasets for NMF Demonstration

Downloads/loads standard test images and prepares them in binary format
for NMF benchmarking. Creates datasets at multiple sizes.

Usage:
    python3 prepare_image_datasets.py --sizes 128 256 512
    python3 prepare_image_datasets.py --local path/to/images/*.png
"""

import argparse
import numpy as np
import os
import sys
from pathlib import Path

try:
    from PIL import Image
    from scipy.io import loadmat
    import urllib.request
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pillow scipy")
    sys.exit(1)


# Standard test image URLs
TEST_IMAGES = {
    'lena': 'https://homepages.cae.wisc.edu/~ece533/images/lena.png',
    'peppers': 'https://homepages.cae.wisc.edu/~ece533/images/peppers.png',
    'cameraman': 'https://homepages.cae.wisc.edu/~ece533/images/cameraman.tif',
}


def download_image(url, output_path):
    """Download image from URL."""
    try:
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  Failed to download: {e}")
        return False


def create_synthetic_image(name, size=(512, 512)):
    """Create synthetic test image if download fails."""
    print(f"  Creating synthetic '{name}' image...")

    if name == 'lena':
        # Radial gradient
        y, x = np.ogrid[-size[0]//2:size[0]//2, -size[1]//2:size[1]//2]
        r = np.sqrt(x*x + y*y)
        img = 255 * (1 - r / r.max())
    elif name == 'peppers':
        # Checkerboard
        block_size = size[0] // 8
        img = np.zeros(size)
        for i in range(0, size[0], block_size):
            for j in range(0, size[1], block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    img[i:i+block_size, j:j+block_size] = 255
    else:  # cameraman
        # Concentric circles
        y, x = np.ogrid[-size[0]//2:size[0]//2, -size[1]//2:size[1]//2]
        r = np.sqrt(x*x + y*y)
        img = 255 * (0.5 + 0.5 * np.sin(r / 20))

    return img.astype(np.uint8)


def load_or_download_images(output_dir, force_download=False):
    """Load test images, downloading if necessary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    images = {}

    for name, url in TEST_IMAGES.items():
        # Determine file extension
        ext = Path(url).suffix
        local_path = output_dir / f"{name}{ext}"

        # Download if needed
        if not local_path.exists() or force_download:
            success = download_image(url, local_path)
            if not success:
                print(f"  Using synthetic image for {name}")
                img_array = create_synthetic_image(name)
                img = Image.fromarray(img_array)
                local_path = output_dir / f"{name}.png"
                img.save(local_path)

        # Load image
        try:
            img = Image.open(local_path)
            # Convert to grayscale
            img_gray = img.convert('L')
            images[name] = np.array(img_gray, dtype=np.float32)
            print(f"  ✓ Loaded {name}: {images[name].shape}")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
            # Create synthetic fallback
            images[name] = create_synthetic_image(name).astype(np.float32)

    return images


def prepare_dataset(images, size, num_images=3):
    """
    Prepare dataset matrix from images.

    Args:
        images: Dict of image arrays
        size: Target size (will be size × size)
        num_images: Number of images to include

    Returns:
        X: Matrix of shape (size*size, num_images)
    """
    img_names = list(images.keys())[:num_images]
    num_pixels = size * size

    X = np.zeros((num_pixels, num_images), dtype=np.float32)

    for i, name in enumerate(img_names):
        # Resize image
        img = images[name]
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_resized = img_pil.resize((size, size), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)

        # Normalize to [0, 1] (NMF requires non-negative)
        img_norm = img_array / 255.0

        # Flatten and store as column
        X[:, i] = img_norm.flatten()

        print(f"    Image {i+1}/{num_images} ({name}): {size}×{size} → {num_pixels} pixels")

    return X, img_names


def save_matrix_binary(filename, X):
    """
    Save matrix in binary format compatible with C++ code.

    Format:
        - 4 bytes: m (rows) as int32
        - 4 bytes: n (cols) as int32
        - m*n*4 bytes: data as float32 (row-major)
    """
    m, n = X.shape

    with open(filename, 'wb') as f:
        # Write dimensions
        f.write(np.array([m, n], dtype=np.int32).tobytes())

        # Write data (row-major, but since we're storing columns, this is already correct)
        f.write(X.astype(np.float32).tobytes())

    print(f"  ✓ Saved: {filename} ({m}×{n} matrix, {X.nbytes/1024:.1f} KB)")


def save_metadata(filename, size, num_images, img_names):
    """Save dataset metadata."""
    with open(filename, 'w') as f:
        f.write(f"Image Size: {size}×{size}\n")
        f.write(f"Matrix Dimensions: {size*size}×{num_images}\n")
        f.write(f"Number of Images: {num_images}\n")
        f.write(f"Total Elements: {size*size*num_images:,}\n")
        f.write(f"\nImages:\n")
        for i, name in enumerate(img_names, 1):
            f.write(f"  {i}. {name}\n")
        f.write(f"\nFormat: Binary (float32)\n")
        f.write(f"Compatible with: ./nmf_*_opt data/images_{size}.bin <rank> <iters>\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare image datasets for NMF')
    parser.add_argument('--sizes', type=int, nargs='+', default=[128, 256, 512],
                        help='Image sizes to generate (default: 128 256 512)')
    parser.add_argument('--num-images', type=int, default=3,
                        help='Number of images per dataset (default: 3)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download of images')

    args = parser.parse_args()

    print("=" * 70)
    print("NMF Image Dataset Preparation")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load/download source images
    print("Step 1: Loading source images...")
    image_dir = output_dir / 'image_originals'
    images = load_or_download_images(image_dir, args.force_download)
    print(f"  Loaded {len(images)} images")
    print()

    # Prepare datasets at each size
    print("Step 2: Preparing datasets...")
    for size in args.sizes:
        print(f"\n  Size: {size}×{size}")
        print(f"  {'─' * 60}")

        # Prepare dataset
        X, img_names = prepare_dataset(images, size, args.num_images)

        # Save binary file
        output_file = output_dir / f"images_{size}.bin"
        save_matrix_binary(output_file, X)

        # Save metadata
        meta_file = output_dir / f"images_{size}_meta.txt"
        save_metadata(meta_file, size, args.num_images, img_names)
        print(f"  ✓ Metadata: {meta_file}")

    print()
    print("=" * 70)
    print("✓ Dataset preparation complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    for size in args.sizes:
        print(f"  • data/images_{size}.bin  ({size}×{size} × {args.num_images} images)")
    print()
    print("Test with:")
    print(f"  ./nmf_memory_opt data/images_{args.sizes[0]}.bin 10 50")
    print()


if __name__ == '__main__':
    main()
