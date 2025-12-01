import cv2
import numpy as np


def tile_img(img, tile_size=512, overlap=False):
    """
    Tile an image into fixed-size squares.

    Args:
        img: Input image (H, W, C)
        tile_size: Size of each tile (default 512)
        overlap: If True, generates additional "filling" tiles placed at the
                 center of each 2x2 block of regular tiles to cover seams.

    Returns:
        tiles: List of tile images
        coords: List of (y, x) pixel coordinates for each tile's top-left corner
        dims: Tuple (n_rows, n_cols) of the regular tile grid
    """
    [h, w, c] = img.shape

    # Pad the image such that the dimensions match the tile size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    img_padded = img_padded[:, :, :3]  # Ignore IR channel

    tiles = []
    coords = []  # (y, x) pixel coordinates

    n_rows = img_padded.shape[0] // tile_size
    n_cols = img_padded.shape[1] // tile_size
    dims = (n_rows, n_cols)

    print(f"\tPadded img shape: {img_padded.shape}")
    print(f"\tN rows: {dims[0]}")
    print(f"\tN columns: {dims[1]}")

    # Regular tiles (end-to-end, no overlap)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            y = row_idx * tile_size
            x = col_idx * tile_size
            tile = img_padded[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            coords.append((y, x))

    # Filling tiles (placed at center of each 2x2 block of regular tiles)
    if overlap:
        half = tile_size // 2
        n_fill_rows = n_rows - 1
        n_fill_cols = n_cols - 1
        print(f"\tOverlap enabled: adding {n_fill_rows * n_fill_cols} filling tiles")

        for row_idx in range(n_fill_rows):
            for col_idx in range(n_fill_cols):
                # Offset by half tile_size to center on the 2x2 intersection
                y = row_idx * tile_size + half
                x = col_idx * tile_size + half
                tile = img_padded[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                coords.append((y, x))

    print(f"\tTotal tiles: {len(tiles)}")
    return tiles, coords, dims
