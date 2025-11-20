import cv2
import numpy as np


def tile_img(img, tile_size=512):
    [h,w,c] = img.shape

    # Pad the image such that the dimensions match the tile size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    img_padded = img_padded[:, :, :3] # Ignore IR channel

    # Tile the image
    tiles = []
    coords = []

    n_rows = img_padded.shape[0] // tile_size
    n_cols = img_padded.shape[1] // tile_size
    dims = (n_rows, n_cols)

    print(f"\tPadded img shape: {img_padded.shape}")
    print(f"\tN rows: {dims[0]}")
    print(f"\tN columns: {dims[1]}")

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            y = row_idx * tile_size
            x = col_idx * tile_size
            tile = img_padded[y:y+tile_size, x:x+tile_size]

            # Track grid coords relative to padded img
            tiles.append(tile)
            coords.append((row_idx, col_idx))

    
    """print(f"\tNum. tiles: {len(tiles)}")
    print(f"\tTile shape:{tiles[150][0].shape}")
    print(f"\tTile coords:{tiles[150][1]}")"""

    return tiles, coords, dims
