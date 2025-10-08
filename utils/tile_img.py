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
    for y in range(0, img_padded.shape[0], tile_size):
        for x in range(0, img_padded.shape[1], tile_size):
            tile = img_padded[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)

    print(f"\tNum. tiles: {len(tiles)}")
    print(f"\tTile shape:{tiles[150].shape}")

    return tiles
