import numpy as np

# Perform percentile-based contrast stretching of a single RGB band
def percentile_stretch(band):
    band = band.astype(np.float32)

    p2 = np.nanpercentile(band, 2)              # 2% intensity
    p98 = np.nanpercentile(band, 98)            # 98% intensity
    dyanmic_range = p98 - p2                    # Dynamic intensity range

    # If dynamic range is not valid, return 0
    if dyanmic_range < 1e-6:
        return np.zeros_like(band)

    # Generate clipped band
    band = (band - p2) / dyanmic_range
    clipped = np.clip(band, 0, 1)

    return clipped