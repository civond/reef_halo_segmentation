import rasterio

def load_tif(tif_pth=None):

    if tif_pth != None:
        print(f"\tLoading {tif_pth}")
        with rasterio.open(tif_pth) as src:
            img = src.read()
            profile = src.profile  

        return img, profile

    else:
        print("invalid .tif file")
        pass