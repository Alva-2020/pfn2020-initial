import importlib
from PIL import Image


def pil_loader(path):
    with Image.open(path) as img:
        return img.convert('RGB')


def accimage_loader(path):
    # Tries to use accimage with PIL fallback.

    is_accimage_avail = importlib.find_loader('accimage')
    if is_accimage_avail:
        import accimage
    else:
        return pil_loader(path)
    try:
        return accimage.Image(path)
    except IOError as e:
        print("WARN: Exception in accimage_loader: {}".format(e))
        return pil_loader(path)
