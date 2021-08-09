from glob import glob
from pathlib import Path
from PIL import Image


def img_dir_stats(dir: str, allowed_ext=['png', 'jpg']):
    """
    Gets all images from a directory and returns stats relating to height,
    width, and aspect ratio.
    """
    heights = []
    widths = []
    ars = []
    file_paths = glob(f'{dir}/*')
    file_paths = [fp for fp in file_paths
                  if Path(fp).name.rsplit('.', 1)[-1].lower() in allowed_ext]
    for fp in file_paths:
        with Image.open(fp) as img:
            heights.append(img.height)
            widths.append(img.width)
            ars.append(img.height/img.width)
    return {
        'heights': heights,
        'widths': widths,
        'ars': ars
    }    
