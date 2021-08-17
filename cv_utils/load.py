from urllib import request

import numpy as np
import cv2


def cv2_img_from_url(url: str) -> np.ndarray:
    req = request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img