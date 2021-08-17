import numpy as np


def adapt_to_dims(f):
    """
    Many bbox and polygon utilities should be able to work with a single input
    (1D) or with multiple (2D). Using this decorator we can make 1D inputs 
    2D then flatten the result before returning it.
    The decorated function should take bboxes/polys as the first argument and
    must also return bboxes/polys.
    """
    def wrapper(*args, **kwargs):
        inp, args = args[0], args[1:]
        is_1d = False
        if len(inp.shape) == 1:
            is_1d = True
            inp = np.expand_dims(inp, axis=0)
        out = f(inp, *args, **kwargs)
        if is_1d:
            out = out.flatten()
        return out
    return wrapper