import warnings
import numpy as np

# qpython uses deprecated np.bool; patch before it's imported
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = np.bool_
