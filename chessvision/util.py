
# =====================
# ENVIRONMENT VARIABLES
# =====================

STORAGE_ENV = "CV_DIR"


# =================
# UTILITY FUNCTIONS
# =================

from itertools import groupby
def count(collection): 
    for key, group in groupby(collection): 
        yield key, sum(1 for _ in group)


def skip(iterator, n):
    for _ in range(n):
        next(iterator)
    return iterator


from functools import reduce
from operator import mul
def mult(iterator):
    return reduce(mul, iterator)