
# =====================
# ENVIRONMENT VARIABLES
# =====================

STORAGE_ENV = "CV_DIR"


# =================
# UTILITY FUNCTIONS
# =================

from itertools import groupby
def count(collection):
    """ Count consecutive items in a collection

    Returns: Generator[element, run length]

    Example:
        >>> arr = [1, 1, 2, 1, 5, 5, 5, 5]
        >>> list(count(arr))
        [(1, 2), (2, 1), (1, 1), (5, 4)]
    """
    for key, group in groupby(collection): 
        yield key, sum(1 for _ in group)


def skip(iterator, n):
    """ Calls next on the iterator n times
    Returns: the resulting iterator
    """
    for _ in range(n):
        next(iterator)
    return iterator


from functools import reduce
from operator import mul
def mult(iterator):
    """ Similar to built-ins `sum` and `max` for multiplication
    
    Example:
        >>> arr = [4, 3, 2, 1]
        >>> mult(arr)
        24
    """
    return reduce(mul, iterator)