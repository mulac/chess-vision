
STORAGE_ENV = "CV_DIR"


from itertools import groupby

def count(collection): 
    for key, group in groupby(collection): 
        yield key, sum(1 for _ in group)

def skip(iterator, n):
    for _ in range(n):
        next(iterator)
    return iterator