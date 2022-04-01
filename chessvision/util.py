from itertools import groupby

def count(collection): 
    for key, group in groupby(collection): 
        yield key, sum(1 for _ in group)