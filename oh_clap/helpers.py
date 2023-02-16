def exists(x):
    return x is not None

def default(x, y):
    return x if exists(x) else y