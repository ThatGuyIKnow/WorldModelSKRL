
def prefix_keys(d: dict, prefix: str) -> dict:
    return {prefix+k: v for k, v in d.items()}

def transpose_2d(l: list) -> list:
    return list(zip(*l))