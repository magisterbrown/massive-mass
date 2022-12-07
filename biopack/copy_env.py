import os

def get_vars(pid: int) -> dict:
    var = os.popen(f'cat -v /proc/{pid}/environ').read()
    vals = var.split('^@')
    res = dict()
    for val in vals:
        pair = val.split('=')
        if len(pair)==2:
            res[pair[0]] = pair[1]

    return res

def copy_process(pid: int):

    res = get_vars(pid)
    req = set(res.keys())
    exis = set(dict(os.environ).keys())
    toset = req.difference(exis)
    for seti in toset:
        os.environ[seti] = res[seti]
