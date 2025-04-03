__version__ = "3.7.10"

def increment_version(minor=True, current=None):
    if current is None:
        current = __version__
    import os
    location = os.path.abspath(__file__)
    module_lines = open(location, 'r').readlines()

    full, ver, vminor = map(int, current.split('.'))
    if minor:
        vminor += 1
    else:
        ver += 1
        vminor = 0
    new_version = f"{full}.{ver}.{vminor}"
    for i, line in enumerate(module_lines):
        if line.startswith("__version__"):
            module_lines[i] = f'__version__ = "{new_version}"\n'
    with open(location, 'w') as f:
        f.writelines(module_lines)
