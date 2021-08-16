from genx.version import __version__

mv=int(__version__.split('.')[1])
found_prev=False

with open('README_latest.txt', 'w') as fh:
    for l in open('README.txt', 'r').readlines():
        if f'Changes 3.{mv-1}' in l:
            if found_prev:
                break
            else:
                found_prev=True
        fh.write(l)
