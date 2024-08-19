from genx.version import __version__

mv = int(__version__.split(".")[1])

with open("README_latest.txt", "w") as fh:
    pc = f"Changes 3.{mv-1}"
    for l in open("README.txt", "r").readlines():
        if pc in l:
            break
        fh.write(l)
