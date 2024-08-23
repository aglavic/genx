from genx.version import __version__

mv = int(__version__.split(".")[1])

RELEASE_HEAD = """
Changes since previous release
==============================
"""

ALPHA_NOTE = """
CAUTION
=======
This is an alpha release version. It may provide new functionality but likely contains new bugs
that may break important functionality. Please don't use it in production environments.

Please be sure to report any unexpected behavior and issues as
[ticket on SourceForge](https://sourceforge.net/p/genx/tickets/) or 
[via e-mail](mailto:artur.glavic@psi.ch).
"""

BETA_NOTE = """
NOTE
====
This is a beta release version. It may provide new functionality but can contain new bugs
that could break some functionality.

Please be sure to report any unexpected behavior and issues as
[ticket on SourceForge](https://sourceforge.net/p/genx/tickets/) or 
[via e-mail](mailto:artur.glavic@psi.ch).
"""

with open("release_notes.md", "w") as fh:
    found_first = 0
    if "a" in __version__:
        fh.write(ALPHA_NOTE)
    elif "b" in __version__:
        fh.write(BETA_NOTE)
    fh.write(RELEASE_HEAD)
    for l in open("README.txt", "r").readlines():
        if f"Changes 3." in l:
            if found_first:
                break
            else:
                found_first = 3
        if found_first == 1:
            fh.write(l)
        elif found_first:
            found_first -= 1
