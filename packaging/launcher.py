"""Entry point for the desktop bundles (PyInstaller)."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from nesicode.desktop import main

    main()
