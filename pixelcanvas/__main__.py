"""python -m pixelcanvas <command> ...: the same as the pixelcanvas command."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
