"""Enable `python -m mata` as an alias for the `mata` CLI entry point."""
from mata.cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
