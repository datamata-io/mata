"""Single source of truth for the MATA package version.

``pyproject.toml`` reads this module statically (``attr:``), so this file must
stay import-free: no dependencies, no side effects, one literal assignment.
"""

__version__ = "2.2.0"
