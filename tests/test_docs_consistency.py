"""Keep the package version, CHANGELOG and README in sync.

Every assertion here corresponds to a real drift found in the v2.2.0 docs:
the README still advertised v2.1.0 as current and a 1.9.7 version badge.
"""

from __future__ import annotations

import re
from pathlib import Path

import mata

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
README = REPO_ROOT / "README.md"

_RELEASE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] — (\d{4}-\d{2}-\d{2})$", re.MULTILINE)


def _released_versions() -> list[str]:
    return _RELEASE.findall(CHANGELOG.read_text(encoding="utf-8"))


def test_changelog_head_matches_package_version():
    versions = [version for version, _ in _released_versions()]
    assert versions, "CHANGELOG.md has no '## [X.Y.Z] — YYYY-MM-DD' headings"
    assert versions[0] == mata.__version__, (
        f"CHANGELOG's newest release is {versions[0]}, package is {mata.__version__}"
    )


def test_readme_roadmap_marks_current_version():
    readme = README.read_text(encoding="utf-8")
    assert f"- **v{mata.__version__}** (current)" in readme, (
        "README roadmap does not mark the installed version as current"
    )


def test_version_literal_is_declared_once():
    """Catches re-introducing a second hardcoded copy of the version."""
    literal = f'__version__ = "{mata.__version__}"'
    hits = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src" / "mata").rglob("*.py")
        if literal in path.read_text(encoding="utf-8")
    ]
    assert hits == ["src/mata/_version.py"], hits


def test_pyproject_version_is_dynamic():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text, (
        "pyproject.toml must read the version from mata._version, not hardcode it"
    )
