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
    assert (
        versions[0] == mata.__version__
    ), f"CHANGELOG's newest release is {versions[0]}, package is {mata.__version__}"


def test_readme_roadmap_marks_current_version():
    readme = README.read_text(encoding="utf-8")
    assert (
        f"- **v{mata.__version__}** (current)" in readme
    ), "README roadmap does not mark the installed version as current"


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
    assert 'dynamic = ["version"]' in text, "pyproject.toml must read the version from mata._version, not hardcode it"


# --- stale version banners ------------------------------------------------

_STAMP = re.compile(
    r"(?m)^(?:> \*\*(?:Version|MATA)\*\*.*\d+\.\d+\.\d+"
    r"|\*\*Version:\*\*.*\d+\.\d+\.\d+"
    r"|_Last updated:.*\d+\.\d+\.\d+"
    r"|\*\*(?:Last Updated|MATA Version):\*\*.*\d+\.\d+)"
)


def _markdown_files():
    yield README
    for path in sorted((REPO_ROOT / "docs").rglob("*.md")):
        if "plans" in path.parts:  # dated research notes keep their own stamps
            continue
        yield path
    for name in ("QUICK_REFERENCE.md", "QUICKSTART.md", "INSTALLATION.md"):
        candidate = REPO_ROOT / name
        if candidate.exists():
            yield candidate


def test_docs_do_not_pin_a_version_stamp():
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _markdown_files()
        if _STAMP.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "These docs pin a version/date banner that goes stale at every release. "
        "Delete the banner — CHANGELOG.md is the only version-stamped document: " + ", ".join(offenders)
    )


# --- documented transformers floors ---------------------------------------

FLOOR_DOC_FILES = (
    "README.md",
    "docs/SUPPORTED_MODELS.md",
    "docs/VLM_MODEL_SUPPORT.md",
)

# Documented on purpose but NOT enforced at load time: the model works on
# older transformers for most inputs, so a hard gate would deny working setups.
UNENFORCED_DOCUMENTED_FLOORS: dict[str, str] = {
    "HuggingFaceTB/SmolVLM-256M-Instruct": "5.21.0",
}

_MODEL_ID = re.compile(r"`([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)`")
_FLOOR = re.compile(r"transformers\s*(?:>=|≥)\s*v?(\d+\.\d+\.\d+)")


def _documented_floors() -> dict[str, set[str]]:
    """Map model id -> declared transformers floors, parsed from table rows."""
    found: dict[str, set[str]] = {}
    for rel in FLOOR_DOC_FILES:
        path = REPO_ROOT / rel
        assert path.exists(), f"{rel} is missing"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("|"):
                continue
            floor = _FLOOR.search(line)
            if floor is None:
                continue
            for model_id in _MODEL_ID.findall(line):
                found.setdefault(model_id, set()).add(floor.group(1))
    return found


def test_enforced_floors_are_documented():
    from mata.adapters.transformers_compat import MIN_TRANSFORMERS

    documented = _documented_floors()
    missing = {
        model_id: floor for model_id, floor in MIN_TRANSFORMERS.items() if floor not in documented.get(model_id, set())
    }
    assert not missing, (
        "Add a table row containing the model id and `transformers>=X` to "
        f"README.md or docs/SUPPORTED_MODELS.md for: {missing}"
    )


def test_documented_floors_exist_in_code():
    from mata.adapters.transformers_compat import MIN_TRANSFORMERS

    known = {**MIN_TRANSFORMERS, **UNENFORCED_DOCUMENTED_FLOORS}
    orphans = {model_id: sorted(floors) for model_id, floors in _documented_floors().items() if model_id not in known}
    assert not orphans, (
        "Docs claim a transformers floor for models that are neither enforced "
        f"in MIN_TRANSFORMERS nor listed as advisory in this test: {orphans}"
    )


def test_readme_documents_the_transformers_contract():
    from mata.adapters.transformers_compat import MIN_TRANSFORMERS

    readme = README.read_text(encoding="utf-8")
    assert "### Transformers version" in readme, "README never states that the base pin is wider than some model floors"
    for model_id, floor in MIN_TRANSFORMERS.items():
        assert model_id in readme, f"README install section does not mention {model_id}"
        assert f"transformers>={floor}" in readme, f"README does not state transformers>={floor} for {model_id}"
