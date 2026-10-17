"""Enforces the four-folder dependency rule from Sprint 7 Step 7.0 (4.4):

common            -> common only
model_risk_regimes -> common
model_renewables   -> common
model_post_crisis_resolution -> common, model_renewables

Also guards against importing the Repo 2-only "arena"/"ops" packages, which
must never appear in this repo.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "energy_price_forecast"
SCRIPTS_ROOT = REPO_ROOT / "scripts"

FOLDERS = ("common", "model_risk_regimes", "model_renewables", "model_post_crisis_resolution")

ALLOWED_IMPORTS = {
    "common": {"common"},
    "model_risk_regimes": {"common", "model_risk_regimes"},
    "model_renewables": {"common", "model_renewables"},
    "model_post_crisis_resolution": {"common", "model_renewables", "model_post_crisis_resolution"},
}

FORBIDDEN_TOP_LEVEL_PACKAGES = {"arena", "ops"}


def _collect_files() -> list[Path]:
    files: list[Path] = []
    for folder in FOLDERS:
        files.extend(sorted((PACKAGE_ROOT / folder).rglob("*.py")))
        script_dir = SCRIPTS_ROOT / folder
        if script_dir.is_dir():
            files.extend(sorted(script_dir.rglob("*.py")))
    return [f for f in files if "__pycache__" not in f.parts]


def _home_folder(path: Path) -> str:
    parts = path.relative_to(REPO_ROOT).parts
    anchor = "energy_price_forecast" if "energy_price_forecast" in parts else "scripts"
    return parts[parts.index(anchor) + 1]


def _module_and_package(path: Path) -> tuple[str, str]:
    """Dotted module name and its containing package, as Python would see them."""
    relative = path.relative_to(SRC_ROOT)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        module_name = ".".join(parts)
        return module_name, module_name
    module_name = ".".join(parts)
    return module_name, ".".join(parts[:-1])


def _folder_of(dotted_name: str) -> str | None:
    parts = dotted_name.split(".")
    if parts[0] == "energy_price_forecast" and len(parts) > 1:
        return parts[1]
    return None


def _imports_in(path: Path) -> list[tuple[str, int, str | None]]:
    """Returns (file-relative package, ast import level, resolved absolute module) tuples."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_script = "scripts" in path.relative_to(REPO_ROOT).parts
    package_name = "" if is_script else _module_and_package(path)[1]

    results: list[tuple[str, int, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((package_name, 0, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                results.append((package_name, 0, node.module))
            else:
                resolved = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package_name
                )
                results.append((package_name, node.level, resolved))
    return results


FILES = _collect_files()


@pytest.mark.parametrize("path", FILES, ids=[str(f.relative_to(REPO_ROOT)) for f in FILES])
def test_module_respects_dependency_rule(path: Path) -> None:
    home = _home_folder(path)
    allowed = ALLOWED_IMPORTS[home]

    for _package, _level, resolved in _imports_in(path):
        if resolved is None:
            continue
        top_level = resolved.split(".")[0]
        assert top_level not in FORBIDDEN_TOP_LEVEL_PACKAGES, (
            f"{path.relative_to(REPO_ROOT)} imports forbidden Repo 2-only package "
            f"'{resolved}' -- 'arena' and 'ops' must never appear in this repo"
        )

        imported_folder = _folder_of(resolved)
        if imported_folder is None or imported_folder not in FOLDERS:
            continue
        assert imported_folder in allowed, (
            f"{path.relative_to(REPO_ROOT)} (folder '{home}') imports '{resolved}' "
            f"from folder '{imported_folder}', which is not in its allowed set {sorted(allowed)}"
        )
