"""Guard test for Sprint 7 Step 7.7's cleanup (docs/sprint7_step7_7_spec.md,
section 5, D.3): every relative link/image in the project's public markdown
files must resolve to a file that actually exists, every file under
``outputs/assets/`` must be used by at least one of them, and every file
under ``outputs/results/`` (recursively) must be either linked from one of
them or referenced by path/filename somewhere in ``src/`` or ``scripts/``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# README.md, any markdown directly under outputs/, and the public model
# card (wherever it currently lives -- it moved out of docs/ after Step
# 7.6, but both locations are checked so a future move is caught too).
PUBLIC_MARKDOWN_FILES: list[Path] = [
    REPO_ROOT / "README.md",
    *sorted((REPO_ROOT / "outputs").glob("*.md")),
    *sorted((REPO_ROOT / "outputs" / "results").glob("model_card_*.md")),
    *sorted((REPO_ROOT / "docs").glob("model_card_*.md")),
]

_LINK_OR_IMAGE_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)|<img[^>]+src=\"([^\"]+)\"")

# Configured directly in the GitHub repo settings (social preview image),
# never referenced from markdown -- confirmed with the owner (Sprint 7
# Step 7.7, D.5), not an orphan despite the lack of a markdown reference.
ASSET_ALLOWLIST = {"social_preview.png"}


def _local_relative_paths(markdown_text: str) -> list[str]:
    """Every link/image target in ``markdown_text`` that is a local
    relative path -- skips http(s) URLs, mailto:, and bare in-page anchors.
    """
    paths = []
    for match in _LINK_OR_IMAGE_RE.finditer(markdown_text):
        target = match.group(1) or match.group(2)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        paths.append(target.split("#", 1)[0])
    return paths


def _existing_public_markdown_files() -> list[Path]:
    return [p for p in PUBLIC_MARKDOWN_FILES if p.exists()]


def test_every_link_and_image_resolves_to_an_existing_file() -> None:
    broken = []
    for md_file in _existing_public_markdown_files():
        text = md_file.read_text(encoding="utf-8")
        for target in _local_relative_paths(text):
            resolved = (md_file.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{md_file.relative_to(REPO_ROOT)} -> {target}")
    assert not broken, "broken relative link(s)/image(s):\n" + "\n".join(broken)


def test_every_asset_is_used_by_at_least_one_public_markdown_file() -> None:
    assets_dir = REPO_ROOT / "outputs" / "assets"
    referenced: set[str] = set()
    for md_file in _existing_public_markdown_files():
        text = md_file.read_text(encoding="utf-8")
        for target in _local_relative_paths(text):
            resolved = (md_file.parent / target).resolve()
            if resolved.is_relative_to(assets_dir):
                referenced.add(resolved.name)

    orphaned = [
        f.name
        for f in assets_dir.iterdir()
        if f.is_file() and f.name not in referenced and f.name not in ASSET_ALLOWLIST
    ]
    assert not orphaned, f"orphaned file(s) in outputs/assets/: {orphaned}"


def test_every_result_file_is_linked_or_referenced_in_code() -> None:
    results_dir = REPO_ROOT / "outputs" / "results"
    # A public markdown file living under outputs/results/ (the model
    # card) is itself a scanned source of links, not something that needs
    # a link pointing *to* it.
    own_markdown_files = {p.resolve() for p in _existing_public_markdown_files()}
    referenced_paths: set[Path] = set(own_markdown_files)
    for md_file in _existing_public_markdown_files():
        text = md_file.read_text(encoding="utf-8")
        for target in _local_relative_paths(text):
            resolved = (md_file.parent / target).resolve()
            if resolved.is_relative_to(results_dir):
                referenced_paths.add(resolved)

    code_text = ""
    for folder in ("src", "scripts"):
        for py_file in (REPO_ROOT / folder).rglob("*.py"):
            code_text += py_file.read_text(encoding="utf-8")

    orphaned = []
    for result_file in results_dir.rglob("*"):
        if not result_file.is_file():
            continue
        if result_file in referenced_paths:
            continue
        rel = result_file.relative_to(results_dir).as_posix()
        if result_file.name in code_text or rel in code_text:
            continue
        orphaned.append(str(result_file.relative_to(REPO_ROOT)))

    assert not orphaned, f"orphaned file(s) in outputs/results/: {orphaned}"
