"""Update generated result sections in the two maintained Combo reports."""

from pathlib import Path


def document_path(repo, version):
    if version not in ("2.10", "2.13"):
        raise ValueError(f"Unknown Combo report version: {version}")
    return Path(repo) / "docs/notes/compiler/codegen" / f"inductor-combo-tests-v{version}.md"


def section_bounds(text, section):
    start = text.index(f"<!-- combo-section:{section}:start -->")
    start = text.index("\n", start) + 1
    end = text.index(f"<!-- combo-section:{section}:end -->", start)
    return start, end


def read_section(path, section):
    text = path.read_text()
    start, end = section_bounds(text, section)
    return text[start:end].strip() + "\n"


def write_section(path, section, content):
    text = path.read_text()
    start, end = section_bounds(text, section)
    content = content.strip()
    if content.startswith("# "):
        content = "#" + content
    path.write_text(text[:start] + content + "\n\n" + text[end:])
