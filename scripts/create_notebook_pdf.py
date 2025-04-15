import subprocess
import sys
from pathlib import Path
from shutil import copyfile
from shutil import which

import nbformat
import papermill as pm
from nbconvert import LatexExporter

REPO_ROOT = Path(__file__).parent.parent

CONVERTER = LatexExporter(template_file=(REPO_ROOT / "scripts/latex_template.tex").as_posix())

OUTPUT_DIR = REPO_ROOT / "pdfs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_notebook(notebook_path: Path) -> None:
    """Run notebook to update results."""
    pm.execute_notebook(notebook_path, notebook_path, cwd=notebook_path.parent)


def convert_notebook(notebook_path: Path) -> None:
    """Convert notebooks to latex and copy resources."""
    notebook = nbformat.reads(notebook_path.read_bytes(), as_version=4)
    (body, resources) = CONVERTER.from_notebook_node(notebook)
    (OUTPUT_DIR / f"{notebook_path.stem}.tex").write_text(body, encoding="utf8")
    for resource_name, resource in resources["outputs"].items():
        (OUTPUT_DIR / resource_name).write_bytes(resource)

    for extension in ("png", "jpg"):
        for image_path in notebook_path.parent.rglob(f"*.{extension}"):
            copyfile(image_path, OUTPUT_DIR / image_path.relative_to(notebook_path.parent))


def compile_pdfs():
    """Compile all tex files with ``tectonic``."""
    for text_path in OUTPUT_DIR.rglob("*.tex"):
        subprocess.run(
            [which("tectonic"), "-X", "compile", text_path],
            check=True,
        )


def main():
    root_path = REPO_ROOT
    if len(sys.argv) > 1:
        root_path = REPO_ROOT / sys.argv[1]

    for notebook_path in sorted(root_path.rglob("*.ipynb")):
        run_notebook(notebook_path)
        convert_notebook(notebook_path)
    compile_pdfs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
