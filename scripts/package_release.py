"""Build a source-only folder and ZIP, without overwriting an existing release."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = ['README.md', 'LICENSE', 'CHANGELOG.md', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md',
             'SECURITY.md', 'requirements.txt', 'requirements-core.txt', 'main.py', 'launch_gui.py',
             'Start GUI.cmd', '.gitignore', '.gitattributes', '.streamlit/config.toml']


def release_files():
    files = {Path(name) for name in TOP_LEVEL}
    for directory, patterns in [('src', ['*.py']), ('app', ['*.py']), ('tests', ['*.py']),
                                ('examples', ['*.py']), ('scripts', ['*.py']),
                                ('docs', ['*.md', '*.png']), ('.github', ['*.yml', '*.md'])]:
        for pattern in patterns:
            files.update(p.relative_to(ROOT) for p in (ROOT / directory).rglob(pattern))
    for relative in files:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing file or symlink in release selection: {relative}')
        if relative.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(relative))
    return sorted(files)


def build(output):
    output = Path(output).resolve()
    archive = output.with_suffix('.zip')
    if output.exists() or archive.exists() or output == ROOT or ROOT.is_relative_to(output):
        raise ValueError('Choose a new output folder that is neither the source root nor an ancestor; existing releases are preserved.')
    files = release_files()
    output.mkdir(parents=True)
    for relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256((ROOT / relative).read_bytes()).digest():
            raise RuntimeError(f'Copy verification failed: {relative}')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative in files:
            bundle.write(output / relative, relative.as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or set(bundle.namelist()) != {p.as_posix() for p in files}:
            raise RuntimeError('ZIP verification failed.')
    print(f'Verified {len(files)} source/release files.\nFolder: {output}\nZIP: {archive}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.output)
