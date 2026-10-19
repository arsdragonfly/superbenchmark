# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Protect a rebuilt compiler wheel from replacement during overlay installs."""

import argparse
import hashlib
from importlib.metadata import distribution, version
from pathlib import Path
import re

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    """Pin the installed rebuilt wheel and fingerprint its native libraries."""
    parser = argparse.ArgumentParser()
    parser.add_argument('package', choices=('triton', 'flydsl'))
    package = parser.parse_args().package
    installed = version(package)
    wheels = list(Path(f'/opt/fixed-{package}-wheels').glob(f'{package}-{installed}-*.whl'))
    if len(wheels) != 1:
        raise RuntimeError(f'Expected one rebuilt {package} wheel, found {wheels}')
    wheel = wheels[0]
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    root = Path('/opt/base-provenance')
    constraints = root / 'core-constraints.txt'
    lines = constraints.read_text().splitlines()
    lines = [
        line for line in lines
        if canonicalize_name(Requirement(line).name) != canonicalize_name(package)
    ]
    lines.append(f'{package} @ {wheel.as_uri()}#sha256={digest}')
    constraints.write_text('\n'.join(lines) + '\n')
    installed_distribution = distribution(package)
    libraries = []
    for entry in installed_distribution.files or []:
        if re.search(r'\.so(?:\.|$)', entry.name):
            path = Path(installed_distribution.locate_file(entry)).resolve()
            libraries.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}')
    if not libraries:
        raise RuntimeError(f'No native libraries found for {package}')
    (root / f'fixed-{package}-installed.sha256').write_text('\n'.join(sorted(set(libraries))) + '\n')
    print(f'Protected rebuilt {package} wheel and {len(libraries)} native library paths.')


if __name__ == '__main__':
    main()
