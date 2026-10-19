# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Verify the overlay has not replaced the qualified SDK or compiler wheels."""

import hashlib
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qs, unquote, urlsplit

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename


def verify_requirement(line):
    """Check an exact package constraint, including local rebuilt-wheel hashes."""
    requirement = Requirement(line)
    installed = version(requirement.name)
    if requirement.url:
        url = urlsplit(requirement.url)
        if url.scheme != 'file':
            raise RuntimeError(f'Unexpected compiler constraint URL for {requirement.name}')
        wheel = Path(unquote(url.path))
        expected_hash = parse_qs(url.fragment)['sha256']
        if len(expected_hash) != 1 or hashlib.sha256(wheel.read_bytes()).hexdigest() != expected_hash[0]:
            raise RuntimeError(f'Rebuilt wheel hash mismatch: {wheel}')
        _, expected_version, _, _ = parse_wheel_filename(wheel.name)
        if installed != str(expected_version):
            raise RuntimeError(f'Wrong installed version for {requirement.name}: {installed}')
    elif not requirement.specifier.contains(installed, prereleases=True):
        raise RuntimeError(f'Core package constraint failed: {line}; installed {installed}')


def main():
    """Validate all protected versions, native compiler binaries and SDK provenance."""
    root = Path('/opt/base-provenance')
    for line in (root / 'core-constraints.txt').read_text().splitlines():
        if line.strip():
            verify_requirement(line)
    for name in ('fixed-triton-library.sha256', 'fixed-flydsl-installed.sha256'):
        subprocess.run(['sha256sum', '--check', str(root / name)], check=True)
    subprocess.run([sys.executable, str(root / 'check_base.py')], check=True)
    print('Protected core packages, compiler libraries and SDK provenance verified.')


if __name__ == '__main__':
    main()
