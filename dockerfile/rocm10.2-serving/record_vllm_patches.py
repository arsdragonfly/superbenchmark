# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Verify the installed vLLM wheel contains every source-patched Python module."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    """Fingerprint installed patched modules without changing wheel contents."""
    source = Path('/opt/vllm')
    installed = Path(importlib.util.find_spec('vllm').origin).parent
    changed = subprocess.check_output(
        ['git', '-C', str(source), 'diff', '--name-only', '--', 'vllm/'], text=True).splitlines()
    if len(changed) != 12 or any(not name.endswith('.py') for name in changed):
        raise RuntimeError(f'Unexpected AMD patch file set: {changed}')
    hashes = {}
    for name in changed:
        target = installed / Path(name).relative_to('vllm')
        data = target.read_bytes()
        if data != (source / name).read_bytes():
            raise RuntimeError(f'Installed vLLM module differs from patched source: {name}')
        hashes[str(target)] = hashlib.sha256(data).hexdigest()
    Path('/opt/base-provenance/amd-vllm/installed.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(f'Verified {len(hashes)} installed AMD-patched vLLM modules.')


if __name__ == '__main__':
    main()
