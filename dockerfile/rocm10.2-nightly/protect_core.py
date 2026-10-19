# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate constraints that keep the matched SDK and rebuilt Triton installed."""

import hashlib
from importlib.metadata import distributions, version
from pathlib import Path


root = Path('/opt/base-provenance')
names = sorted({
    distribution.metadata['Name']
    for distribution in distributions()
    if distribution.metadata['Name'].lower().replace('_', '-').startswith(
        ('rocm', 'amd-torch', 'torch', 'triton', 'apex'))
})
lines = []
for name in names:
    installed = version(name)
    if name == 'triton':
        wheels = list(Path('/opt/fixed-triton-wheels').glob(f'triton-{installed}-*.whl'))
        if len(wheels) != 1:
            raise RuntimeError(f'Expected one fixed Triton wheel, found {wheels}')
        wheel = wheels[0]
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        lines.append(f'triton @ {wheel.as_uri()}#sha256={digest}')
    else:
        lines.append(f'{name}=={installed}')
(root / 'core-constraints.txt').write_text('\n'.join(lines) + '\n')
