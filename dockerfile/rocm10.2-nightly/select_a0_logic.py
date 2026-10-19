# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Select unmodified generic scalar GEMM logic, including paired FP8 types."""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml


source, target = map(Path, sys.argv[1:])
selected = {}
solutions = 0
types_seen = set()
for path in sorted(source.rglob('*.yaml')):
    if path.parent.name != 'GridBased':
        continue
    data = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
    problem = data['ProblemType'] if isinstance(data, dict) else data[4]
    kernels = data['Solutions'] if isinstance(data, dict) else data[5]
    architecture = data['ArchitectureName'] if isinstance(data, dict) else data[2]
    assert architecture == 'gfx1250-strict', str(path)
    types = tuple(problem[key] for key in ('DataTypeA', 'DataTypeB', 'DestDataType'))
    if not all(dtype in (0, 4, 7, 15, 16, 17, 18) for dtype in types[:2]):
        continue
    if types[2] not in (0, 4, 7, 15, 16):
        continue
    if any(problem[key] for key in ('Sparse', 'GroupedGemm', 'SwizzleTensorA', 'SwizzleTensorB')):
        continue
    if problem['UseScaleAB'] not in (False, '', 'Scalar'):
        continue
    if any(problem.get(key, False) for key in ('FusedGemmA2A', 'UseGateResidual', 'MXBlockA', 'MXBlockB')):
        continue
    relative = path.relative_to(source)
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.read_bytes() == path.read_bytes(), str(destination)
    else:
        shutil.copy2(path, destination)
    selected[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    solutions += len(kernels)
    types_seen.update(types)

assert selected and {0, 4, 7, 15, 16} <= types_seen
assert {str(path.relative_to(target)) for path in target.rglob('*.yaml')} == set(selected)
record = {
    'logic_files': len(selected), 'solutions': solutions, 'dtype_ids': sorted(types_seen),
    'sha256': selected,
    'selection': 'Unmodified GridBased scalar GEMM logic; no FP4/MX/sparse/grouped/swizzled/A2A',
}
(target.parent / f'{target.name}-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(f'SELECTED_LOGICS={len(selected)} SOLUTIONS={solutions} DTYPES={sorted(types_seen)}')
