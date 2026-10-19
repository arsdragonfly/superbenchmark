# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run the pinned 54-case MI455X profile sequentially with host-side health checks."""

import argparse
import datetime
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'superbench/config/amd_mi455x_rocm102_smoke.yaml'
VALIDATED_DRIVER = 'DD338B73DA3807A0184A715'
FAULT = re.compile(
    r'page fault|GPU reset|Mode2 reset|failed to respond|failed to reset|'
    r'Failed to export SMU|LSDMA PIO failed|ring .*timeout|BUG:|Oops:', re.I
)


def checked_output(command):
    """Run a read-only host command and preserve failures."""
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)


def privileged(command):
    """Use existing noninteractive sudo permissions when the caller is not root."""
    return command if os.geteuid() == 0 else ['sudo', '-n', *command]


def kernel_log():
    """Read kernel messages without touching unsafe KFD debugfs interfaces."""
    return checked_output(privileged(['journalctl', '-k', '-b', '--no-pager']))


def blocked_tasks():
    """List current D-state tasks for bounded persistence checks."""
    return [
        line for line in checked_output(['ps', '-eo', 'pid,stat,comm']).splitlines()
        if re.match(r'\s*\d+\s+D', line)
    ]


def health():
    """Distinguish transient host I/O waits from persistent blocked tasks."""
    kernel = kernel_log()
    observed = blocked_tasks()
    pending = observed
    for _ in range(3):
        if not pending:
            break
        time.sleep(2)
        pending = blocked_tasks()
    return kernel, {
        'faults': [line for line in kernel.splitlines() if FAULT.search(line)],
        'observed_blocked_tasks': observed,
        'persistent_blocked_tasks': pending,
    }


def require_idle():
    """Refuse to interfere with other users, containers or GPU holders."""
    caller = os.environ.get('SUDO_USER', getpass.getuser())
    others = [line for line in checked_output(['who']).splitlines() if line.split()[0] != caller]
    containers = checked_output(['docker', 'ps', '--format', '{{.Names}}']).strip()
    devices = ['/dev/kfd', *map(str, Path('/dev/dri').glob('renderD*'))]
    holders = subprocess.run(
        privileged(['fuser', *devices]), capture_output=True, text=True, timeout=15
    )
    if others or containers or holders.returncode != 1 or holders.stdout.strip() or holders.stderr.strip():
        raise RuntimeError(
            f'Host is not idle: users={others}, containers={containers!r}, '
            f'GPU holders={holders.stdout}{holders.stderr}'
        )


def commands(image, config, case, name, entry, groups, seconds):
    """Construct argument arrays; never execute shell prefixes from configuration."""
    mode = entry['modes'][0]
    distributed = mode['name'] == 'torch.distributed'
    count = 1 if distributed else mode['proc_num']
    ranks = list(range(4)) if distributed else list(range(count))
    result = []
    for rank in range(count):
        container = f'sb-rocm102-smoke-{os.getpid()}-{case.name}-r{rank}'
        command = [
            'docker', 'run', '--rm', '--name', container, '--network=none',
            '--ipc=private', '--shm-size=8g', '--ulimit', 'core=0', '--ulimit', 'memlock=-1:-1',
            '--device=/dev/kfd', '--device=/dev/dri',
            '-v', f'{config}:/smoke.yaml:ro', '-v', f'{case}:/results',
        ]
        for group in groups:
            command += ['--group-add', str(group)]
        if name.startswith('gpu-copy-bw:'):
            command += ['--cap-add', 'SYS_NICE']
        visible = '0,1,2,3' if distributed or count == 1 else str(rank)
        environment = {
            'HIP_VISIBLE_DEVICES': visible, 'HSA_HOTSWAP_ENABLE': '1', 'HSA_TOOLS_LIB': '',
            'HSA_DISABLE_COREDUMP_ON_EXCEPTION': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
            'NCCL_SOCKET_IFNAME': 'lo', 'NCCL_CUMEM_ENABLE': '0', 'NCCL_NVLS_ENABLE': '0',
            'TRITON_HIP_USE_ASYNC_COPY': '0', 'PYTHONUNBUFFERED': '1',
            'ROCBLAS_USE_HIPBLASLT': '0', 'TORCH_BLAS_PREFER_HIPBLASLT': '0',
        }
        if not distributed:
            environment['PROC_RANK'] = str(rank)
        for key, value in environment.items():
            command += ['-e', f'{key}={value}']
        command += [image, 'timeout', '--signal=TERM', '--kill-after=15s', str(seconds)]
        if distributed:
            command += [
                'torchrun', '--nnodes=1', '--nproc_per_node=4', '--rdzv_backend=static',
                '--master_addr=127.0.0.1', '--master_port=29500', '--no_python',
            ]
        if entry.get('frameworks') == ['pytorch']:
            command += ['python3', '/opt/superbench/dockerfile/rocm10.2-nightly/exec_rocblas.py']
        else:
            command += ['sb']
        command += ['exec', '-c', '/smoke.yaml', '-C', f'superbench.enable={name}', '--output-dir', '/results/sb']
        result.append((rank, container, command))
    return ranks, result


def numeric_values(value):
    """Visit numeric result values while retaining nested result shapes."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from numeric_values(item)


def verify_benchmark(benchmark, rank):
    """Check one benchmark record without assuming every rank owns metrics."""
    values = benchmark['result']
    if values.get('return_code') != [0]:
        raise RuntimeError(f'Benchmark failure on rank {rank}: {values}')
    metrics = 0
    for key, value in values.items():
        if key == 'return_code':
            continue
        numbers = list(numeric_values(value))
        if any(not math.isfinite(number) for number in numbers):
            raise RuntimeError(f'Nonfinite {key} on rank {rank}')
        metrics += len(numbers)
    return metrics


def verify_results(case, ranks, name):
    """Require complete rank results and explicit RCCL correctness evidence."""
    results = []
    metrics = 0
    for rank in ranks:
        files = list((case / 'sb/benchmarks').glob(f'*/rank{rank}/results.json'))
        if len(files) != 1:
            raise RuntimeError(f'Expected one result file for rank {rank}, found {files}')
        benchmarks = json.loads(files[0].read_text())
        if not benchmarks:
            raise RuntimeError(f'Empty results for rank {rank}')
        for benchmark in benchmarks:
            metrics += verify_benchmark(benchmark, rank)
        results.append({'rank': rank, 'benchmarks': benchmarks})
    if metrics == 0:
        raise RuntimeError('No numeric benchmark metrics were produced')
    if name.startswith('rccl-bw:'):
        for rank in ranks:
            console = case / f'rank{rank}-console.log'
            if not console.is_file():
                raise RuntimeError(f'Missing RCCL console for rank {rank}')
            checks = re.findall(r'Out of bounds values\s*:\s*(\d+)', console.read_text())
            if not checks or any(int(value) for value in checks):
                raise RuntimeError(f'Missing or failed RCCL correctness checks on rank {rank}: {checks}')
    return results


def execute_case(image, config, case, name, entry, groups, seconds):
    """Run one case, stopping submissions immediately if host health deteriorates."""
    case.mkdir()
    ranks, invocations = commands(image, config, case, name, entry, groups, seconds)
    record = {'benchmark': name, 'start': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'processes': [], 'result': 'RUNNING'}
    for rank, container, command in invocations:
        require_idle()
        with (case / f'rank{rank}-console.log').open('w') as console:
            process = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    current = kernel_log()
                    if any(FAULT.search(line) for line in current.splitlines()):
                        subprocess.run(['docker', 'stop', '--timeout=10', container], check=True, timeout=45)
                        break
        record['processes'].append({'rank': rank, 'returncode': process.wait(timeout=30), 'command': command})
        kernel, state = health()
        (case / 'kernel-after.log').write_text(kernel)
        record['health'] = state
        (case / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
        if state['faults'] or state['persistent_blocked_tasks']:
            raise RuntimeError(f'Unhealthy GPU/host state after {name}')
        if record['processes'][-1]['returncode']:
            raise RuntimeError(f'Process failed in {name}; see {case}')
    record['results'] = verify_results(case, ranks, name)
    record['result'] = 'PASS'
    record['end'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (case / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def record_failure(record, case, name, error):
    """Persist a failed case even when rank results were never produced."""
    record['result'] = 'STOP_REQUIRES_INVESTIGATION'
    record['error'] = str(error)
    failure = {'benchmark': name, 'result': 'FAIL', 'error': str(error)}
    if case.is_dir():
        status = case / 'status.json'
        if status.exists():
            failure = {**json.loads(status.read_text()), **failure}
        status.write_text(json.dumps(failure, indent=2) + '\n')
    if not record['cases'] or record['cases'][-1]['benchmark'] != name:
        record['cases'].append(failure)


def main():
    """Validate the image/config identity and run the selected smoke cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Immutable repository@sha256 digest or local sha256 image ID.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--only', nargs='+', help='Run an explicit subset; omitted means all 54 cases.')
    parser.add_argument('--timeout', type=int, default=900, help='Per-container deadline, including cold startup.')
    parser.add_argument('--expected-driver-srcversion', default=VALIDATED_DRIVER)
    args = parser.parse_args()
    if '@sha256:' not in args.image and not args.image.startswith('sha256:'):
        parser.error('Use an immutable image identity, not a floating tag')
    config = args.config.resolve()
    data = yaml.safe_load(config.read_text())['superbench']['benchmarks']
    names = args.only or list(data)
    if len(data) != 54 or any(name not in data for name in names):
        parser.error('The shipped 54-case profile and valid case names are required')
    require_idle()
    actual_driver = Path('/sys/module/amdgpu/srcversion').read_text().strip()
    if actual_driver != args.expected_driver_srcversion:
        raise RuntimeError(f'Driver differs from the requested baseline: {actual_driver}')
    kernel, state = health()
    if state['faults'] or state['persistent_blocked_tasks']:
        raise RuntimeError('Host health is not suitable for GPU submissions')
    image = json.loads(checked_output(['docker', 'image', 'inspect', args.image]))[0]
    embedded_hash = checked_output([
        'docker', 'run', '--rm', '--network=none', '--entrypoint', 'sha256sum', image['Id'],
        '/opt/superbench/superbench/config/amd_mi455x_rocm102_smoke.yaml',
    ]).split()[0]
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    if embedded_hash != config_hash:
        raise RuntimeError('Host and image smoke configurations differ')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'kernel-before.log').write_text(kernel)
    groups = sorted({os.stat(path).st_gid for path in ['/dev/kfd', *map(str, Path('/dev/dri').glob('renderD*'))]})
    record = {'image': args.image, 'image_id': image['Id'], 'labels': image['Config']['Labels'],
              'config_sha256': config_hash, 'driver': actual_driver,
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'selected_cases': names, 'cases': [], 'result': 'RUNNING'}
    try:
        for index, name in enumerate(names):
            slug = name.replace(':', '-')
            case = output / f'{index + 1:02}-{slug}'
            print(f'SMOKE_BEGIN {index + 1}/{len(names)} {name}', flush=True)
            record['cases'].append(execute_case(image['Id'], config, case, name, data[name], groups, args.timeout))
            (output / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
            print(f'SMOKE_PASS {name}', flush=True)
        record['result'] = 'PASS'
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        record_failure(record, case, name, error)
        raise
    finally:
        (output / 'results.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
