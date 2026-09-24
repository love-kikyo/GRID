"""CPU-bounded HDF5 entry point; leaves the accepted statistical core unchanged."""
import json
import os
from pathlib import Path

from src.runtime_cpu import THREAD_VARIABLES
from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot


def paired_cpu_budget():
    if os.environ.get('GRID_PAIRED_CPU_BUDGET') != '4':
        raise RuntimeError('Paired analysis requires GRID_PAIRED_CPU_BUDGET=4')
    if any(os.environ.get(name) != '1' for name in THREAD_VARIABLES):
        raise RuntimeError('Library thread limits must be set before Python starts')
    return check_export_allocation()


def run(args):
    budget = paired_cpu_budget()
    samples = [export_cpu_snapshot(budget['allowed_cpus'])]
    # Import numeric libraries only after verifying the launch environment.
    from scripts.analyze_paired_evaluation import (
        analyze_h5, file_sha256, markdown_report, parse_named_value,
    )

    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError('Refusing to overwrite an existing paired report')
    if args.output_json.resolve() == args.output_md.resolve():
        raise ValueError('JSON and Markdown outputs must differ')
    if args.expected_split != 'test':
        raise ValueError('Bounded paired entry point requires full test exports')
    if not all(Path(parse_named_value(spec, '=')[1]).suffix == '.h5' for spec in args.input):
        raise ValueError('Bounded paired entry point requires HDF5 inputs')
    print('[cpu budget verified]', budget, flush=True)
    output = analyze_h5(args)
    samples.append(export_cpu_snapshot(budget['allowed_cpus']))
    root = Path(__file__).resolve().parents[1]
    output.update(
        cpu_budget=budget, cpu_process_tree_samples=samples,
        source_git_commit=os.environ.get('SOURCE_GIT_COMMIT'),
        source_files=[{'path': str(root / name), 'sha256': file_sha256(root / name)}
                      for name in ('scripts/analyze_paired_cpu4.py',
                                   'scripts/analyze_paired_evaluation.py',
                                   'src/runtime_cpu.py', 'src/runtime_export_cpu.py',
                                   'run_paired_analysis.sh')],
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('x') as stream:
        stream.write(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with args.output_md.open('x') as stream:
        stream.write(markdown_report(output))
    print(f'[done] wrote {args.output_json} and {args.output_md}', flush=True)
    return output


def main():
    paired_cpu_budget()
    from scripts.analyze_paired_evaluation import parse_args
    run(parse_args())


if __name__ == '__main__':
    main()
