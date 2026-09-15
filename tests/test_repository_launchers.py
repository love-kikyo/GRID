"""Keep deprecated defaults out of supported launchers; no Slurm or GPU calls."""
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LEGACY = (
    'run_sid.sh', 'run_click.sh', 'run_click_seed43.sh',
    'run_rerank_small.sh', 'run_rerank_medium.sh',
    'run_eval_context_length.sh', 'run_eval_context_mask_length.sh',
    'run_eval_retrieval_length.sh', 'run_profile_context_length.sh',
    'run_profile_context_mask_length.sh', 'run_profile_kv_cache.sh',
    'run_profile_retrieval_length.sh',
)


def test_legacy_launchers_are_only_in_archive():
    for name in LEGACY:
        assert not (ROOT / name).exists()
        assert (ROOT / 'scripts/legacy/2026-09-15' / name).is_file()


def test_supported_launchers_do_not_use_legacy_data_or_checkpoints():
    for path in ROOT.glob('run_*.sh'):
        script = path.read_text()
        assert not re.search(r'data_dir=["\']?(?:\./)?data/taobao["\']?(?:\s|$)', script), path
        assert not re.search(r'logs/train/runs/2026-0[1-8]-', script), path


def test_controls_config_prints_use_formal_teacher_without_training():
    script = ROOT / 'run_test_retraining_controls.sh'
    mocks = r'''
function /home/MMReco2021/.conda/envs/GRID/bin/python() { printf 'CALL'; printf '\t%s' "$@"; printf '\n'; }
mkdir() { :; }
cd() { :; }
'''
    result = subprocess.run(
        ['bash', '-c', mocks + script.read_text()],
        env={'PATH': os.defpath}, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = [line.split('\t')[1:] for line in result.stdout.splitlines() if line.startswith('CALL\t')]
    configs = [args for args in calls if args[:2] == ['-m', 'src.train']]
    assert len(configs) == 2
    for args in configs:
        assert args[-3:] == ['--cfg', 'job', '--resolve']
        teacher = next(arg for arg in args if arg.startswith('model.manual_ckpt_path='))
        assert teacher.endswith('/2026-09-05/sid-seed42-131908/checkpoints/checkpoint_000_090000.ckpt')
