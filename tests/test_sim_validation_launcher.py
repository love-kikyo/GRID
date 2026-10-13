from pathlib import Path
import subprocess


def test_sim_validation_stages_enforce_budget_and_acceptance_gate():
    path = Path(__file__).resolve().parents[1] / 'run_sim_seed42_validation_cpu4.sh'
    script = path.read_text()
    subprocess.run(['bash', '-n', str(path)], check=True)
    for marker in ('#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=4', '#SBATCH --gres=gpu:1',
                   '#SBATCH --nodelist=aias-compute-2', 'SIM_SMOKE_ACCEPTANCE_SHA256',
                   'checkpoint_acceptance_sha256', 'scripts.accept_sim_checkpoint',
                   'scripts.accept_sim_validation', 'export MAX_BATCHES=2'):
        assert marker in script
    assert 'val "$RUN_DIR"' in script and 'test "$RUN_DIR"' not in script
