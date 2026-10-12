from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_sim_launcher_is_bounded_and_isolated():
    path = ROOT / 'run_sim_soft_history_last_user_v1.sh'
    script = path.read_text()
    subprocess.run(['bash', '-n', str(path)], check=True)
    for marker in ('#SBATCH --ntasks-per-node=3', '#SBATCH --cpus-per-task=1',
                   '#SBATCH --gres=gpu:3', '#SBATCH --nodelist=aias-compute-2',
                   'source scripts/cpu_budget.sh', '--cpu-bind=verbose,threads',
                   'data/taobao_history_last_user_v1', 'SOURCE_GIT_COMMIT',
                   'test=false', 'sim_rerank_model.py', 'src/baselines/sim_soft.py'):
        assert marker in script
    assert 'experiment=[tiger_train_flat,tiger_train_sim_soft]' in script
    assert 'tiger_train_rerank_small]' not in script
    assert 'trainer.max_steps=2' in script and 'SIMSmokeAudit' in script


def test_sim_config_does_not_inherit_small_reranker_features():
    config = (ROOT / 'configs/experiment/tiger_train_sim_soft.yaml').read_text()
    for marker in ('sim_top_k: 10', 'sim_num_heads: 4', 'sim_gsu_loss_weight: 1.0',
                   'rerank_objective: pointwise', 'use_beam_score_in_rerank_loss: false',
                   'no_hit_aux_loss_weight: 0.5'):
        assert marker in config
    assert 'rank_embedding_dim' not in config and 'reranker_hidden_dim' not in config
