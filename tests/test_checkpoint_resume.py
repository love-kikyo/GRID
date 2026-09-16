import pytest

from src.checkpoint_resume import relocate_best_checkpoint, validate_resume_mode
from src.checkpoint_resume import test_checkpoint_after_fit as choose_test_checkpoint


def payload(source):
    return {'global_step': 20000, 'state_dict': {'weight': object()},
            'optimizer_states': [{'state': {0: {'step': 20000, 'exp_avg': object()}}}],
            'lr_schedulers': [{'last_epoch': 20000}], 'loops': {'fit_loop': {}},
            'callbacks': {'EarlyStopping{test}': {'wait_count': 0, 'best_score': 0.27},
                          'ModelCheckpoint{test}': {'dirpath': str(source.parent), 'best_model_score': 0.27,
                          'best_model_path': str(source), 'best_k_models': {str(source): 0.27},
                          'kth_best_model_path': str(source), 'last_model_path': ''}}}


def test_relocation_preserves_training_state_and_parent_metadata(tmp_path):
    source, dest = tmp_path/'old/checkpoints/best.ckpt', tmp_path/'new/checkpoints/best.ckpt'
    original = payload(source)
    result = relocate_best_checkpoint(original, source, dest)
    assert result['state_dict'] is original['state_dict']
    for key in ['optimizer_states', 'lr_schedulers', 'loops']:
        assert result[key] is original[key]
    new = result['callbacks']['ModelCheckpoint{test}']
    assert new['dirpath'] == str(dest.parent)
    assert new['best_model_path'] == new['kth_best_model_path'] == str(dest)
    assert new['best_k_models'] == {str(dest): 0.27}
    assert original['callbacks']['ModelCheckpoint{test}']['best_model_path'] == str(source)
    assert result['callbacks']['EarlyStopping{test}'] is original['callbacks']['EarlyStopping{test}']


@pytest.mark.parametrize('broken', ['optimizer_states', 'lr_schedulers', 'loops', 'callbacks'])
def test_reject_incomplete_resume(tmp_path, broken):
    source, dest = tmp_path/'old/best.ckpt', tmp_path/'new/best.ckpt'
    original = payload(source)
    original[broken] = [] if broken.endswith('s') else {}
    with pytest.raises((ValueError, AttributeError)):
        relocate_best_checkpoint(original, source, dest)


def test_reject_overwriting_parent_or_resuming_non_best(tmp_path):
    source = tmp_path/'old/best.ckpt'
    original = payload(source)
    with pytest.raises(ValueError): relocate_best_checkpoint(original, source, source)
    with pytest.raises(ValueError): relocate_best_checkpoint(original, source, source.parent/'copy.ckpt')
    original['callbacks']['ModelCheckpoint{test}']['best_k_models'] = {str(tmp_path/'other.ckpt'): 0.28}
    with pytest.raises(ValueError): relocate_best_checkpoint(original, source, tmp_path/'new/best.ckpt')


def test_full_resume_cannot_reset_optimizer():
    with pytest.raises(ValueError): validate_resume_mode('resume.ckpt', True)
    validate_resume_mode('resume.ckpt', False)
    validate_resume_mode(None, True)


def test_test_after_fit_uses_selected_best_not_resume_input():
    assert choose_test_checkpoint(trained=True, explicit='resume.ckpt', best='newbest.ckpt') == 'newbest.ckpt'
    assert choose_test_checkpoint(trained=False, explicit='evaluate.ckpt', best='best.ckpt') == 'evaluate.ckpt'
    assert choose_test_checkpoint(trained=True, explicit='resume.ckpt', best='') is None
