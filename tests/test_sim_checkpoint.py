import pytest
import torch

from scripts.accept_sim_checkpoint import finite, frozen_teacher_checks, file_sha256


def test_manifest_string_paths_hash_identically_to_path_objects(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    assert file_sha256(str(path)) == file_sha256(path)


def test_tensor_finiteness_including_nested_optimizer_state():
    assert finite({'state': [{'value': torch.ones(3)}, torch.tensor(2.)]}) == 2
    with pytest.raises(AssertionError):
        finite({'optimizer': [torch.tensor(float('nan'))]})


def test_both_registered_teacher_aliases_must_be_exact():
    teacher = {'weight': torch.arange(4.)}
    state = {prefix + 'weight': teacher['weight'].clone()
             for prefix in ('teacher_model.', 'teacher.sid_model.')}
    assert len(frozen_teacher_checks(state, teacher)) == 2
    state['teacher.sid_model.weight'][0] += 1
    with pytest.raises(AssertionError):
        frozen_teacher_checks(state, teacher)


def test_missing_teacher_alias_fails():
    with pytest.raises(AssertionError):
        frozen_teacher_checks({'teacher_model.weight': torch.ones(1)}, {'weight': torch.ones(1)})
