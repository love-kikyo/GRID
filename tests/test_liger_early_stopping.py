"""Pure policy regressions; no dataset or accelerator is needed."""
import copy
import json
from pathlib import Path

import pytest

from src.baselines.liger_early_stopping import initial_policy, advance_policy, validate_policy


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1] /
        'configs/baselines/liger_capacity384_multiepoch_seed42_v1.json').read_text())


def validation(epoch, recall=.1, ndcg=.05):
    ref = lambda role: dict(path=f'/toy/epoch{epoch:02d}/{role}', sha256=str(epoch % 10)*64)
    return dict(epoch=epoch, checkpoint=ref('checkpoint.pt'), report=ref('report.json'),
                acceptance=ref('acceptance.json'), metrics=dict(hybrid={'Recall@5':recall, 'NDCG@5':ndcg}))


def test_flat_curve_stops_exactly_at_six_and_preserves_first_best(config):
    state = initial_policy()
    for epoch in range(1, 7):
        before, entry = copy.deepcopy(state), validation(epoch)
        state = advance_policy(state, entry, config)
        assert before['completed_epochs'] == epoch - 1
        assert state['bad_epochs'] == epoch - 1
        assert state['stopped'] is (epoch == 6)
        assert validate_policy(state, config) == state
    assert state['stop_reason'] == 'patience' and state['best']['epoch'] == 1
    with pytest.raises(AssertionError): advance_policy(state, validation(7, .9), config)


def test_improvement_resets_patience_and_restored_history_preserves_count(config):
    state = initial_policy()
    recalls = [.1, .09, .1, .11, .105, .11, .109, .10, .11]
    for epoch, recall in enumerate(recalls, 1):
        state = advance_policy(state, validation(epoch, recall), config)
        state = json.loads(json.dumps(state))
        assert state['stopped'] is (epoch == 9)
    assert state['bad_epochs'] == 5 and state['monitor_best'] == .11
    assert state['best']['epoch'] == 4


def test_improving_curve_obeys_absolute_twelve_epoch_cap(config):
    state = initial_policy()
    for epoch in range(1, 13):
        state = advance_policy(state, validation(epoch, epoch / 20), config)
        assert state['bad_epochs'] == 0
        assert state['stopped'] is (epoch == 12)
    assert state['stop_reason'] == 'max_epochs' and state['best']['epoch'] == 12


def test_ndcg_tiebreak_changes_best_but_does_not_reset_recall_patience(config):
    state = initial_policy()
    for epoch in range(1, 7):
        state = advance_policy(state, validation(epoch, .1, epoch / 100), config)
    assert state['best']['epoch'] == 6 and state['bad_epochs'] == 5
    assert state['stopped'] is True and state['stop_reason'] == 'patience'


def test_increasing_recall_wins_even_when_ndcg_decreases(config):
    state = advance_policy(initial_policy(), validation(1, .1, .09), config)
    state = advance_policy(state, validation(2, .100001, .05), config)
    assert state['best']['epoch'] == 2 and state['bad_epochs'] == 0


def test_input_entries_and_states_are_not_mutated_or_aliased(config):
    state, entry = initial_policy(), validation(1)
    original = copy.deepcopy(entry)
    out = advance_policy(state, entry, config)
    assert state == initial_policy() and entry == original
    entry['metrics']['hybrid']['Recall@5'] = .9
    assert out['best']['Recall@5'] == .1 and out['validations'][0] == original


@pytest.mark.parametrize('key,value', [
    ('bad_epochs', 0), ('bad_epochs', True), ('completed_epochs', 9),
    ('monitor_best', .9), ('best', None), ('stopped', True), ('stop_reason', 'patience'),
    ('validations', []), ('unknown', 1),
])
def test_restored_summary_cannot_override_history(config, key, value):
    state = advance_policy(initial_policy(), validation(1), config)
    state = advance_policy(state, validation(2), config)
    state[key] = value
    with pytest.raises(AssertionError): validate_policy(state, config)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1., 1.1, True, '0.5', None])
@pytest.mark.parametrize('metric', ['Recall@5', 'NDCG@5'])
def test_invalid_validation_is_immediate_error_even_before_minimum(config, metric, value):
    entry = validation(1)
    entry['metrics']['hybrid'][metric] = value
    with pytest.raises(AssertionError): advance_policy(initial_policy(), entry, config)


@pytest.mark.parametrize('epoch', [0, 2, True, 1.0])
def test_no_duplicate_skipped_or_noninteger_epoch(config, epoch):
    entry = validation(1)
    entry['epoch'] = epoch
    with pytest.raises(AssertionError): advance_policy(initial_policy(), entry, config)


@pytest.mark.parametrize('ref', ['checkpoint', 'report', 'acceptance'])
def test_unbound_validation_reference_rejected(config, ref):
    entry = validation(1)
    entry[ref]['sha256'] = 'not-a-digest'
    with pytest.raises(AssertionError): advance_policy(initial_policy(), entry, config)


def test_minimum_delays_permission_without_discarding_bad_history(config):
    config['early_stopping']['patience'] = 2
    state = initial_policy()
    for epoch in range(1, 7):
        state = advance_policy(state, validation(epoch), config)
        assert state['stopped'] is (epoch == 6)
    assert state['bad_epochs'] == 5


def test_diagnostic_census_does_not_change_epoch_policy(config):
    config['training'].update(batches_per_epoch=3, prefixes_per_epoch=6144, parents_per_epoch=850)
    state = advance_policy(initial_policy(), validation(1), config)
    assert state['completed_epochs'] == 1 and not state['stopped']


@pytest.mark.parametrize('key,value', [('patience', 0), ('patience', True), ('min_epochs', 13),
    ('mode', 'min'), ('monitor', 'test Recall@5'), ('min_delta', .1)])
def test_invalid_policy_rules_rejected(config, key, value):
    config['early_stopping'][key] = value
    with pytest.raises(AssertionError): advance_policy(initial_policy(), validation(1), config)
