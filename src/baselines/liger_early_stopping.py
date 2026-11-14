"""Immutable full-validation policy; stopping state survives process boundaries.

This pure module performs no filesystem access, tensor loading or submission.
Callers must first accept full validation and bind its references. Replaying the
stored history detects counter/best drift rather than trusting saved summaries.
"""
import copy
import math
import re


def initial_policy():
    return dict(completed_epochs=0, bad_epochs=0, monitor_best=None, best=None,
                validations=[], stopped=False, stop_reason=None)


def _rules(config):
    r = config['early_stopping']
    assert r['monitor'] == 'hybrid Recall@5' and r['mode'] == 'max'
    assert type(r['min_delta']) in (int, float) and r['min_delta'] == 0
    assert all(type(r[k]) is int and r[k] > 0 for k in ('patience', 'min_epochs', 'max_epochs'))
    assert r['min_epochs'] <= r['max_epochs']
    assert config['training']['min_epochs'] == r['min_epochs']
    assert config['training']['max_epochs'] == r['max_epochs']
    return r


def _reference(value):
    assert isinstance(value, dict) and isinstance(value.get('path'), str) and value['path']
    assert isinstance(value.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', value['sha256'])


def _validation(entry, epoch):
    assert isinstance(entry, dict) and type(entry.get('epoch')) is int and entry['epoch'] == epoch
    assert isinstance(entry.get('metrics'), dict) and isinstance(entry['metrics'].get('hybrid'), dict)
    scores = entry['metrics']['hybrid']
    for key in ('Recall@5', 'NDCG@5'):
        assert type(scores.get(key)) in (int, float) and math.isfinite(scores[key]) and 0 <= scores[key] <= 1
    for key, value in scores.items():
        assert isinstance(key, str) and type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    for key in ('checkpoint', 'report', 'acceptance'): _reference(entry[key])
    return scores


def _advance(state, entry, rules):
    assert state['stopped'] is False, 'No validation or training may follow a terminal policy'
    epoch = state['completed_epochs'] + 1
    assert epoch <= rules['max_epochs']
    scores = _validation(entry, epoch)
    recall = scores['Recall@5']
    out = copy.deepcopy(state)
    out['completed_epochs'] = epoch
    improved = out['monitor_best'] is None or recall > out['monitor_best']
    if improved:
        out['monitor_best'], out['bad_epochs'] = recall, 0
    else:
        out['bad_epochs'] += 1
    best = out['best']
    key = (recall, scores['NDCG@5'], -epoch)
    if best is None or key > (best['Recall@5'], best['NDCG@5'], -best['epoch']):
        out['best'] = dict(epoch=epoch, checkpoint=copy.deepcopy(entry['checkpoint']), **copy.deepcopy(scores))
    out['validations'].append(copy.deepcopy(entry))
    if epoch >= rules['max_epochs']:
        out.update(stopped=True, stop_reason='max_epochs')
    elif epoch >= rules['min_epochs'] and out['bad_epochs'] >= rules['patience']:
        out.update(stopped=True, stop_reason='patience')
    return out


def validate_policy(state, config):
    """Reconstruct all counters/best decisions from contiguous accepted epochs."""
    assert isinstance(state, dict) and set(state) == set(initial_policy())
    assert type(state['completed_epochs']) is type(state['bad_epochs']) is int
    assert type(state['stopped']) is bool and isinstance(state['validations'], list)
    assert state['monitor_best'] is None or type(state['monitor_best']) in (int, float)
    rules, expected = _rules(config), initial_policy()
    for entry in state['validations']:
        expected = _advance(expected, entry, rules)
    assert state == expected, 'Early-stopping history/counters/best disagree'
    return state


def advance_policy(state, validation_entry, config):
    """One accepted epoch only; NDCG ties never reset Recall patience."""
    validate_policy(state, config)
    return _advance(state, validation_entry, _rules(config))
