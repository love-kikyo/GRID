"""Pure metadata checks for a full-state resume into a new, isolated run."""

from pathlib import Path


def validate_resume_mode(checkpoint, load_only_weights, train=True):
    if train and checkpoint and load_only_weights:
        raise ValueError("Full resume requires model.load_only_weights=false; refusing optimizer reset")


def test_checkpoint_after_fit(*, trained, explicit, best):
    # A fit-resume checkpoint is not an explicit request to test old weights.
    return (best or None) if trained else (explicit or best or None)


def relocate_best_checkpoint(checkpoint, source, destination):
    """Return a shallow payload copy; preserve every tensor and optimizer state.

    Deliberately supports only resuming the single best checkpoint. No source
    files are removed or rewritten. The saved destination becomes the retained
    best candidate, so Lightning's top-1 cleanup cannot delete the parent run.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.parent == source.parent:
        raise ValueError("Resume output must be in a separate run directory")
    if int(checkpoint.get('global_step', -1)) <= 0:
        raise ValueError("Expected a nonzero optimizer step")
    if len(checkpoint.get('optimizer_states', [])) != 1 or len(checkpoint.get('lr_schedulers', [])) != 1:
        raise ValueError("Full optimizer and scheduler states are required")
    if not checkpoint['optimizer_states'][0].get('state') or not checkpoint.get('loops'):
        raise ValueError("Missing optimizer moments or trainer loop state")
    callbacks = dict(checkpoint.get('callbacks', {}))
    keys = [key for key in callbacks if key.startswith('ModelCheckpoint{')]
    if len(keys) != 1 or not any(key.startswith('EarlyStopping{') for key in callbacks):
        raise ValueError("Expected exactly one best-checkpoint callback and early stopping state")
    state = dict(callbacks[keys[0]])
    best = state.get('best_k_models', {})
    if len(best) != 1 or {Path(path).resolve() for path in best} != {source}:
        raise ValueError("Resume source must be the sole retained best checkpoint")
    if Path(state.get('best_model_path', '')).resolve() != source or state.get('best_model_score') is None:
        raise ValueError("Resume source and recorded best checkpoint disagree")
    for name in ('best_model_path', 'kth_best_model_path', 'last_model_path'):
        value = state.get(name, '')
        if value:
            if Path(value).resolve() != source:
                raise ValueError(f"Unmapped historical checkpoint reference: {name}")
            state[name] = str(destination)
    state['dirpath'] = str(destination.parent)
    state['best_k_models'] = {str(destination): next(iter(best.values()))}
    callbacks[keys[0]] = state
    return {**checkpoint, 'callbacks': callbacks}
