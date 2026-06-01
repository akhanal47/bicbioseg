def progress_iter(iterable, enabled=True, **kwargs):
    if not enabled:
        return iterable

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable

    return tqdm(iterable, **kwargs)
