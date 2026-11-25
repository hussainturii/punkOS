"""Task-local timing aggregates. Never store prompts, responses, or screenshots."""
from contextlib import contextmanager
from time import perf_counter


class TaskMetrics:
    def __init__(self):
        self.started = perf_counter()
        self.groups = {}

    @contextmanager
    def instrument(self, instance, method, category):
        if instance is None:
            yield
            return
        original = getattr(instance, method)
        had_attribute = method in vars(instance) or not hasattr(type(instance), method)
        def measured(*args, **kwargs):
            started = perf_counter()
            failed = False
            image = bool(kwargs.get('image_b64')) if category == 'model' else False
            try:
                result = original(*args, **kwargs)
                failed = (getattr(result, 'success', None) is False or
                          isinstance(getattr(result, 'error', None), str) and bool(result.error))
                return result
            except Exception:
                failed = True
                raise
            finally:
                row = self.groups.setdefault(category, {'calls': 0, 'seconds': 0.0, 'failures': 0, 'image_calls': 0})
                row['calls'] += 1
                row['seconds'] += perf_counter() - started
                row['failures'] += int(failed)
                row['image_calls'] += int(image)
        setattr(instance, method, measured)
        try:
            yield
        finally:
            if had_attribute:
                setattr(instance, method, original)
            else:
                delattr(instance, method)

    def snapshot(self):
        return {'version': 1, 'elapsed_seconds': round(perf_counter() - self.started, 4),
                'categories': {name: {**row, 'seconds': round(row['seconds'], 4)}
                               for name, row in self.groups.items()}}
