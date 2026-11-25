import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from core.metrics import TaskMetrics
from core.history import TaskHistoryStore, TaskRecord
from core.llm import LLMResponse


class MetricsTests(unittest.TestCase):
    def test_calls_images_and_errors_count_without_content(self):
        provider = Mock()
        original = provider.call_with_schema
        original.side_effect = [LLMResponse({}, ''), LLMResponse(None, '', 'failure')]
        metrics = TaskMetrics()
        with metrics.instrument(provider, 'call_with_schema', 'model'):
            provider.call_with_schema(user_text='private', image_b64='secret-image')
            provider.call_with_schema(user_text='private')
        self.assertIs(provider.call_with_schema, original)
        report = metrics.snapshot()
        self.assertEqual(report['categories']['model']['calls'], 2)
        self.assertEqual(report['categories']['model']['failures'], 1)
        self.assertEqual(report['categories']['model']['image_calls'], 1)
        self.assertNotIn('private', str(report))
        self.assertNotIn('secret-image', str(report))

    def test_exception_restores_bound_method(self):
        class Service:
            def execute(self):
                raise RuntimeError('fixture')
        obj = Service()
        metrics = TaskMetrics()
        with self.assertRaises(RuntimeError):
            with metrics.instrument(obj, 'execute', 'action'):
                obj.execute()
        self.assertNotIn('execute', vars(obj))
        self.assertEqual(metrics.snapshot()['categories']['action']['failures'], 1)

    def test_history_migrates_old_schema_and_roundtrips_metrics(self):
        import sqlite3
        from contextlib import closing
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.db'
            with closing(sqlite3.connect(path)) as conn:
                conn.executescript(TaskHistoryStore.SCHEMA)
            store = TaskHistoryStore(path)
            record = TaskRecord('a', 'goal', None, None, [], True, 'done', 0, 1, 'fixture', 'date',
                                metrics={'version': 1, 'categories': {}})
            store.add_task(record)
            self.assertEqual(store.get_task('a')['metrics'], record.metrics)
            self.assertEqual(TaskHistoryStore(path).task_count(), 1)
            from benchmarks.efficiency_report import report
            exported = report(path)
            self.assertEqual(exported['tasks'][0]['metrics'], record.metrics)
            self.assertNotIn('goal', exported['tasks'][0])
