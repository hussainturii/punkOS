import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from core.file_batch import run_batch, resolve_reference
from core.safety import SafetySystem
from core.llm import LLMResponse


class FileBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.src = self.root / 'incoming'
        self.src.mkdir()
        (self.src / 'one.txt').write_text('one')
        (self.src / 'two.txt').write_text('two')
        (self.src / 'keep.csv').write_text('three')
        self.dest = self.root / 'output'
        self.safety = SafetySystem([str(self.root)])
        self.provider = Mock()
        self.review = Mock(return_value=True)

    def run_plan(self, kind='copy', pattern='', preview=False, review=None):
        spec = dict(operation=kind, source=str(self.src), destination=str(self.dest), pattern=pattern)
        self.provider.call_with_schema.return_value = LLMResponse({'operations': [spec], 'unsupported': ''}, '')
        goal = f'{kind.title()} folder "{self.src}" to "{self.dest}".' + (' Preview only.' if preview else '')
        return run_batch(goal, self.provider, self.safety, confirm=review or self.review)

    def test_recursive_copy_preserves_empty_folders_and_bytes(self):
        (self.src / 'empty').mkdir()
        success, message, _ = self.run_plan()
        self.assertTrue(success, message)
        self.assertTrue((self.dest / 'empty').is_dir())
        self.assertEqual((self.dest / 'one.txt').read_text(), 'one')
        self.assertTrue(self.src.exists())
        self.review.assert_called_once()
        self.assertEqual(self.provider.call_with_schema.call_count, 1)

    def test_pattern_copy_uses_observed_filenames_only(self):
        self.dest.mkdir()
        success, message, _ = self.run_plan(pattern='*.txt')
        self.assertTrue(success, message)
        self.assertEqual(sorted(p.name for p in self.dest.iterdir()), ['one.txt','two.txt'])

    def test_dry_run_does_not_mutate_or_request_approval(self):
        self.assertTrue(self.run_plan(preview=True)[0])
        self.assertFalse(self.dest.exists())
        self.review.assert_not_called()

    def test_declined_plan_does_not_mutate(self):
        self.review.return_value = False
        self.assertFalse(self.run_plan()[0])
        self.assertFalse(self.dest.exists())

    def test_collision_stops_whole_batch_before_first_write(self):
        self.dest.mkdir()
        (self.dest / 'two.txt').write_text('protected')
        self.assertFalse(self.run_plan(pattern='*.txt')[0])
        self.assertFalse((self.dest / 'one.txt').exists())
        self.assertEqual((self.dest / 'two.txt').read_text(), 'protected')

    def test_source_change_during_review_is_rejected(self):
        def approve(*args):
            (self.src / 'one.txt').write_text('changed')
            return True
        self.assertFalse(self.run_plan(review=approve)[0])
        self.assertFalse(self.dest.exists())

    def test_nested_destination_is_rejected(self):
        self.dest = self.src / 'nested'
        self.assertFalse(self.run_plan()[0])
        self.assertFalse(self.dest.exists())

    def test_relative_paths_require_unambiguous_authorized_base(self):
        self.assertEqual(resolve_reference('incoming/one.txt', self.safety), self.src / 'one.txt')
        with self.assertRaises(ValueError):
            resolve_reference('one.txt', SafetySystem([str(self.root), str(self.root.parent)]))

    @unittest.skipUnless(os.name == 'nt', 'Windows rename semantics')
    def test_folder_move_verifies_contents_and_removal(self):
        success, message, _ = self.run_plan('move')
        self.assertTrue(success, message)
        self.assertFalse(self.src.exists())
        self.assertEqual((self.dest / 'two.txt').read_text(), 'two')

    def test_mid_copy_failure_reports_partial_destination(self):
        with patch('core.file_batch._execute_file', return_value=(False, 'disk failure', [])):
            success, message, _ = self.run_plan()
        self.assertFalse(success)
        self.assertIn('disk failure', message)
        self.assertTrue(self.dest.exists())

    def test_multiple_operations_are_all_reviewed(self):
        first, second = self.root / 'first.txt', self.root / 'second.txt'
        self.provider.call_with_schema.return_value = LLMResponse({'unsupported':'', 'operations':[
            dict(operation='copy', source=str(self.src/'one.txt'), destination=str(first), pattern=''),
            dict(operation='copy', source=str(self.src/'two.txt'), destination=str(second), pattern='')]}, '')
        goal = f'Copy "{self.src / "one.txt"}" to "{first}" and copy "{self.src / "two.txt"}" to "{second}".'
        success, message, _ = run_batch(goal, self.provider, self.safety, confirm=self.review)
        self.assertTrue(success, message)
        self.assertEqual(len(self.review.call_args.args[1]), 2)
        self.assertEqual(second.read_text(), 'two')

    def test_unmatched_second_pattern_rejects_entire_plan(self):
        self.dest.mkdir()
        self.provider.call_with_schema.return_value = LLMResponse({'unsupported': '', 'operations': [
            dict(operation='copy', source=str(self.src), destination=str(self.dest), pattern='*.txt'),
            dict(operation='copy', source=str(self.src), destination=str(self.dest), pattern='*.absent')]}, '')
        success, message, _ = run_batch(f'Copy files from "{self.src}" to "{self.dest}".',
                                        self.provider, self.safety, confirm=self.review)
        self.assertFalse(success)
        self.assertIn('No regular files', message)
        self.assertEqual(list(self.dest.iterdir()), [])
        self.review.assert_not_called()

    def test_desktop_destination_uses_known_folder(self):
        self.dest.mkdir()
        self.provider.call_with_schema.return_value = LLMResponse({'unsupported': '', 'operations': [
            dict(operation='copy', source=str(self.src / 'one.txt'), destination='Desktop', pattern='')]}, '')
        with patch('core.file_destination.desktop_path', return_value=self.dest):
            success, message, _ = run_batch(f'Copy "{self.src / "one.txt"}" to my Desktop.',
                                            self.provider, self.safety, confirm=self.review)
        self.assertTrue(success, message)
        self.assertEqual((self.dest / 'one.txt').read_text(), 'one')
