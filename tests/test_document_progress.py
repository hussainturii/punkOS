import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from core.document_evidence import DocumentEvidence
from core.reasoning import NextAction


class DocumentProgressTests(unittest.TestCase):
    def test_existing_draft_cannot_be_saved_directly(self):
        for shortcut in ('ctrl+s','ctrl+shift+s'):
            self.assertTrue(self.evidence.new_document_guard(self.state('Untitled-1','draft'),NextAction('key',text=shortcut),[]))
        dialog=SimpleNamespace(foreground_window_title='Save As',elements=[SimpleNamespace(tag=1,name='Save')])
        for action in (NextAction('click',target_tag=1),NextAction('key',text='enter'),NextAction('type',text='new.py')):
            self.assertTrue(self.evidence.new_document_guard(dialog,action,[]))
        self.assertEqual(self.evidence.new_document_guard(dialog,NextAction('key',text='esc'),[]),'')

    def test_completion_cannot_skip_document_creation_or_source_open(self):
        self.remember()
        self.assertIn('separate new document',self.evidence.completion_prerequisite())
        self.evidence._new_document_observed=True
        self.assertIn('source has not been verified open',self.evidence.completion_prerequisite())
        self.evidence._source_opened=True
        self.assertEqual(self.evidence.completion_prerequisite(),'')

    def test_document_and_file_requests_use_same_new_document_guard(self):
        for noun in ('file', 'document', 'editor tab'):
            evidence = DocumentEvidence(f'Open input.py from Desktop. Keep it unchanged. Create a separate new {noun} and save it on Desktop as output.py.')
            self.assertTrue(evidence.separate_new_file)
            self.assertEqual(evidence.source_prerequisite(self.state('input.py'), []), '')
            self.assertTrue(evidence.new_document_guard(self.state('Untitled-1','draft'),NextAction('type',text='new'),[]))

    def test_vscode_source_open_accepts_multi_step_open_dialog(self):
        from pathlib import PureWindowsPath
        source = r'C:\Work\source.py'
        evidence = DocumentEvidence('Inspect existing document')
        evidence.source = source
        history = [
            {'action': f'type tag=7 text={source!r}', 'success': True, 'observation': 'Open File'},
            {'action': 'click tag=8', 'success': True, 'observation': 'Open File', 'outcome': "ButtonControl 'Open'"},
            {'action': 'click tag=5', 'success': True, 'observation': 'Open File', 'outcome': "ButtonControl 'Open'"},
            {'action': "key tag=None text='enter'", 'success': True, 'observation': 'Open File'},
        ]
        self.assertTrue(evidence._opened_exact_path(history,PureWindowsPath(source)))
        self.assertEqual(evidence.source_prerequisite(self.state('source.py'),history), '')
        history.insert(-1,{'action': "type tag=7 text='C:\\\\Work\\\\other.py'",'success':True,'observation':'Open File'})
        self.assertFalse(evidence._opened_exact_path(history,PureWindowsPath(source)))

    def test_existing_draft_cannot_be_reused_for_separate_file(self):
        state = self.state('Untitled-1', 'unrelated draft')
        action = NextAction('type',target_tag=1,text='new text')
        self.assertIn('Preserve the existing draft', self.evidence.new_document_guard(state,action,[]))
        history = [{'action': "key tag=None text='ctrl+n'",'success':True,
                    'observation': state.foreground_window_title}]
        self.assertTrue(self.evidence.new_document_guard(state,action,history))
        new_state = self.state('Untitled-2','')
        self.assertEqual(self.evidence.new_document_guard(new_state,action,history),'')

    def test_completion_retains_local_evidence_across_navigation(self):
        self.remember()
        evidence = self.evidence.completion_evidence()
        self.assertIn('new content',evidence)
        self.assertIn('"source_unchanged": true',evidence)
        self.output.write_text('changed',encoding='utf-8')
        self.assertEqual(self.evidence.completion_evidence(),'')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.desktop = patch('core.file_destination.desktop_path', return_value=self.folder)
        self.desktop.start()
        self.addCleanup(self.desktop.stop)
        self.source = self.folder / 'input.py'
        self.output = self.folder / 'output.py'
        self.source.write_text('original', encoding='utf-8')
        self.evidence = DocumentEvidence('Open input.py from Desktop. Keep it unchanged. Create a separate new file and save it on Desktop as output.py.')
        self.output.write_text('new content', encoding='utf-8')

    def state(self, name='output.py', text='new content'):
        return SimpleNamespace(foreground_window_title=name+' - Visual Studio Code',
            elements=[SimpleNamespace(tag=1,role='EditControl',name=name,text=text)])

    def remember(self):
        history = [{'action': 'click tag=1', 'success': True, 'observation': 'Save As\nButtonControl: Save'}]
        self.evidence.observe_progress(self.state(), history)

    def test_output_evidence_survives_navigation_and_history_truncation(self):
        self.remember()
        self.assertIn('VERIFIED TASK PROGRESS', self.evidence.progress_hint())
        self.evidence.observe_progress(self.state('input.py','original'), [])
        self.assertEqual(self.evidence.check(self.state('input.py','original')), '')
        self.assertTrue(self.evidence.redundant_creation(NextAction('key',text='ctrl+n')))
        self.assertEqual(self.evidence.redundant_creation(NextAction('key',text='ctrl+o')), '')

    def test_changed_disk_invalidates_saved_progress(self):
        self.remember()
        self.output.write_text('external change',encoding='utf-8')
        self.assertIn('invalid', self.evidence.progress_hint())
        self.assertTrue(self.evidence.check(self.state('input.py','original')))
        self.assertEqual(self.evidence.redundant_creation(NextAction('key',text='ctrl+n')), '')

    def test_changed_source_still_fails_after_output_verified(self):
        self.remember()
        self.source.write_text('unexpected change',encoding='utf-8')
        self.assertIn('protected source changed', self.evidence.check(self.state('input.py','unexpected change')))

    def test_unsaved_changes_in_output_do_not_pass_from_old_snapshot(self):
        self.remember()
        self.assertIn('do not match',self.evidence.check(self.state(text='unsaved edit')))

    def test_save_receipt_without_matching_editor_is_not_evidence(self):
        self.evidence.observe_progress(self.state(text='wrong'),[{'action':'click tag=1','success':True,'observation':'Save As'}])
        self.assertEqual(self.evidence.progress_hint(),'')


if __name__ == '__main__':
    unittest.main()
