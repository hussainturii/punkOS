import unittest
from unittest.mock import patch, Mock
from core.document_evidence import DocumentEvidence
from core.perception import ScreenState, UIElement


class OutputRecoveryTests(unittest.TestCase):
    goal = r'Open C:\Fixtures\source.txt. Save as C:\Fixtures\reviewed.txt, leaving source.txt unchanged.'

    def state(self, title='reviewed.txt - Notepad', text='PENDING\n'):
        return ScreenState('', '', [UIElement(1, 'DocumentControl', 'Text editor', (0,0,10,10), True, text=text)], title, 100, 100)

    def receipts(self, path=r'C:\Fixtures\reviewed.txt'):
        return [dict(action=f'type tag=9 text={path!r}', success=True, observation='Open\nEditControl: File name:'),
                dict(action="click tag=3 text=''", success=True, observation='Open\nButtonControl: Open',
                     outcome="Left-clicked [3] ButtonControl 'Open' at (10,10)")]

    @patch('core.document_evidence._read', return_value=b'PENDING\n')
    def test_source_basename_does_not_satisfy_exact_open(self, read):
        evidence = DocumentEvidence(self.goal)
        state = self.state('source.txt - Notepad')
        self.assertTrue(evidence.source_prerequisite(state, []))
        self.assertTrue(evidence.source_prerequisite(state, self.receipts(r'C:\Old\source.txt')))
        self.assertFalse(evidence.source_prerequisite(state, self.receipts(r'C:\Fixtures\source.txt')))
        self.assertFalse(evidence.source_prerequisite(self.state(), []))

    @patch('core.document_evidence._read', return_value=b'PENDING\n')
    def test_same_basename_edit_is_blocked_before_open(self, read):
        from core.agent_checks import misplaced_document_edit
        from core.reasoning import NextAction
        evidence = DocumentEvidence(self.goal)
        state = self.state('source.txt - Notepad')
        reason = evidence.source_prerequisite(state, [])
        self.assertTrue(misplaced_document_edit(state, NextAction('key', text='ctrl+h'), self.goal, reason))
        self.assertFalse(misplaced_document_edit(state, NextAction('key', text='ctrl+o'), self.goal, reason))

    @patch('core.document_evidence._read', return_value=b'PENDING\n')
    def test_exact_open_unlocks_repair_then_leaving_tab_revokes(self, read):
        evidence = DocumentEvidence(self.goal)
        self.assertTrue(evidence.allow_output_recovery(self.state(), self.receipts()))
        self.assertTrue(evidence.allow_output_recovery(self.state('*reviewed.txt - Notepad', 'READY\n'), []))
        self.assertFalse(evidence.allow_output_recovery(self.state('other.txt - Notepad'), []))
        self.assertFalse(evidence.allow_output_recovery(self.state(), []))

    @patch('core.document_evidence._read', return_value=b'PENDING\n')
    def test_stale_tab_wrong_directory_and_failed_open_stay_blocked(self, read):
        evidence = DocumentEvidence(self.goal)
        self.assertFalse(evidence.allow_output_recovery(self.state(), []))
        self.assertFalse(evidence.allow_output_recovery(self.state(), self.receipts(r'C:\Old\reviewed.txt')))
        receipts = self.receipts(); receipts[-1]['success'] = False
        self.assertFalse(evidence.allow_output_recovery(self.state(), receipts))

    @patch('core.document_evidence._read')
    def test_disk_mismatch_stays_blocked(self, read):
        read.side_effect = [b'PENDING\n', b'PENDING\n', b'OTHER\n']
        self.assertFalse(DocumentEvidence(self.goal).allow_output_recovery(self.state(), self.receipts()))

    @patch('core.document_evidence._read', return_value=b'PENDING\n')
    def test_save_as_receipts_cannot_impersonate_open(self, read):
        receipts = self.receipts()
        for item in receipts:
            item['observation'] = 'Save as\nButtonControl: Open'
        self.assertFalse(DocumentEvidence(self.goal).allow_output_recovery(self.state(), receipts))


if __name__ == '__main__':
    unittest.main()
