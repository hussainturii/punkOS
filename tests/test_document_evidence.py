import unittest
from unittest.mock import Mock, patch
from core.document_evidence import DocumentEvidence
from core.perception import ScreenState, UIElement


class DocumentEvidenceTests(unittest.TestCase):
    goal = r'Open Notepad, then open C:\Fixtures\source.txt. Replace PENDING with READY. Save the edited document as C:\Fixtures\reviewed.txt, leaving source.txt unchanged.'

    def state(self):
        return ScreenState('', '', [UIElement(1, 'DocumentControl', 'Text editor', (0,0,10,10), True, text='READY\n')], 'reviewed.txt - Notepad', 100, 100)

    @patch('core.document_evidence._read')
    def test_existing_tab_cannot_prove_missing_output(self, read):
        read.side_effect = [b'PENDING\n', b'PENDING\n', FileNotFoundError()]
        evidence = DocumentEvidence(self.goal)
        self.assertEqual(evidence.output, r'C:\Fixtures\reviewed.txt')
        self.assertIn('missing or unreadable', evidence.check(self.state()))

    @patch('core.document_evidence._read')
    def test_changed_source_fails_even_with_correct_output(self, read):
        read.side_effect = [b'PENDING\n', b'READY\n']
        self.assertIn('protected source changed', DocumentEvidence(self.goal).check(self.state()))

    @patch('core.document_evidence._read')
    def test_preserved_source_and_matching_output_pass_local_gate(self, read):
        read.side_effect = [b'PENDING\r\n', b'PENDING\r\n', b'READY\r\n']
        self.assertEqual(DocumentEvidence(self.goal).check(self.state()), '')

    @patch('core.document_evidence._read')
    def test_wrong_output_contents_rejected(self, read):
        read.side_effect = [b'PENDING\n', b'PENDING\n', b'WRONG\n']
        self.assertIn('do not match', DocumentEvidence(self.goal).check(self.state()))

    @patch('core.document_evidence._read')
    def test_unrelated_goal_does_not_read_files(self, read):
        self.assertEqual(DocumentEvidence('Open Calculator').check(self.state()), '')
        read.assert_not_called()

    @patch('core.document_evidence._read')
    def test_planner_rejects_missing_output_before_model_verification(self, read):
        from core.planner import StrategicPlanner
        from core.reasoning import NextAction
        from core.safety import SafetySystem
        read.side_effect = [b'PENDING\n', b'PENDING\n', FileNotFoundError()]
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.return_value = self.state()
        reasoning.decide_next_action.return_value = NextAction('done', text='Already complete')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
            use_planning_phase=False, max_steps=1, step_delay=0, on_log=lambda _: None)
        result = planner.run(self.goal)
        self.assertFalse(result.success)
        self.assertIn('exact source path', result.history[-1]['outcome'])
        reasoning.provider.call_with_schema.assert_not_called()
        controller.execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
