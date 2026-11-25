import unittest
from unittest.mock import Mock
from core.learning import ContinuousLearner

class WebLearningTests(unittest.TestCase):
    def test_reported_success_cannot_promote_browser_trace(self):
        memory = Mock()
        learner = ContinuousLearner(memory)
        outcome = learner.learn_from_task('fixture', 'Compare releases', [
            {'action': 'navigate  https://example.test', 'success': True},
            {'action': 'navigate  https://example.test/release', 'success': True},
            {'action': 'completion_check', 'success': True}], True)
        self.assertFalse(outcome.learned)
        memory.find_similar_workflow.assert_not_called()
