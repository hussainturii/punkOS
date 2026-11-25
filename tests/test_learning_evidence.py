import unittest
from unittest.mock import Mock
from core.learning import ContinuousLearner, extract_steps


class LearningEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.memory = Mock()
        self.memory.find_similar_workflow.return_value = None
        self.learner = ContinuousLearner(self.memory)
        self.trace = [{'action': "key tag=None text='tab'", 'success': True},
                      {'action': "type tag=2 text='Hello\\nWorld'", 'success': True},
                      {'action': 'completion_check', 'success': True}]

    def test_unverified_success_is_not_learned(self):
        outcome = self.learner.learn_from_task('a', 'goal', self.trace[:-1], True)
        self.assertFalse(outcome.learned)
        self.memory.save_workflow.assert_not_called()

    def test_clean_verified_sequence_preserves_literal_newlines(self):
        self.assertTrue(self.learner.learn_from_task('a', 'goal', self.trace, True).learned)
        self.assertEqual(self.memory.save_workflow.call_args.args[0].steps[1].text, 'Hello\nWorld')

    def test_failed_action_is_not_silently_removed_from_learned_sequence(self):
        self.trace.insert(1, {'action': "type tag=2 text='uncertain'", 'success': False})
        self.assertFalse(self.learner.learn_from_task('a', 'goal', self.trace, True).learned)

    def test_different_literal_goal_does_not_merge(self):
        self.learner.learn_from_task('a', 'write FIRST', self.trace, True)
        first = self.memory.save_workflow.call_args.args[0]
        self.memory.find_similar_workflow.return_value = (first, 0.99)
        result = self.learner.learn_from_task('b', 'write SECOND', self.trace, True)
        self.assertEqual(result.action, 'created')
        self.assertEqual(first.execution_count, 1)

    def test_repeated_actions_keep_order_and_independent_objects(self):
        steps = extract_steps([self.trace[0], self.trace[0], self.trace[1]])
        result = self.learner._merge_steps(steps, steps)
        self.assertEqual([s.text for s in result], ['tab', 'tab', 'Hello\nWorld'])
        self.assertIsNot(result[0], result[1])
        self.assertEqual(steps[0].success_weight, 1)

    def test_force_failed_template_does_not_merge_with_successful_procedure(self):
        result = self.learner.learn_from_task('t', 'goal', self.trace[:-1], False, force=True)
        self.assertTrue(result.learned)
        saved = self.memory.save_workflow.call_args.args[0]
        self.assertTrue(saved.is_template)
        self.assertEqual(saved.success_count, 0)
        self.memory.find_similar_workflow.assert_not_called()

    def test_missing_success_is_not_a_success_receipt(self):
        self.assertEqual(extract_steps([{'action': "key tag=None text='tab'"}]), [])
