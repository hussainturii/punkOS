import unittest
from unittest.mock import Mock
from core.requirements import grounded_requirements, requirement_context
from core.agent_checks import verify_completion
from core.llm import LLMResponse


class RequirementTests(unittest.TestCase):
    def test_historical_field_evidence_does_not_override_missing_save_evidence(self):
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse(dict(complete=True,
            evidence='Inbox', missing='', requirements=[
                dict(id='R1', status='verified', source='prior_observation', evidence='Subject: Example', missing=''),
                dict(id='R2', status='pending', source='current', evidence='', missing='No saved confirmation')]), '')
        requirements = [dict(id='R1', kind='outcome', description='Verify subject before closing'),
                        dict(id='R2', kind='outcome', description='Confirm saved')]
        assessments = {}
        ok, message = verify_completion(provider, 'Verify subject and saved draft', 'Inbox', [],
            requirements=requirements, assessments=assessments,
            observation_history='Captured after history entry 3:\nSubject: Example')
        self.assertFalse(ok)
        self.assertIn('No saved confirmation', message)
        self.assertEqual(assessments['R1']['source'], 'prior_observation')
        self.assertFalse(verify_completion(provider, 'Verify subject and saved draft', 'Inbox', [],
            requirements=requirements, observation_history='Different field')[0])

    def setUp(self):
        self.goal = 'Save the output. Keep the source unchanged.'
        self.requirements = grounded_requirements([
            dict(description='Output saved', kind='outcome', user_quote='Save the output'),
            dict(description='Source unchanged', kind='constraint', user_quote='Keep the source unchanged')], self.goal)
        self.observed = 'Output saved at destination. Source hash unchanged.'

    def verify(self, rows):
        provider = Mock()
        provider.call_with_schema.return_value = LLMResponse(dict(
            complete=True, evidence='Output saved at destination.', missing='', requirements=rows), '')
        assessments = {}
        ok, message = verify_completion(provider, self.goal, self.observed, [],
            requirements=self.requirements, assessments=assessments)
        return ok, message, assessments

    def rows(self):
        return [dict(id='R1', status='verified', evidence='Output saved at destination.', missing=''),
                dict(id='R2', status='verified', evidence='Source hash unchanged.', missing='')]

    def test_invented_user_quote_rejected(self):
        with self.assertRaises(ValueError):
            grounded_requirements([dict(description='Send it', kind='outcome', user_quote='Send it')], self.goal)

    def test_all_requirements_need_evidence(self):
        ok, _, assessments = self.verify(self.rows())
        self.assertTrue(ok)
        self.assertEqual(len(assessments), 2)

    def test_partial_success_cannot_approve_complete(self):
        rows = self.rows()
        rows[1].update(status='pending', evidence='', missing='No source comparison')
        ok, message, assessments = self.verify(rows)
        self.assertFalse(ok)
        self.assertIn('R2: No source comparison', message)
        self.assertEqual(assessments['R1']['status'], 'verified')
        self.assertIn('pending', requirement_context(self.requirements, assessments))

    def test_missing_duplicate_and_invented_ids_rejected(self):
        for rows in [self.rows()[:1], [self.rows()[0]]*2,
                     [self.rows()[0], dict(self.rows()[1], id='R9')]]:
            self.assertFalse(self.verify(rows)[0])

    def test_fabricated_evidence_never_updates_assessments(self):
        rows = self.rows()
        rows[1]['evidence'] = 'I promise nothing changed'
        ok, _, assessments = self.verify(rows)
        self.assertFalse(ok)
        self.assertEqual(assessments, {})

    def test_pending_is_default_not_action_receipt_success(self):
        text = requirement_context(self.requirements, {})
        self.assertEqual(text.count('"status": "pending"'), 2)

    def test_grounded_violation_blocks_completion(self):
        rows = self.rows()
        rows[1].update(status='violated', missing='Constraint violated')
        self.assertFalse(self.verify(rows)[0])
