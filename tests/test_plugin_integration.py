import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from core.plugins import PluginRegistry, validate_manifest, PluginWorkflow, WorkflowStep
from core.reasoning import ReasoningCore, NextAction


class PluginIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = json.loads((Path(__file__).resolve().parents[1] /
                              'data/plugins/excel/plugin.json').read_text(encoding='utf-8'))
        self.registry = PluginRegistry(self.root)
        self.assertTrue(self.registry.install_from_dict(self.data)[0])

    def test_shipped_shortcuts_are_supported(self):
        folder = Path(__file__).resolve().parents[1] / 'data/plugins'
        for path in folder.glob('*/plugin.json'):
            manifest = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(validate_manifest(manifest), [], path)

    def test_specific_trigger_beats_popular_generic_workflow(self):
        plugin = self.registry.plugins['excel']
        broad = PluginWorkflow('generic', '', ['cell'], [WorkflowStep('wait')], 100, 100)
        plugin.workflows.insert(0, broad)
        self.assertEqual(plugin.find_workflow('go to cell A1').name, 'jump_to_cell')
        duplicate = copy.deepcopy(plugin.workflows[1])
        duplicate.name = 'duplicate'
        plugin.workflows.append(duplicate)
        self.assertIsNone(plugin.find_workflow('go to cell A1'))

    def test_malformed_counter_and_action_are_validation_errors(self):
        for key, value in [('execution_count', 'broken'), ('success_count', -1), ('success_count', 4)]:
            data = copy.deepcopy(self.data)
            data['workflows'][0][key] = value
            self.assertTrue(validate_manifest(data))
        data = copy.deepcopy(self.data)
        data['workflows'][0]['steps'][0]['action_type'] = []
        self.assertTrue(validate_manifest(data))

    def test_discovery_survives_malformed_manifest(self):
        bad = self.root / 'bad'
        bad.mkdir()
        data = copy.deepcopy(self.data)
        data['workflows'][0]['execution_count'] = 'invalid'
        (bad / 'plugin.json').write_text(json.dumps(data))
        self.assertEqual(self.registry.discover(), 1)
        self.assertIn('bad', self.registry.rejected)

    def test_toggle_persists_and_discovery_respects_disabled_state(self):
        self.assertTrue(self.registry.set_active('excel', False)[0])
        self.registry.discover()
        self.assertEqual(self.registry.hint_for_prompt('Book1 - Excel', 'go to cell A1'), '')
        self.assertTrue(self.registry.set_active('excel', True)[0])
        self.assertIn('ctrl+g', self.registry.hint_for_prompt('Book1 - Excel', 'go to cell A1'))

    def test_failed_persistence_keeps_disk_and_memory_state(self):
        path = self.root / 'excel/plugin.json'
        before = path.read_bytes()
        with patch('core.plugins.os.replace', side_effect=PermissionError('locked')):
            self.assertFalse(self.registry.set_active('excel', False)[0])
        self.assertTrue(self.registry.plugins['excel'].is_active)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob('.plugin-*.tmp')), [])

    def test_autosum_key_is_not_converted_into_failure(self):
        action = ReasoningCore._parse_next_action({'action_type': 'key', 'text': 'alt+='})
        self.assertEqual(action.action_type, 'key')

    def test_recovery_receives_plugin_guidance(self):
        from core.planner import StrategicPlanner
        from core.perception import ScreenState
        from core.safety import SafetySystem
        from core.controller import ExecutionResult
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.return_value = ScreenState('', '', [], 'Book1 - Excel', 100, 100)
        reasoning.decide_next_action.return_value = NextAction('wait')
        reasoning.replan_after_failure.return_value = NextAction('fail', text='Stop fixture')
        controller.execute.return_value = ExecutionResult(False, 'Fixture failure')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
            plugins=self.registry, use_planning_phase=False, max_steps=3, step_delay=0, on_log=lambda _: None)
        planner.run('go to cell A1')
        self.assertIn('ctrl+g', reasoning.replan_after_failure.call_args.args[3])
