import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from core.mouse_window_focus import click_titlebar, minimize_own_dashboard


class MouseWindowFocusTests(unittest.TestCase):
    def setUp(self):
        self.auto, self.click = Mock(), Mock()
        self.window = Mock(ControlTypeName='WindowControl')
        self.bar = Mock(ControlTypeName='TitleBarControl', IsEnabled=True, IsOffscreen=False,
                        BoundingRectangle=SimpleNamespace(left=100, top=100, right=500, bottom=130))
        self.window.GetChildren.return_value = [self.bar]
        self.auto.ControlFromHandle.return_value = self.window
        self.auto.ControlFromPoint.return_value = self.bar
        self.auto.ControlsAreSame.side_effect = lambda a, b: a is b

    def run_focus(self):
        return click_titlebar(10, automation=self.auto, click=self.click)

    def test_visible_titlebar_clicked_once(self):
        self.assertTrue(self.run_focus())
        self.click.assert_called_once_with(300, 115)

    def test_covered_or_interactive_titlebar_does_not_receive_click(self):
        for role in ['ButtonControl', 'TabItemControl', 'MenuItemControl', 'PaneControl']:
            self.auto.ControlFromPoint.return_value = Mock(ControlTypeName=role)
            self.assertFalse(self.run_focus())
        self.click.assert_not_called()

    def test_unrelated_titlebar_is_not_accepted(self):
        other = Mock(ControlTypeName='TitleBarControl')
        other.GetParentControl.return_value = None
        self.auto.ControlFromPoint.return_value = other
        self.assertFalse(self.run_focus())
        self.click.assert_not_called()

    def test_inert_title_text_can_focus_exact_bar(self):
        text = Mock(ControlTypeName='TextControl')
        text.GetParentControl.return_value = self.bar
        self.auto.ControlFromPoint.return_value = text
        self.assertTrue(self.run_focus())

    def test_hidden_and_ambiguous_bars_rejected(self):
        self.bar.IsOffscreen = True
        self.assertFalse(self.run_focus())
        self.bar.IsOffscreen = False
        self.window.GetChildren.return_value = [self.bar, self.bar]
        self.assertFalse(self.run_focus())
        self.click.assert_not_called()

    def test_alternative_point_is_checked_before_click(self):
        self.auto.ControlFromPoint.side_effect = [Mock(ControlTypeName='ButtonControl'), self.bar]
        self.assertTrue(self.run_focus())
        self.click.assert_called_once_with(220, 115)

    def test_only_owned_dashboard_minimize_button_is_clicked(self):
        self.window.ProcessId = 123
        self.window.Name = 'PunkOS — Agentic Desktop AI'
        button = Mock(ControlTypeName='ButtonControl', Name='Minimize', IsEnabled=True,
                      IsOffscreen=False, BoundingRectangle=self.bar.BoundingRectangle)
        self.bar.GetChildren.return_value = [button]
        self.auto.GetForegroundControl.return_value = self.window
        self.auto.ControlFromPoint.return_value = button
        self.assertTrue(minimize_own_dashboard(automation=self.auto, click=self.click, process_id=123))
        self.click.assert_called_once()
        self.click.reset_mock()
        for pid, title in [(456, 'PunkOS — Agentic Desktop AI'), (123, 'Other window')]:
            self.window.ProcessId, self.window.Name = pid, title
            self.assertFalse(minimize_own_dashboard(automation=self.auto, click=self.click, process_id=123))
        self.click.assert_not_called()

    def test_covered_minimize_button_is_not_clicked(self):
        self.window.ProcessId = 123
        self.window.Name = 'PunkOS — Agentic Desktop AI'
        button = Mock(ControlTypeName='ButtonControl', Name='Minimize', IsEnabled=True,
                      IsOffscreen=False, BoundingRectangle=self.bar.BoundingRectangle)
        self.bar.GetChildren.return_value = [button]
        self.auto.GetForegroundControl.return_value = self.window
        self.auto.ControlFromPoint.return_value = Mock()
        self.assertFalse(minimize_own_dashboard(automation=self.auto, click=self.click, process_id=123))
        self.click.assert_not_called()

    @patch('core.window_handoff.time.sleep')
    def test_backend_recovers_from_covered_taskbar_by_minimizing_own_dashboard(self, sleep):
        from core.window_handoff import WindowsBackend, Window
        backend = WindowsBackend.__new__(WindowsBackend)
        target = Window(10, 'Calculator', 'calculator.exe')
        backend.windows = Mock(return_value=[target])
        backend.foreground = Mock(return_value=target)
        backend.activate = Mock()
        with patch('core.mouse_window_focus.click_titlebar', return_value=False), \
             patch('core.taskbar_handoff.click_app_button', return_value=False), \
             patch('core.mouse_window_focus.minimize_own_dashboard', return_value=True) as minimize:
            self.assertTrue(backend.click_taskbar('calculator'))
            minimize.assert_called_once()
        backend.activate.assert_not_called()
