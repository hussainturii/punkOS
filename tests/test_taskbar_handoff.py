import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from core.taskbar_handoff import click_app_button, matching_label


class TaskbarTests(unittest.TestCase):
    def setUp(self):
        self.bar = Mock(ClassName='Shell_TrayWnd', NativeWindowHandle=10, ControlTypeName='PaneControl')
        self.button = Mock(ClassName='TaskListButton', NativeWindowHandle=0,
            ControlTypeName='ButtonControl', Name='Calculator, 1 running window',
            IsEnabled=True, IsOffscreen=False,
            BoundingRectangle=SimpleNamespace(left=100, top=900, right=140, bottom=940))
        self.button.GetChildren.return_value = []
        self.button.GetParentControl.return_value = self.bar
        self.bar.GetChildren.return_value = [self.button]
        self.auto = Mock()
        self.auto.GetRootControl.return_value.GetChildren.return_value = [self.bar]
        self.auto.ControlFromPoint.return_value = self.button
        self.auto.ControlFromPoint2.return_value = None
        self.auto.ControlsAreSame.side_effect = lambda a, b: a is b
        self.click = Mock()

    def run_click(self):
        return click_app_button('calculator', automation=self.auto, click=self.click)

    def test_observed_button_clicked_once(self):
        self.assertTrue(self.run_click())
        self.click.assert_called_once_with(120, 920)

    def test_name_boundary(self):
        for name in ['Calculator', 'Calculator, 1 running window.', 'Calculator, 2 running windows',
                     'Calculator, 1 window', 'Calculator - 1 window']:
            self.assertTrue(matching_label(name, 'calculator'))
        for name in ['Calculator tutorial - Chrome', 'My Calculator', 'Calculator malicious']:
            self.assertFalse(matching_label(name, 'calculator'))

    def test_non_taskbar_container_rejected(self):
        self.bar.ClassName = 'BrowserWindow'
        self.assertFalse(self.run_click())
        self.click.assert_not_called()

    def test_ambiguous_buttons_rejected(self):
        self.bar.GetChildren.return_value = [self.button, self.button]
        self.assertFalse(self.run_click())
        self.click.assert_not_called()

    def test_covered_button_rejected(self):
        overlay = Mock()
        overlay.GetParentControl.return_value = None
        self.auto.ControlFromPoint.return_value = overlay
        self.assertFalse(self.run_click())
        self.click.assert_not_called()

    def test_hidden_or_disabled_button_rejected(self):
        self.button.IsOffscreen = True
        self.assertFalse(self.run_click())
        self.button.IsOffscreen = False
        self.button.IsEnabled = False
        self.assertFalse(self.run_click())
        self.click.assert_not_called()

    def test_invalid_geometry_rejected(self):
        self.button.BoundingRectangle.right = 100
        self.assertFalse(self.run_click())
        self.click.assert_not_called()

    def test_discovery_budget_fails_without_click(self):
        self.bar.GetChildren.return_value = [self.button] * 251
        self.assertFalse(self.run_click())
        self.click.assert_not_called()

    def test_native_fallback_requires_exact_taskbar_window(self):
        overlay = Mock()
        overlay.GetParentControl.return_value = None
        self.auto.ControlFromPoint.return_value = overlay
        # A missing UIA hit returns early; use a complete unmatched ancestry.
        overlay.GetParentControl.return_value = overlay
        self.auto.ControlFromPoint2.return_value = self.bar
        self.assertTrue(self.run_click())
        self.click.assert_called_once_with(120, 920)

    def test_hover_reobserves_and_clicks_only_after_hit_check_passes(self):
        overlay = Mock()
        overlay.GetParentControl.return_value = overlay
        self.auto.ControlFromPoint.side_effect = [overlay, self.button]
        reveal = Mock(return_value=True)
        self.assertTrue(click_app_button('calculator', automation=self.auto,
            click=self.click, reveal=reveal))
        reveal.assert_called_once()
        self.click.assert_called_once()
        self.assertEqual(self.auto.GetRootControl.call_count, 2)

    def test_hover_never_loops_or_clicks_a_covering_window(self):
        overlay = Mock()
        overlay.GetParentControl.return_value = overlay
        self.auto.ControlFromPoint.return_value = overlay
        self.auto.ControlFromPoint2.return_value = Mock(NativeWindowHandle=99, ClassName='OtherWindow')
        reveal = Mock(return_value=True)
        self.assertFalse(click_app_button('calculator', automation=self.auto,
            click=self.click, reveal=reveal))
        reveal.assert_called_once()
        self.click.assert_not_called()
