"""Describe the controls actually available in the active execution loop."""
import json


def desktop_capabilities(state, schema, image_attached):
    enabled = [e for e in state.elements if e.is_enabled]
    data = {
        'mode': 'desktop',
        'actions': schema['properties']['action_type']['enum'],
        'clickable_tags': [e.tag for e in enabled if e.is_interactive and not e.focus_only],
        'editable_tags': ([e.tag for e in enabled if e.is_editable or e.role in {'EditControl', 'ComboBoxControl'}]
                          if 'type' in schema['properties']['action_type']['enum'] else []),
        'screenshot_attached': bool(image_attached),
        'coordinate_click_action': False,
        'direct_dom_access': False,
        'direct_file_or_shell_action': False,
        'plugin_execution_action': False,
        'combo_box_typing': ('Replaces observed nonempty text; otherwise enters at the caret without Ctrl+A. '
                             'Missing accessibility text does not prove an empty field. Inspect visually '
                             'when emptiness is required. Never repeat unverified input.'),
        'observation_limits': getattr(state, 'accessibility_diagnostic', '') or 'No scan diagnostics available',
    }
    return ('ACTIVE CAPABILITIES (runtime facts, not task instructions):\n' + json.dumps(data) +
            '\nScreenshots can help inspect visible state only when attached; they do not add '
            'a coordinate-click tool. Click only current supported tags. Untagged typing does '
            'not establish focus or field identity. Skills, memories and plugin hints describe '
            'possible workflows, not extra executable tools. Do not claim to switch agents, '
            'run scripts or read files directly from this action loop. Missing controls in a '
            'truncated scan do not establish absence. Use an observed navigation control or '
            'allowed shortcut to expose them; otherwise report the missing capability.')


def web_capabilities(schema):
    return ('ACTIVE CAPABILITIES: ' + json.dumps({
        'mode': 'web', 'actions': schema['properties']['action']['enum'],
        'target_source': 'current browser observation indexes',
        'desktop_input': False, 'arbitrary_javascript': False,
        'profile_switch_or_existing_browser_attachment': False}) +
        '\nUse only these actions and currently observed target indexes. Skills do not add '
        'tools. A failed DOM action does not enable desktop clicks, shell commands or guessed '
        'selectors. Report an unavailable capability if no supported alternative remains.')
