"""Replay controlled observations against the configured model; never sends UI input."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.llm import build_provider
from core.perception import ScreenState, UIElement
from core.reasoning import ReasoningCore


def cases():
    calc = ScreenState('', '', [
        UIElement(1, 'ButtonControl', 'Multiply by', (0,0,10,10), True),
        UIElement(2, 'ButtonControl', 'Plus', (0,0,10,10), True),
        UIElement(3, 'ButtonControl', 'Equals', (0,0,10,10), True),
        UIElement(4, 'TextControl', 'Display is 27', (0,0,10,10), True, is_interactive=False),
    ], 'Calculator', 100, 100)
    note = ScreenState('', '', [UIElement(1, 'EditControl', 'Text editor', (0,0,10,10), True)], 'Untitled - Notepad', 100, 100)
    opaque = ScreenState('', '', [UIElement(1, 'DocumentControl', 'App content', (0,0,10,10), True, is_interactive=False)], 'Opaque application', 100, 100)
    return [
        ('calculator_after_input', 'Use Calculator to calculate 27 times 14 then add 125.', calc,
         [{'action': "type tag=None text='27'", 'outcome': 'Number keys sent', 'success': True, 'observation': 'Calculator\nTextControl: Display is 0 '}],
         lambda a: a.action_type == 'click' and a.target_tag == 1 and not a.is_high_risk),
        ('notepad_literal', 'Type exactly Hello benchmark into the empty Notepad document. Leave it unsaved.', note, [],
         lambda a: a.action_type == 'type' and a.text == 'Hello benchmark' and a.target_tag == 1 and not a.is_high_risk),
        ('opaque_app', 'Click the Search button in this application.', opaque, [],
         lambda a: a.action_type == 'fail' and a.raw.get('blocked_by') == 'perception'),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--provider', choices=['ollama', 'gemini', 'grok', 'groq'])
    parser.add_argument('--model')
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise SystemExit('Use a fresh output file to preserve previous runs.')
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    if args.provider:
        config['llm_provider'] = args.provider
    selected = config.get('llm_provider', 'ollama')
    config.setdefault(selected, {})['request_timeout'] = 90
    if args.model:
        config[selected]['model'] = args.model
    provider = build_provider(config)
    readiness = getattr(provider, 'ensure_available', None)
    if callable(readiness):
        readiness()
    reasoning = ReasoningCore(provider)
    results = []
    for name, goal, state, history, oracle in cases():
        provider.last_metrics = {}
        start = time.monotonic()
        action = reasoning.decide_next_action(goal, state, history)
        row = dict(case=name, passed=oracle(action), seconds=round(time.monotonic()-start, 3),
                   action=action.__dict__, metrics=getattr(provider, 'last_metrics', {}))
        results.append(row)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(provider=provider.name(), results=results), indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    return 0 if all(row['passed'] for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
