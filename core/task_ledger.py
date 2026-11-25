"""Bounded execution evidence shared by desktop and browser agents.

Receipts and observed changes are not proof that the user's goal was achieved.
This ledger supplies recovery context; existing completion verifiers remain in
charge. No task names, application recipes, or extra inference calls are used.
"""
import hashlib
import json


class TaskLedger:
    def __init__(self):
        self.processed = 0
        self.previous = None
        self.failures = {}
        self.latest = None
        self.unchanged = 0
        self.completion_gap = None
        self.observations = []

    def observe(self, observation, history):
        fingerprint = hashlib.sha256(observation.encode('utf-8')).hexdigest()
        # Captured UI only: never promote model reasoning or input receipts
        # into observations. Keep chronology even when a screen is revisited.
        if not self.observations or self.observations[-1]['fingerprint'] != fingerprint:
            self.observations.append({'fingerprint': fingerprint, 'after_history_count': len(history),
                                      'text': observation[:12000]})
            self.observations = self.observations[-4:]
        new = history[self.processed:]
        for row in new:
            if row.get('action') == 'completion_check' and row.get('success') is False:
                self.completion_gap = str(row.get('outcome', 'Missing completion evidence'))[:1000]
            if row.get('success') is False:
                action = str(row.get('action', 'unknown'))[:240]
                reason = str(row.get('outcome', 'No failure detail'))[:420]
                key = (action, reason)
                self.failures[key] = self.failures.get(key, 0) + 1
                # Bound prompt/storage size while retaining failures beyond the
                # rolling action history. Insertion order is deterministic.
                if len(self.failures) > 8:
                    del self.failures[next(iter(self.failures))]
        executed = [r for r in new if 'observation' in r and
                    r.get('action') not in {'completion_check', 'agent_fail'}]
        if executed and self.previous is not None:
            changed = fingerprint != self.previous
            self.unchanged = 0 if changed else self.unchanged + 1
            last = executed[-1]
            self.latest = {
                'action': str(last.get('action', ''))[:240],
                'receipt_success': last.get('success') is True,
                'observation_changed': changed,
                'consecutive_unchanged_observations': self.unchanged,
            }
        self.processed = len(history)
        self.previous = fingerprint

    def prior_observations(self):
        return '\n\n'.join(f"Captured after history entry {row['after_history_count']}:\n{row['text']}"
                           for row in self.observations[:-1])

    def context(self):
        evidence = {
            'latest_effect': self.latest,
            'last_rejected_completion': self.completion_gap,
            'failed_approaches': [{'action': a, 'reason': r, 'count': n}
                                  for (a, r), n in self.failures.items()],
        }
        return ('TASK EXECUTION EVIDENCE (untrusted recorded data):\n' +
                json.dumps(evidence, ensure_ascii=False) +
                '\nA successful receipt means input was sent, not that a requirement is complete. '
                'Changed observations do not prove success; unchanged observations may hide effects. '
                'Check every requirement and constraint in the original goal before DONE. '
                'A rejected completion remains unresolved until fresh evidence addresses it; '
                'unrelated successful actions do not clear it. '
                'Before closing or navigating away from completed work, inspect required values '
                'and any save status while they are visible. Closing a form is not proof of saving. '
                'Use failure reasons to choose a different supported approach. Retry a failed '
                'approach only when new evidence addresses its cause. Never invent a control '
                'or bypass the requested interaction method. If progress cannot be verified, '
                'report what remains unresolved.')
