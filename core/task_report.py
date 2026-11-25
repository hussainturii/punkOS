"""Deterministic partial-result summaries; never infer success from actions."""


def partial_report(message, success, plan, history):
    if success:
        return message
    requirements = getattr(plan, 'requirements', None) if plan is not None else None
    if not isinstance(requirements, list):
        requirements = []
    for row in history:
        proposed = row.get('proposed_requirements')
        if not requirements and isinstance(proposed, list):
            requirements = proposed
    if not requirements:
        return message
    assessments = {}
    for row in history:
        if row.get('action') == 'completion_check' and isinstance(row.get('requirements'), dict):
            # The most recent verification supersedes earlier assessments.
            assessments = row['requirements']
    verified, unresolved = [], []
    for requirement in requirements[:16]:
        if not isinstance(requirement, dict):
            continue
        identity = requirement.get('id', '?')
        description = ' '.join(str(requirement.get('description', 'Requested outcome')).split())[:300]
        item = assessments.get(identity, {})
        if item.get('status') == 'verified' and item.get('evidence'):
            verified.append(f'- {identity}: {description}')
        else:
            detail = ' '.join(str(item.get('missing') or 'Not verified').split())[:300]
            status = 'Violation reported' if item.get('status') == 'violated' else 'Unresolved'
            unresolved.append(f'- {identity}: {description} — {status}: {detail}')
    sections = [message, 'Partial task report (task remains incomplete):']
    if verified:
        sections += ['Previously verified during this run; recheck if later actions changed the state:', *verified]
    else:
        sections.append('No requested outcome has a recorded successful verification.')
    if unresolved:
        sections += ['Outstanding requirements:', *unresolved]
    sections.append('Action receipts alone do not prove completion. Existing work may remain; it was not automatically undone.')
    return '\n'.join(sections)
