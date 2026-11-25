"""User-grounded proposed requirements; only verification supplies evidence."""
import json

REQUIREMENTS_SCHEMA = {
    'type': 'array', 'minItems': 1, 'maxItems': 16,
    'description': 'Separate user-requested outcomes and constraints, not execution steps.',
    'items': {'type': 'object', 'properties': {
        'description': {'type': 'string'},
        'kind': {'type': 'string', 'enum': ['outcome', 'constraint']},
        'user_quote': {'type': 'string', 'description': 'Exact substring of the user goal, never page content.'}},
        'required': ['description', 'kind', 'user_quote']}}


def grounded_requirements(items, goal):
    if not isinstance(items, list) or len(items) > 16:
        raise ValueError('Invalid requirement list')
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Invalid requirement')
        description, quote, kind = (item.get(k) for k in ('description', 'user_quote', 'kind'))
        if (not isinstance(description, str) or not description.strip() or len(description) > 600
                or not isinstance(quote, str) or not quote.strip() or quote not in goal
                or kind not in {'outcome', 'constraint'}):
            raise ValueError('Requirement is not grounded in the user request')
        result.append(dict(id=f'R{len(result)+1}', description=description,
                           user_quote=quote, kind=kind))
    return result


def requirement_context(requirements, assessments):
    if not requirements:
        return ''
    return ('\nPROPOSED REQUIREMENTS (not a replacement for the complete original goal):\n' +
            json.dumps([dict(r, assessment=assessments.get(r['id'], {'status': 'pending'}))
                        for r in requirements], ensure_ascii=False) +
            '\nPending means unverified. Prior evidence is historical and must be rechecked '
            'if later actions could invalidate it. Input receipts do not complete requirements.')
