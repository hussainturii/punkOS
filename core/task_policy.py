"""Conservative planning shortcut; complex tasks keep their planning phase."""
import re


def result_presentation(success, message):
    """Keep cancellations distinct from execution failures in the dashboard."""
    if success:
        return 'completed', 'Task complete', 'success'
    if message in ('User stopped task', 'User cancelled',
                   'User cancelled high-risk action.', 'User declined browser interaction'):
        return 'stopped', 'Task stopped', 'warning'
    if message == 'Task time limit exceeded':
        return 'stopped', 'Task time limit exceeded', 'warning'
    return 'failed', 'Task failed', 'danger'


def needs_plan(goal):
    text = goal.lower().strip()
    if len(text) > 240 or any(word in text for word in ('save', 'file', 'send', 'delete', 'notepad', 'browser')):
        return True
    return not ('calculator' in text and re.search(r'\b(calculate|compute)\b', text))
