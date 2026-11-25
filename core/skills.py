"""Small, local, reviewable task skills. Skills advise; they never execute code."""
from pathlib import Path


class SkillLibrary:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else Path(__file__).resolve().parent.parent / 'data' / 'skills'

    def context(self, goal, window=''):
        text = (goal + ' ' + window).lower()
        names = ['desktop-recovery']
        for name, words in {
            'calculator': ('calculator',),
            'messaging': ('whatsapp', 'message', 'chat', 'send'),
            'documents': ('notepad', '.txt', 'document', 'save', 'overwrite'),
            'code-editing': ('vscode', 'vs code', 'visual studio code', '.py', '.js', 'python file', 'code file', 'create file'),
            'web-downloads': ('browser', 'chrome', 'website', 'download', 'search online'),
        }.items():
            if any(word in text for word in words):
                names.append(name)
        sections = []
        for name in names:
            path = self.directory / (name + '.md')
            if path.is_file():
                sections.append(f'SKILL {name}:\n' + path.read_text(encoding='utf-8')[:5000])
        if 'desktop' in goal.lower():
            from .file_destination import desktop_path
            sections.insert(0, 'LOCAL DESTINATION: Windows Desktop directory is ' + str(desktop_path()) + '. Use this exact directory when the user requests Desktop; do not guess a username or ignore folder redirection.')
        return '\n\n'.join(sections)[:16000]
