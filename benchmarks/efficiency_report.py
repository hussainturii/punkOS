"""Read task timing aggregates without exporting prompts or document contents."""
import argparse
import json
from pathlib import Path
import sqlite3
from contextlib import closing


def report(database, limit=20):
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        if 'metrics_json' not in {row[1] for row in conn.execute('PRAGMA table_info(tasks)')}:
            return {'tasks': [], 'note': 'No timing data in this history schema.'}
        rows = conn.execute('SELECT task_id, success, duration_seconds, metrics_json FROM tasks '
                            'ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
    return {'tasks': [{'task_id': row['task_id'], 'success': bool(row['success']),
                       'duration_seconds': row['duration_seconds'],
                       'metrics': json.loads(row['metrics_json'])} for row in rows],
            'note': 'Model counts are provider calls, not HTTP attempts or token usage. Categories may overlap.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', required=True, help='Existing task-history SQLite database')
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000:
        parser.error('--limit must be between 1 and 1000')
    print(json.dumps(report(args.database, args.limit), indent=2))
