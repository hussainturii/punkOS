import unittest
from ui.widgets import result_html

class ResultTableTests(unittest.TestCase):
    def test_table_cells_and_newlines_render(self):
        rendered = result_html('Comparison\n\n| Version | Date |\n|---|---|\n| 1.2 | Today |')
        self.assertIn('<table', rendered)
        self.assertIn('<td>1.2</td>', rendered)
        self.assertNotIn('|---|', rendered)
        self.assertIn('Comparison<br>', rendered)

    def test_raw_html_is_not_executed(self):
        rendered = result_html('| Value |\n|---|\n| <img src="file:///secret"> |')
        self.assertNotIn('<img', rendered)
        self.assertIn('&lt;img', rendered)

    def test_ordinary_pipes_stay_text(self):
        self.assertNotIn('<table', result_html('A | B\nno separator'))
