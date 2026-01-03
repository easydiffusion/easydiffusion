import pathlib
import re
import unittest


class TestTerminologyConsistency(unittest.TestCase):
    def setUp(self):
        self.repo_root = pathlib.Path(__file__).resolve().parent.parent

    def test_ui_label_uses_cfg_scale_abbr(self):
        index_html = (self.repo_root / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn('<abbr title="Classifier-Free Guidance">CFG</abbr> Scale', index_html)

    def test_ui_label_mentions_denoising_strength(self):
        index_html = (self.repo_root / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn("Denoising Strength", index_html)

    def test_abbr_has_dotted_underline_styling(self):
        css = (self.repo_root / "ui" / "media" / "css" / "main.css").read_text(encoding="utf-8")
        self.assertRegex(
            css,
            re.compile(r"abbr\[title\][^{]*\{[^}]*text-decoration\s*:\s*underline\s+dotted", re.S),
        )


if __name__ == "__main__":
    unittest.main()
