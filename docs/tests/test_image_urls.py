"""Regression for stale CDN images after a documentation update."""
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_ext'))
import seareg_notebooks


class References(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.urls=[]; self.feed(text)
    def handle_starttag(self, tag, attrs):
        self.urls.extend(value for key, value in attrs if key in ('src', 'href'))


class ImageCacheRegression(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(dir=Path.cwd()); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'_images').mkdir(); (self.root/'tutorials').mkdir()
        self.image=self.root/'_images/example.png'; self.image.write_bytes(b'first-image-bytes')
        self.page=self.root/'tutorials/example.html'
        self.page.write_text('<img src="../_images/example.png">'
                             '<a href="../_images/example.png?download=1#view">open</a>'
                             '<img src="https://example.org/_images/example.png">',encoding='utf-8')
        self.events={}
        self.app=SimpleNamespace(builder=SimpleNamespace(format='html',outdir=str(self.root)),
                                 connect=lambda event, callback:self.events.setdefault(event,callback))
        seareg_notebooks.setup(self.app)

    def finish(self, error=None):
        # No callback reproduces the original unchanged/cached image URL.
        self.events.get('build-finished',lambda *args:None)(self.app,error)
        return References(self.page.read_text(encoding='utf-8')).urls

    def test_version_follows_image_bytes_and_keeps_other_query_fields(self):
        original=self.image.read_bytes(); first=self.finish()
        expected=sha256(original).hexdigest()
        self.assertEqual(parse_qs(urlsplit(first[0]).query).get('v'),[expected])
        self.assertEqual(parse_qs(urlsplit(first[1]).query),{'download':['1'],'v':[expected]})
        self.assertEqual(urlsplit(first[1]).fragment,'view')
        self.assertEqual(first[2],'https://example.org/_images/example.png')
        self.assertEqual(self.image.read_bytes(),original)
        self.assertEqual(self.finish(),first)
        self.image.write_bytes(b'changed-image-bytes')
        second=self.finish()
        self.assertNotEqual(first[0],second[0])
        self.assertEqual(parse_qs(urlsplit(second[0]).query)['v'],[sha256(self.image.read_bytes()).hexdigest()])

    def test_failed_or_non_html_build_does_not_rewrite_pages(self):
        original=self.page.read_bytes()
        self.finish(RuntimeError('build failed'))
        self.assertEqual(self.page.read_bytes(),original)
        self.app.builder.format='latex'; self.finish()
        self.assertEqual(self.page.read_bytes(),original)

    def test_outside_output_directory_and_missing_images_are_untouched(self):
        source='<img src="../../_images/outside.png"><img src="../_images/missing.png">'
        self.page.write_text(source,encoding='utf-8'); self.finish()
        self.assertEqual(self.page.read_text(encoding='utf-8'),source)


if __name__=='__main__':unittest.main()
