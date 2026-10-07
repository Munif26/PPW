import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import requests

from app import ArticleFetchError, fetch_article, parse_article_html, validate_detik_url


class DetikUrlTests(unittest.TestCase):
    def test_accepts_detik_domain_and_subdomains(self):
        self.assertEqual(
            validate_detik_url("https://sport.detik.com/bola/berita-123#bagian"),
            "https://sport.detik.com/bola/berita-123",
        )
        self.assertEqual(
            validate_detik_url("https://detik.com/berita"),
            "https://detik.com/berita",
        )

    def test_rejects_urls_outside_https_detik(self):
        invalid_urls = (
            "http://sport.detik.com/berita",
            "https://detik.com.example.org/berita",
            "https://example.com/?next=detik.com",
            "https://user:pass@detik.com/berita",
            "https://detik.com:8443/berita",
        )
        for url in invalid_urls:
            with self.subTest(url=url), self.assertRaises(ArticleFetchError):
                validate_detik_url(url)

    def test_rejects_redirects_outside_detik_without_fetching_them(self):
        response = requests.Response()
        response.status_code = 302
        response.headers["Location"] = "https://example.org/redirect-target"
        response.raw = BytesIO()

        with patch("app.requests.get", return_value=response) as get:
            with self.assertRaises(ArticleFetchError):
                fetch_article("https://www.detik.com/berita")

        get.assert_called_once()

    def test_fetches_and_extracts_a_detik_article(self):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.iter_content.return_value = [
            b"""
            <html><body><h1>Berita Olahraga</h1>
            <div class="detail__body-text">
              <p>Artikel ini berisi laporan pertandingan olahraga dengan informasi lengkap.</p>
              <p>Tim berhasil menang setelah bermain dengan baik sepanjang pertandingan.</p>
            </div></body></html>
            """
        ]
        response.__enter__.return_value = response

        with patch("app.requests.get", return_value=response) as get:
            article = fetch_article("https://sport.detik.com/bola/berita-123")

        self.assertEqual(article.title, "Berita Olahraga")
        self.assertIn("Tim berhasil menang", article.text)
        get.assert_called_once()


class ArticleParsingTests(unittest.TestCase):
    def test_extracts_article_body_without_sidebar_text(self):
        html = b"""
        <html><head><meta property="og:title" content="Judul berita"></head>
        <body>
          <div class="detail__body-text">
            <p>Isi berita membahas pertandingan olahraga dengan lengkap dan terperinci.</p>
            <p>Pemain berhasil mencetak gol pada menit akhir pertandingan tersebut.</p>
          </div>
          <aside><p>Berita rekomendasi yang tidak boleh ikut diproses.</p></aside>
        </body></html>
        """
        article = parse_article_html(html)
        self.assertEqual(article.title, "Judul berita")
        self.assertIn("Pemain berhasil", article.text)
        self.assertNotIn("Berita rekomendasi", article.text)

    def test_rejects_pages_without_an_article_body(self):
        with self.assertRaises(ArticleFetchError):
            parse_article_html(b"<html><body><h1>Judul</h1><p>Singkat.</p></body></html>")


if __name__ == "__main__":
    unittest.main()
