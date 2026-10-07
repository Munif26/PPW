from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import joblib
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


MODEL_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = MODEL_DIR / "model_skipgram_naive_bayes_skenario2.joblib"
WORD2VEC_PATH = MODEL_DIR / "model_skipgram_skenario2.model"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class ArticleFetchError(ValueError):
    pass


class ModelArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class Article:
    title: str
    text: str


def validate_detik_url(url: str) -> str:
    try:
        parsed = urlsplit(url.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ArticleFetchError("Format URL tidak valid.") from exc

    if (
        parsed.scheme.lower() != "https"
        or hostname is None
        or not (hostname.lower() == "detik.com" or hostname.lower().endswith(".detik.com"))
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise ArticleFetchError("Masukkan URL artikel HTTPS dari domain detik.com.")

    return parsed._replace(fragment="").geturl()


def parse_article_html(html: bytes) -> Article:
    soup = BeautifulSoup(html, "html.parser")

    heading = soup.find("h1")
    title = heading.get_text(" ", strip=True) if heading else ""
    if not title:
        meta_title = soup.find("meta", attrs={"property": "og:title"})
        title = meta_title.get("content", "").strip() if meta_title else ""

    paragraphs: list[str] = []
    for selector in (".detail__body-text", "article", "main"):
        container = soup.select_one(selector)
        if container is None:
            continue
        paragraphs = [
            paragraph.get_text(" ", strip=True)
            for paragraph in container.find_all("p")
            if paragraph.get_text(" ", strip=True)
        ]
        if paragraphs:
            break

    text = "\n\n".join(dict.fromkeys(paragraphs)).strip()
    if len(text) < 40:
        raise ArticleFetchError(
            "Isi artikel tidak ditemukan. Pastikan URL mengarah ke halaman berita Detik.com."
        )

    return Article(title=title or "Berita Detik.com", text=text)


def fetch_article(url: str) -> Article:
    current_url = validate_detik_url(url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        for redirect_count in range(6):
            with requests.get(
                current_url,
                headers=headers,
                timeout=(5, 15),
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.status_code in REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if not location or redirect_count == 5:
                        raise ArticleFetchError("Terlalu banyak pengalihan atau URL tujuan tidak tersedia.")
                    current_url = validate_detik_url(urljoin(current_url, location))
                    continue

                if response.status_code != 200:
                    raise ArticleFetchError(
                        f"Halaman berita gagal dimuat (HTTP {response.status_code})."
                    )

                content_type = response.headers.get("Content-Type", "").lower()
                if content_type and "html" not in content_type:
                    raise ArticleFetchError("URL tersebut tidak mengarah ke halaman HTML berita.")

                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > MAX_RESPONSE_BYTES:
                            raise ArticleFetchError("Halaman berita terlalu besar untuk diproses.")
                    except ValueError as exc:
                        raise ArticleFetchError("Ukuran respons halaman tidak valid.") from exc

                chunks: list[bytes] = []
                received_bytes = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    received_bytes += len(chunk)
                    if received_bytes > MAX_RESPONSE_BYTES:
                        raise ArticleFetchError("Halaman berita terlalu besar untuk diproses.")
                    chunks.append(chunk)
                return parse_article_html(b"".join(chunks))
    except requests.RequestException as exc:
        raise ArticleFetchError("Tidak dapat mengakses halaman berita. Coba lagi beberapa saat.") from exc

    raise ArticleFetchError("Halaman berita tidak dapat diproses.")


def preprocess_article(text: str, stopwords: set[str], stemmer: object) -> list[str]:
    cleaned_text = text.lower()
    cleaned_text = re.sub(r"[^\w\s]", " ", cleaned_text)
    tokens = cleaned_text.split()
    tokens = [token for token in tokens if token not in stopwords]
    stemmed_text = stemmer.stem(" ".join(tokens))
    return stemmed_text.split()


@st.cache_resource
def load_artifacts():
    missing = [
        path.name for path in (CLASSIFIER_PATH, WORD2VEC_PATH) if not path.is_file()
    ]
    if missing:
        raise ModelArtifactError(
            "File model tidak ditemukan: "
            + ", ".join(missing)
            + ". Letakkan kedua file model di folder tugas."
        )

    classifier = joblib.load(CLASSIFIER_PATH)
    word2vec = Word2Vec.load(str(WORD2VEC_PATH))
    expected_features = getattr(classifier, "n_features_in_", None)
    if expected_features != word2vec.vector_size:
        raise ModelArtifactError(
            "Dimensi model tidak cocok: "
            f"Naive Bayes mengharapkan {expected_features} fitur, "
            f"sedangkan Skip-gram menghasilkan {word2vec.vector_size}."
        )

    stopwords = set(StopWordRemoverFactory().get_stop_words())
    stemmer = StemmerFactory().create_stemmer()
    return classifier, word2vec, stopwords, stemmer


def document_vector(tokens: list[str], word2vec: Word2Vec) -> np.ndarray:
    known_vectors = [word2vec.wv[token] for token in tokens if token in word2vec.wv]
    if not known_vectors:
        raise ValueError(
            "Tidak ada kata artikel yang cocok dengan kosakata model. "
            "Artikel ini belum dapat diklasifikasikan."
        )
    return np.mean(known_vectors, axis=0).reshape(1, -1)


def display_category(category: object) -> str:
    labels = {"sport": "SPORT (Olahraga)", "finance": "FINANCE (Keuangan)"}
    return labels.get(str(category).lower(), str(category).upper())


APP_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --ink: #202925;
    --muted: #737b74;
    --paper: #f5f4ef;
    --panel: #ffffff;
    --line: #e5e5dc;
    --accent: #bf4938;
    --green: #255b49;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 88% 1%, rgba(214, 223, 207, .58), transparent 29rem),
        var(--paper);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1.5rem; }
.block-container {
    max-width: 1120px;
    padding: 2.2rem 2.2rem 4rem;
}

[data-testid="stVerticalBlock"] { gap: 1rem; }
.masthead {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.05rem;
    border-bottom: 1px solid #d8dbd1;
    color: var(--ink);
}
.brand {
    display: flex;
    align-items: center;
    gap: .7rem;
    font-family: 'Manrope', sans-serif;
    font-size: .9rem;
    font-weight: 800;
    letter-spacing: .025em;
}
.brand-mark {
    display: grid;
    width: 2rem;
    height: 2rem;
    place-items: center;
    border-radius: 7px;
    background: var(--ink);
    color: #fff;
    font-size: .85rem;
}
.masthead-note {
    color: var(--muted);
    font-size: .75rem;
    letter-spacing: .09em;
    text-transform: uppercase;
}

.hero { padding: 2.3rem 0 1.2rem; }
.eyebrow {
    margin-bottom: .65rem;
    color: var(--accent);
    font-size: .73rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
}
.hero h1 {
    max-width: 690px;
    margin: 0;
    color: var(--ink);
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2.3rem, 5vw, 3.7rem);
    font-weight: 800;
    letter-spacing: -.065em;
    line-height: 1.08;
}
.hero-copy {
    max-width: 570px;
    margin: .9rem 0 0;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.7;
}
.category-chips { display: flex; gap: .5rem; margin-top: 1.2rem; }
.category-chip {
    padding: .38rem .7rem;
    border: 1px solid #dfded5;
    border-radius: 999px;
    background: rgba(255,255,255,.55);
    color: #525a53;
    font-size: .77rem;
    font-weight: 600;
}

[data-testid="stHorizontalBlock"] { gap: 1.15rem; }
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line) !important;
    border-radius: 15px !important;
    background: var(--panel);
    box-shadow: 0 8px 28px rgba(38, 43, 35, .035);
}
.panel-heading {
    margin: .2rem 0 .2rem;
    color: var(--ink);
    font-family: 'Manrope', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -.025em;
}
.panel-copy {
    margin: 0 0 .65rem;
    color: var(--muted);
    font-size: .88rem;
    line-height: 1.6;
}
.step-label {
    display: inline-block;
    margin-bottom: .45rem;
    color: var(--accent);
    font-size: .69rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}
[data-testid="stTextInput"] label {
    color: var(--ink);
    font-size: .83rem;
    font-weight: 700;
}
[data-testid="stTextInput"] input {
    min-height: 3rem;
    border-color: #dbded7;
    border-radius: 9px;
    background: #fbfbf8;
    color: var(--ink);
}
[data-testid="stTextInput"] input::placeholder {
    color: #858d85;
    opacity: 1;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--green);
    box-shadow: 0 0 0 1px var(--green);
}
[data-testid="stFormSubmitButton"] button,
[data-testid="stButton"] button {
    min-height: 2.8rem;
    border: 0;
    border-radius: 9px;
    background: var(--ink);
    color: #fff;
    font-weight: 700;
    transition: background .15s ease, transform .15s ease;
}
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stButton"] button:hover {
    border: 0;
    background: var(--green);
    color: #fff;
    transform: translateY(-1px);
}
.method-item {
    display: flex;
    gap: .75rem;
    padding: .9rem 0;
    border-bottom: 1px solid #efefe9;
}
.method-item:last-child { border-bottom: 0; }
.method-num {
    display: grid;
    flex: 0 0 1.65rem;
    height: 1.65rem;
    place-items: center;
    border-radius: 50%;
    background: #f5eee7;
    color: var(--accent);
    font-size: .71rem;
    font-weight: 800;
}
.method-title { color: var(--ink); font-size: .85rem; font-weight: 700; }
.method-copy {
    margin-top: .15rem;
    color: var(--muted);
    font-size: .77rem;
    line-height: 1.5;
}
.model-note {
    margin-top: .55rem;
    padding: .8rem .9rem;
    border-radius: 9px;
    background: #f2f4ef;
    color: #58645b;
    font-size: .75rem;
    line-height: 1.55;
}
.result-shell { padding-top: .5rem; }
.result-kicker {
    color: var(--muted);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.result-card {
    margin-top: .5rem;
    padding: 1.35rem 1.5rem;
    border: 1px solid #dce5dc;
    border-left: 5px solid var(--green);
    border-radius: 12px;
    background: #f0f5ef;
}
.result-card.sport { border-color: #eadbd5; border-left-color: var(--accent); background: #faf1ed; }
.result-class {
    margin: .35rem 0 .2rem;
    color: var(--green);
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.5rem, 3vw, 2.1rem);
    font-weight: 800;
    letter-spacing: -.045em;
}
.result-card.sport .result-class { color: var(--accent); }
.result-summary { color: #626d63; font-size: .87rem; }
.article-link {
    display: inline-block;
    max-width: 100%;
    margin-top: .9rem;
    color: var(--ink) !important;
    font-size: .95rem;
    font-weight: 700;
    overflow-wrap: anywhere;
    text-decoration-color: #aab2a8 !important;
    text-underline-offset: 3px;
}
[data-testid="stProgressBar"] > div > div { background: var(--green); }
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #59635b !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: var(--ink) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: var(--ink) !important;
}
[data-testid="stExpander"] summary {
    background: #f0f1ec !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"],
[data-testid="stExpander"] [data-testid="stText"],
[data-testid="stExpander"] [data-testid="stText"] *,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: var(--ink) !important;
}
[data-testid="stExpander"] [data-testid="stCode"] pre,
[data-testid="stExpander"] [data-testid="stCode"] code,
[data-testid="stExpander"] [data-testid="stCode"] code * {
    color: #FFFFFF !important;
}
div[data-testid="stExpander"] {
    border-color: var(--line);
    border-radius: 10px;
    background: rgba(255,255,255,.75);
}
.footer-note {
    padding-top: .8rem;
    border-top: 1px solid #d8dbd1;
    color: var(--muted);
    font-size: .74rem;
}

@media (max-width: 700px) {
    .block-container { padding: 1.2rem 1rem 2.5rem; }
    .masthead-note { font-size: .62rem; }
    .hero { padding-top: 1.7rem; }
    .hero h1 { font-size: 2.45rem; }
    [data-testid="stHorizontalBlock"] { gap: .65rem; }
}
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="Ruang Berita | Klasifikasi Detik.com",
        page_icon="R",
        layout="wide",
    )
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="masthead">
            <div class="brand"><span class="brand-mark">R.</span> RUANG BERITA</div>
            <div class="masthead-note">Laboratorium klasifikasi berita · 2026</div>
        </div>
        <section class="hero">
            <div class="eyebrow">Baca polanya, kenali beritanya</div>
            <h1>Satu tautan.<br>Perspektif baru.</h1>
            <p class="hero-copy">
                Tempelkan artikel Detik.com dan biarkan model bahasa kami membaca
                konteksnya. Cepat, sederhana, dan transparan.
            </p>
            <div class="category-chips">
                <span class="category-chip">01 &nbsp; Sport</span>
                <span class="category-chip">02 &nbsp; Finance</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    input_column, method_column = st.columns([1.12, 0.88], gap="large")
    with input_column:
        with st.container(key="analysis_panel", border=True):
            st.markdown(
                """
                <span class="step-label">Mulai di sini · 01</span>
                <div class="panel-heading">Masukkan tautan berita</div>
                <p class="panel-copy">
                    Gunakan URL artikel lengkap dari Detik.com untuk mendapatkan
                    hasil klasifikasi.
                </p>
                """,
                unsafe_allow_html=True,
            )
            with st.form("article-form"):
                url = st.text_input(
                    "URL artikel",
                    placeholder="https://sport.detik.com/...",
                    help="Hanya URL HTTPS dari detik.com atau subdomain resminya yang diproses.",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button(
                    "Analisis berita  →",
                    type="primary",
                    use_container_width=True,
                )

    with method_column:
        with st.container(key="method_panel", border=True):
            st.markdown(
                """
                <span class="step-label">Di balik layar · 02</span>
                <div class="panel-heading">Bagaimana cara kerjanya?</div>
                <p class="panel-copy">Teks artikel melalui tiga tahap sebelum hasil ditampilkan.</p>
                <div class="method-item">
                    <span class="method-num">1</span>
                    <div><div class="method-title">Ekstraksi artikel</div>
                    <div class="method-copy">Mengambil judul dan isi berita langsung dari halaman Detik.com.</div></div>
                </div>
                <div class="method-item">
                    <span class="method-num">2</span>
                    <div><div class="method-title">Pembersihan teks</div>
                    <div class="method-copy">Case folding, penghapusan stopword, lalu stemming bahasa Indonesia.</div></div>
                </div>
                <div class="method-item">
                    <span class="method-num">3</span>
                    <div><div class="method-title">Prediksi kategori</div>
                    <div class="method-copy">Vektor Skip-gram dibaca oleh Gaussian Naive Bayes.</div></div>
                </div>
                <div class="model-note">
                    MODEL &nbsp;·&nbsp; Skip-gram 50 dimensi + Gaussian Naive Bayes
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not submitted:
        st.markdown(
            '<p class="footer-note">Eksperimen pembelajaran mesin untuk klasifikasi artikel berita Indonesia.</p>',
            unsafe_allow_html=True,
        )
        return

    try:
        with st.spinner("Mengambil dan menganalisis artikel..."):
            article = fetch_article(url)
            classifier, word2vec, stopwords, stemmer = load_artifacts()
            tokens = preprocess_article(article.text, stopwords, stemmer)
            features = document_vector(tokens, word2vec)
            prediction = classifier.predict(features)[0]
            probabilities = classifier.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
    except ArticleFetchError as exc:
        st.error(str(exc))
        return
    except ModelArtifactError as exc:
        st.error(str(exc))
        return
    except FileNotFoundError as exc:
        st.error(f"File model tidak dapat dibuka: {exc}")
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    category_key = str(prediction).lower()
    category_name = escape(display_category(prediction))
    article_title = escape(article.title)
    article_url = escape(validate_detik_url(url), quote=True)
    result_class = "sport" if category_key == "sport" else "finance"
    st.markdown(
        f"""
        <section class="result-shell">
            <div class="result-kicker">Hasil pembacaan model · 03</div>
            <div class="result-card {result_class}">
                <div class="result-kicker">Kategori terdeteksi</div>
                <div class="result-class">{category_name}</div>
                <div class="result-summary">Tingkat keyakinan model: <strong>{confidence:.1%}</strong></div>
                <a class="article-link" href="{article_url}" target="_blank" rel="noopener noreferrer">{article_title} ↗</a>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence, text=f"Keyakinan klasifikasi · {confidence:.1%}")

    finance_probability = next(
        (float(probability) for label, probability in zip(classifier.classes_, probabilities) if str(label).lower() == "finance"),
        0.0,
    )
    sport_probability = next(
        (float(probability) for label, probability in zip(classifier.classes_, probabilities) if str(label).lower() == "sport"),
        0.0,
    )
    score_columns = st.columns(2)
    with score_columns[0]:
        st.metric("Finance", f"{finance_probability:.1%}")
    with score_columns[1]:
        st.metric("Sport", f"{sport_probability:.1%}")

    with st.expander("Jelajahi teks yang dianalisis"):
        st.markdown("**Isi artikel**")
        st.text(article.text)
        st.markdown(f"**Token hasil preprocessing · {len(tokens)} kata**")
        st.code(" ".join(tokens) or "(tidak ada token)", language="text")


if __name__ == "__main__":
    main()
