# PPW
Pencarian dan Penambangan Web

## Menjalankan klasifikasi berita

Aplikasi Streamlit untuk mengklasifikasikan artikel Detik.com tersedia di
[`tugas/app.py`](./tugas/app.py). Jalankan dari direktori utama repository:

```bash
streamlit run tugas/app.py
```

Untuk deploy ke Streamlit Community Cloud, hubungkan repository ini dan pilih
`tugas/app.py` sebagai **Main file path**. Dependensi aplikasi tercantum di
[`requirements.txt`](./requirements.txt). Pastikan kedua file model berikut
ikut tersedia di repository pada folder `tugas/`:

- `model_skipgram_naive_bayes_skenario2.joblib`
- `model_skipgram_skenario2.model`

Masukkan URL artikel berawalan `https://` dari domain `detik.com` atau
subdomain-nya, lalu pilih **Analisis Berita**.
