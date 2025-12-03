# Pengujian Algoritma Task Scheduler pada Server IT - j2020 vs. SHC, RR, FCFS

**Kelas A - Kelompok E**

| Nama | NRP |
| -- | -- |
| Nathan Kho Pancras | 5027231002 |

Repo ini memuat kode server pengujian Task Scheduling pada Server IT serta implementasi algoritma penjadwalan **J2020** yang diadaptasi dari dokumen *Differential J2020* (`differential_j2020.pdf`) untuk kebutuhan mata kuliah **Strategi Optimasi Komputasi Awan (SOKA)**.

## Usage

1. Install `uv` sebagai dependency manager. Lihat [link berikut](https://docs.astral.sh/uv/getting-started/installation/)

2. Install semua requirement

```bash
uv sync
```

3. Buat file `.env` kemudian isi menggunakan variabel pada `.env.example`. Isi nilai setiap variabel sesuai kebutuhan

```conf
VM1_IP=""
VM2_IP=""
VM3_IP=""
VM4_IP=""

VM_PORT=5000
```

4. Scheduler utama kini menggunakan algoritma `J2020`. Implementasi ini mengikuti pendekatan pada *Differential J2020* dengan langkah-langkah:
	- Melakukan profiling VM menggunakan tugas probe.
	- Menghitung skor komposit (ECT, load balance, energi) untuk setiap pasangan tugas–VM.
	- Menentukan penempatan tugas dengan strategi *min-min* berbobot.
	Dokumentasi asli dapat dilihat pada `differential_j2020.pdf`. Selain J2020, repositori masih menyediakan algoritma pembanding (`Stochastic Hill Climbing`, `Round Robin`, `FCFS`) untuk eksperimen.

5. Untuk menjalankan server, jalankan docker

```bash
docker compose build --no-cache
docker compose up -d
```

6. Scheduler secara otomatis membaca tiga dataset contoh (`dataset-low-high.txt`, `dataset-rand.txt`, `dataset-rand-stratified.txt`). Anda dapat menyesuaikan isi file-file tersebut (nilai 1–10) sesuai kebutuhan eksperimen, atau mengarahkan ke dataset lain lewat variabel lingkungan.

7. Untuk menjalankan scheduler (menjalankan J2020 + algoritma pembanding pada ketiga dataset sebanyak `DATASET_ITERATIONS` kali, default 10), jalankan `scheduler.py`. **Jangan lupa menggunakan VPN / Wifi ITS**

```bash
uv run scheduler.py
```

8. Setelah selesai, output berikut akan tersedia:
	- `shc_results.csv`, `j2020_results.csv`, `rr_results.csv`, `fcfs_results.csv`: rata-rata metrik per algoritma & dataset.
	- `comparison_results.csv`: tabel perbandingan lintas algoritma.
	- `summary.txt`: ringkasan setiap iterasi.
	- `scheduler_run.log`: log JSONL berisi detail penjadwalan tiap tugas.
	Konsol tetap menampilkan metrik satu kali jalan untuk debugging cepat.