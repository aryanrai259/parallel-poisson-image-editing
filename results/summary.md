# Results

Machine: AMD Ryzen 7 6800H with Radeon Graphics, 8 cores / 16 threads; GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 546.92, 6144 MiB.
Mode: full, 3 repeats per configuration (medians).

## E1 Backends at the largest size

| input | backend | time (s) | speedup vs gcc |
|---|---|---|---|
| square10 | Serial C++ (gcc) | 23.8344 | 1.0x |
| square10 | OpenMP | 5.6613 | 4.2x |
| square10 | MPI | 13.5544 | 1.8x |
| square10 | OpenACC (multicore) | 11.2706 | 2.1x |
| square10 | OpenACC (GPU) | 1.4625 | 16.3x |
| square10 | CUDA | 0.9221 | 25.8x |
| circle10 | Serial C++ (gcc) | 23.4039 | 1.0x |
| circle10 | OpenMP | 5.6646 | 4.1x |
| circle10 | MPI | 14.1275 | 1.7x |
| circle10 | OpenACC (multicore) | 11.5072 | 2.0x |
| circle10 | OpenACC (GPU) | 1.4152 | 16.5x |
| circle10 | CUDA | 0.9255 | 25.3x |

## E2 OpenMP strong scaling

equ: serial fraction (Amdahl fit, up to physical cores) = 0.323

| threads | time (s) | speedup | efficiency |
|---|---|---|---|
| 1 | 14.4405 | 1.00 | 1.00 |
| 2 | 9.2309 | 1.56 | 0.78 |
| 4 | 6.3956 | 2.26 | 0.56 |
| 6 | 6.6406 | 2.17 | 0.36 |
| 8 | 6.0880 | 2.37 | 0.30 |
| 12 | 5.9138 | 2.44 | 0.20 |
| 16 | 7.4128 | 1.95 | 0.12 |

grid: serial fraction (Amdahl fit, up to physical cores) = 0.298

| threads | time (s) | speedup | efficiency |
|---|---|---|---|
| 1 | 38.8022 | 1.00 | 1.00 |
| 2 | 24.5452 | 1.58 | 0.79 |
| 4 | 17.5918 | 2.21 | 0.55 |
| 6 | 15.5943 | 2.49 | 0.41 |
| 8 | 16.0370 | 2.42 | 0.30 |
| 12 | 14.9346 | 2.60 | 0.22 |
| 16 | 16.8436 | 2.30 | 0.14 |

## E3 OpenMP schedules (time relative to static default)

equ: static/default 1.00, static/64 1.80, static/1024 1.19, static/16384 1.01, dynamic/64 1.87, dynamic/1024 1.12, dynamic/16384 0.97, guided/64 0.99, guided/1024 0.98, guided/16384 0.95
grid: static/default 1.00, static/1 1.01, static/4 1.01, static/32 1.01, dynamic/1 1.04, dynamic/4 0.93, dynamic/32 1.01, guided/1 0.90, guided/4 0.89, guided/32 0.92

## E4 MPI

1 processes: 24.7927 s (1.00x)
2 processes: 12.5770 s (1.97x)
4 processes: 14.0309 s (1.77x)
8 processes: 14.2404 s (1.74x)
sync every 10: 24.3769 s
sync every 100: 14.2404 s
sync every 1000: 13.1962 s

## E5 CUDA launch geometry

block 32: 0.9238 s
block 64: 0.9204 s
block 128: 0.9278 s
block 256: 0.9297 s
block 512: 0.9220 s
block 1024: 0.9147 s
tile 4x4: 2.0022 s
tile 8x8: 1.2902 s
tile 16x16: 1.3953 s
tile 32x32: 1.7372 s

## E7 Accuracy against the exact solution (20000 iterations)

| backend | method | PSNR (dB) | max abs | mean abs |
|---|---|---|---|---|
| CUDA | equ | 71.8 | 7 | 0.005 |
| CUDA | grid | 71.8 | 7 | 0.005 |
| Serial C++ (gcc) | equ | 72.7 | 2 | 0.004 |
| Serial C++ (gcc) | grid | 72.7 | 2 | 0.004 |
| MPI | equ | 56.2 | 30 | 0.448 |
| OpenACC (GPU) | equ | 72.8 | 2 | 0.004 |
| OpenACC (GPU) | grid | 71.8 | 7 | 0.005 |
| OpenACC (multicore) | equ | 72.8 | 2 | 0.004 |
| OpenACC (multicore) | grid | 71.8 | 7 | 0.005 |
| OpenMP | equ | 72.8 | 2 | 0.004 |
| OpenMP | grid | 71.8 | 7 | 0.005 |
