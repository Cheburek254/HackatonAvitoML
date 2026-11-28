# GRU-DIN-XL · часть 1 · Савва

Код обучения `bbd85bf` в ветке `codex/v3-robust-sync`. Две эпохи на 1 050 400 строках лент train со смещением 0. Лучший чекпоинт выбран по полной NDCG на `timestamp < 1790107648` (step 8). Метрики на ранней, поздней и полной val равны 0.36795080, 0.370690 и 0.36930530.

```powershell
.\.venv\Scripts\python.exe team_reports.py pull --round gru-din-xl-full --with-artifacts
.\.venv\Scripts\python.exe neural.py train --data competition --output runs/gru-din-xl-p1 --variant gru-din-xl --train-offset 0 --train-feeds 1050400 --chunk-feeds 250000 --val-feeds 0 --epochs 2 --history-window 256 --batch-size 16 --accum-steps 2 --tune-before 1790107648 --gpu 0 --threads 8 --prefetch
.\.venv\Scripts\python.exe neural.py predict --data competition --model runs/team-reports/gru-din-xl-full/part-001/model.pt --output runs/gru-din-xl-p1/test_predictions.parquet --gpu 0 --batch-size 16 --threads 8
```

Для обучения `neural.py` читает статистики train из `runs/global_stats/item_stats.parquet` и `meta.json`. Для test при наличии пары `item_stats_with_val.parquet` и `meta_with_val.json` использует её.
