# GRU-DIN-XL · часть 2 · Кирилл

Код обучения `e5d4584` в ветке `codex/v3-robust-sync`. Две эпохи на 1 050 400 строках лент train со смещением 1 050 400. Лучший чекпоинт выбран по полной NDCG на `timestamp < 1790107648` после второго чанка второй эпохи. Метрики на ранней, поздней и полной val равны 0,36891478, 0,37024151 и 0,36957092.

```powershell
.\.venv\Scripts\python.exe team_reports.py pull --round gru-din-xl-full --with-artifacts
.\.venv\Scripts\python.exe neural.py train --data C:/study/AvitoData/competition --output runs/gru-din-xl-p2 --variant gru-din-xl --train-offset 1050400 --train-feeds 1050400 --chunk-feeds 250000 --val-feeds 0 --epochs 2 --history-window 256 --batch-size 16 --accum-steps 2 --tune-before 1790107648 --gpu 0 --threads 8 --prefetch
.\.venv\Scripts\python.exe neural.py predict --data C:/study/AvitoData/competition --model runs/team-reports/gru-din-xl-full/part-002/model.pt --output runs/gru-din-xl-p2/test_predictions.parquet --gpu 0 --batch-size 16 --threads 8
```

Для обучения `neural.py` читает статистики train из `runs/global_stats/item_stats.parquet` и `meta.json`. Для test при наличии пары `item_stats_with_val.parquet` и `meta_with_val.json` использует её. Команда должна согласовать одни и те же версии статистик для всех частей. Готовые прогнозы передаются через зашифрованный раунд `gru-din-xl-full`, часть 002.
