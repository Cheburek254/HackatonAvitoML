# GRU-DIN-XL · часть 3 · Дима

Код обучения `e5d4584` в ветке `codex/v3-robust-sync`. Две эпохи на 1 050 408 строках лент train со смещением 2 100 800. Лучший чекпоинт выбран по полной NDCG на `timestamp < 1790107648` на 8-м шаге (третий чанк второй эпохи). Метрики на ранней, поздней и полной val равны 0,36732243, 0,36904534 и 0,36817451.

```powershell
.\.venv\Scripts\python.exe -u neural.py train --data .\competition --output .\runs\gru-din-xl-p3 --variant gru-din-xl --train-offset 2100800 --train-feeds 1050408 --chunk-feeds 250000 --val-feeds 0 --epochs 2 --history-window 256 --batch-size 16 --accum-steps 2 --tune-before 1790107648 --gpu 0 --threads 8 --prefetch
.\.venv\Scripts\python.exe -u neural.py predict --data .\competition --model .\runs\gru-din-xl-p3\model.pt --output .\runs\gru-din-xl-p3\test_predictions.parquet --gpu 0 --batch-size 16 --threads 8
```

Для обучения использованы статистики train из `runs/global_stats/item_stats.parquet` и `meta.json`. Для test использованы `item_stats_with_val.parquet` и `meta_with_val.json`. Готовые прогнозы передаются через зашифрованный раунд `gru-din-xl-full`, часть 003.
