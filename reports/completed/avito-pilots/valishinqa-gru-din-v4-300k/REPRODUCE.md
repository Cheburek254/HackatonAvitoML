# Воспроизведение `gru-din-v4` (300k, 5 эпох, BF16 + глобальные статистики)

Ветка/коммит: `feat/lgbm-and-gru-din-v3` (`b47fdcb`)

## Шаг 1. Собрать глобальные статистики по всем 3.15 млн лент `train` (~75 с)
```powershell
.\.venv\Scripts\python.exe build_global_stats.py --data .\competition --output .\runs\global_stats
```

## Шаг 2. Обучить `gru-din-v4` (300k лент, окно 64, 5 эпох, BF16 + prefetch)
```powershell
.\.venv\Scripts\python.exe experiments.py benchmark --data .\competition --output .\runs\gru-v4-300k --models gru-din-v4 --train-feeds 300000 --val-feeds 0 --epochs 5 --batch-size 16 --history-window 64 --prefetch-neural --gpu 0
```

## Шаг 3. Fine-tune на всём `val` (3 эпохи, `lr=1e-4`, `CosineAnnealingLR`) и Z-бленд `sub-v4`
* `40%` `din-events-v3-qsm` (200k)
* `30%` `gru-din-v4` (300k + fine-tune 3 эпохи на всём `val`)
* `15%` `gru-din-v3` (200k + fine-tune 3 эпохи на всём `val`)
* `15%` `lgbm-din-events-v3` (200k, `lr=0.02`)
