# Воспроизведение `din-events-v3-qsm` (200k, 800 деревьев)

Ветка/коммит: `feat/lgbm-and-gru-din-v3` (`b47fdcb`)

## Шаг 1. Собрать глобальные статистики по всем 3.15 млн лент `train` (~75 с)
```powershell
.\.venv\Scripts\python.exe build_global_stats.py --data .\competition --output .\runs\global_stats
```

## Шаг 2. Обучить `din-events-v3-qsm` (CatBoost `QuerySoftMax`, `depth=7`, 800 деревьев)
```powershell
.\.venv\Scripts\python.exe experiments.py benchmark --data .\competition --output .\runs\bench-v3-200k --models din-events-v3-qsm --train-feeds 200000 --val-feeds 0 --iterations 800 --gpu 0
```

## Шаг 3. Предсказание на `test`
```powershell
.\.venv\Scripts\python.exe ranker.py predict --data .\competition --model .\runs\bench-v3-200k\din-events-v3-qsm\model.cbm --output .\runs\sub-v4\qsm_test.parquet --device gpu --gpu-id 0 --threads 6
```
