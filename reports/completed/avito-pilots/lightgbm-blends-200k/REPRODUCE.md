# Воспроизведение Z-блендов с LightGBM

Используйте код из коммита в `manifest.json` и полный датасет Авито с 14 423 лентами val. Скачайте команды через `team_reports.py pull --round avito-pilots --with-artifacts`. Для пересчёта понадобятся локальные чекпоинты шести частей CatBoost `din-events-v2` для 1 050 403 лент, GRU Plus для первых 200 000 лент с окном истории 32 и LightGBM LambdaRank для первых 200 000 лент. Файлы моделей и прогнозов хранятся локально. Ниже приведены ожидаемые имена и контрольные суммы весов.

LightGBM обучен командой из [README](../../../../README.md) с лимитом 600 итераций, 8 потоками CPU, границей ранней val `1790107648` и параметром `--opencl 1 0`, который на этой машине выбирает RTX 4060. Для GRU и CatBoost использованы файлы завершённого прогона [avito-third-v1](https://github.com/Cheburek254/HackatonAvitoML/blob/f927f33182a4417f7c84718e4bc40522897add96/reports/completed/avito-third-v1/cat-gru-blends.md).

Чтобы восстановить оценку без переобучения, разместите локальные чекпоинты по указанным путям относительно папки отчёта и выполните команды из корня проекта. При необходимости замените путь к датасету.

```powershell
$env:AVITO_REPLAY_DATA = 'C:/study/AvitoData/competition'
$env:AVITO_REPLAY_WEIGHTS = (Resolve-Path 'runs/team-reports/avito-pilots/lightgbm-blends-200k/checkpoints').Path
@'
from pathlib import Path
from types import SimpleNamespace
import json
import os
import polars as pl
import torch
from accelerator import select_gpu
from best_submission import tree_predictions
from evaluation import KEYS, attach_scores, ndcg
from experiments import validation
from lgbm_ranker import load_model, to_numpy_features
from lightgbm_blend import CAT, GRU, LGB, compare
from neural import Feeds, Ranker, loader, scores
from ranker import candidate_batches

data = Path(os.environ["AVITO_REPLAY_DATA"])
weights = Path(os.environ["AVITO_REPLAY_WEIGHTS"])
root = Path("runs/lightgbm-replay")
root.mkdir(exist_ok=False)
select_gpu("0")
torch.set_num_threads(6)
torch.set_float32_matmul_precision("high")
labels, cutoff = validation(data, 0)
val_feeds = labels["feed_id"].n_unique()
meta = {"complete": True, "val_feeds": val_feeds, "tuning_cutoff_timestamp": cutoff}
cat_path, gru_path, lgb_path = root/"cat"/CAT, root/"gru"/GRU, root/"lgb"/LGB
for folder in (cat_path, gru_path, lgb_path):
    folder.mkdir(parents=True)
cats = sorted((weights/"catboost").glob("*/model.cbm"))
assert len(cats) == 6
cat = tree_predictions(data/"val", cats, 6, "gpu")
cat.write_parquet(cat_path/"val_predictions.parquet")
(cat_path.parent/"report.json").write_text(json.dumps(dict(meta, train_feeds=1050403)))

checkpoint = torch.load(weights/"gru/model.pt", map_location="cpu", weights_only=True)
gru = Ranker(checkpoint["variant"]).cuda()
gru.load_state_dict(checkpoint["state_dict"])
feeds = Feeds(data/"val", 0, True, checkpoint["history_window"], True, rich_features=True)
scores(gru, loader(feeds, 16), "cuda").write_parquet(gru_path/"val_predictions.parquet")
(gru_path.parent/"report.json").write_text(json.dumps(dict(meta, train_feeds=200000)))
del feeds, gru

lgbm = load_model(weights/"lightgbm/model.txt")
features = lgbm.feature_name()
chunks = []
for frame in candidate_batches(data/"val", features=features):
    prediction = lgbm.predict(to_numpy_features(frame, features), num_threads=8)
    chunks.append(frame.select(KEYS).with_columns(pl.Series("score", prediction, dtype=pl.Float32)))
lgb_scores = pl.concat(chunks)
lgb_scores.write_parquet(lgb_path/"val_predictions.parquet")
lgb_meta = dict(meta, model=LGB, train_feeds=200000, best_iteration=lgbm.current_iteration(),
                val=ndcg(attach_scores(labels, lgb_scores)))
(lgb_path/"metrics.json").write_text(json.dumps(lgb_meta))

output = root/"comparison"
output.mkdir()
compare(SimpleNamespace(data=data, catboost=cat_path, gru=gru_path, lgbm=lgb_path, gpu="0"), output)
'@ | .\.venv\Scripts\python.exe -u -X utf8 -
```

Проверка идёт по полной NDCG. На ранних 7 290 лентах подбираются веса новых смесей с шагом 0,05, на поздних 7 133 лентах сравниваются выбранные варианты. Контроль использует ранее подобранные веса 70% CatBoost и 30% GRU. Повторный расчёт на другой платформе может дать небольшие различия в округлении скоров.

### SHA-256 локальных чекпоинтов

- `checkpoints/catboost/part-001/model.cbm` · `d2a6255cd466c828e2fb1c8ccf6f6efaa59cf515f839a31d0b1f0f577eb59d6b`
- `checkpoints/catboost/part-002/model.cbm` · `a29052e1f446e1a86eab5332a75a46676e60e6e829918d21fad350e5b017258d`
- `checkpoints/catboost/part-003/model.cbm` · `e4ae8bcda9834cbcbaa675b4786dcf9ef462e678ebdbe62d6ad23db7fd43656e`
- `checkpoints/catboost/part-004/model.cbm` · `f2553142665343532c264071f127d7f861711d3551df9d6c364b771e047217f0`
- `checkpoints/catboost/part-005/model.cbm` · `bfea808276a3e04cd443853bdc212818298119abdbd1be86290f45a177085c65`
- `checkpoints/catboost/part-006/model.cbm` · `608aa9f9c77063544fe97ddd7888112f082b52747b6bbe2afc2270eb669bd0a6`
- `checkpoints/gru/model.pt` · `2667dddd86eca84e0718098dd4100fd4e5f9a93b06709747bb27756074a8ac64`
- `checkpoints/lightgbm/model.txt` · `59ea95a520ba3367fe73b8d63b6ada1bb553148e9683f006afa003dd548ebd65`
