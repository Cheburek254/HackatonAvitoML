# Ранжирование лент Авито

Основная метрика соревнования — полная NDCG, усреднённая по лентам. Чем выше значение, тем лучше результат. В отчёте варианты отсортированы по NDCG на поздней части одной общей `val`. NDCG@10 и NDCG@30 служат для диагностики.

Из проекта Альфа-Банка перенесена идея более высокого веса свежих событий. Берём первые 32 события, уменьшаем вес вдвое каждые восемь позиций и учитываем каждый известный товар один раз при оценке интереса к категории и построении вектора истории. Числа 32 и 8 выбраны как стартовые параметры для проверки на `val`.

## Быстрый запуск на синтетике

Из корня клонированного репозитория запустите команды в PowerShell с Python 3.11 или 3.12. Генератор создаст данные локально. Один и тот же каталог используется в примерах и выбран по умолчанию в окне обучения.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe demo.py --data .\runs\synthetic-v2\competition
.\.venv\Scripts\python.exe experiments.py benchmark --data .\runs\synthetic-v2\competition --output .\runs\smoke-01 --models history category-trend --train-feeds 300 --val-feeds 150 --iterations 20
```

Эта проверка работает на CPU. Итоговая таблица сохраняется в `runs/smoke-01/report.json`. Для нового запуска с теми же данными укажите другой `--output`. Для окна обучения на NVIDIA установите `requirements-gpu.txt` и запустите `dashboard.py` из того же окружения.

По умолчанию генератор пишет 6 000 лент `train`, 1 500 лент `val` и 1 500 лент `test`. Структура включает отдельный `int8` memmap на сплит, Parquet с типами опубликованной схемы, `_SUCCESS` и шаблон сабмита. Ленты содержат 5–301 кандидата, истории — 0–256 событий. Есть пересекающиеся и новые ID, пропущенные эмбеддинги, перестановки кандидатов и несколько положительных примеров. Размеры и seed задаются аргументами `demo.py`. Запускайте генератор в новый каталог, чтобы сохранить предыдущий набор. Полученные NDCG относятся к синтетическому распределению.

## Запуск с архивом соревнования

Скачайте `competition.zip` со страницы соревнования и распакуйте его так, чтобы `--data` указывал на папку с `train/`, `val/`, `test/` и `sample_submission.parquet`. Для сравнения с официальными моделями скачайте `baseline.zip` отдельно и установите `requirements-experiments.txt`. [Файлы соревнования](https://ods.ai/competitions/avitotechrecsys2026/dataset)

```powershell
.\.venv\Scripts\python.exe experiments.py benchmark --data .\competition --baseline-zip "$env:USERPROFILE\Downloads\baseline.zip" --output .\runs\real-01 --models avito-catboost history din avito-transformer avito-joint --val-feeds 0
.\.venv\Scripts\python.exe ranker.py train --data .\competition --variant din --output .\runs\catboost-01
.\.venv\Scripts\python.exe ranker.py predict --data .\competition --model .\runs\catboost-01\model.cbm --output .\runs\catboost-01\submission.parquet
```

Сравнение запускает каждый вариант на одинаковом числе лент и проверяет совпадение всех пар `(feed_id, item_id)` на `val`. Граница настройки выбирается по медиане уникальных `timestamp`. Ленты с одинаковым временем остаются по одну сторону границы. Выбор чекпоинта и весов RRF выполняется на ранней части, выбранные варианты оцениваются на поздней. Оригинальный CatBoost использует для отбора итерации NDCG@30, оригинальный Transformer — Logloss, как в архиве. Варианты проекта выбирают чекпоинт по полной NDCG.

`report.json` содержит NDCG на всей `val`, ранней и поздней частях, разбивку поздней части по длине истории и время запуска. В полях `train_feed_limit` и `val_feed_limit` сохранены запрошенные лимиты, в `train_feeds` и `val_feeds` — фактические размеры. Для стекинга число использованных лент указано и у самих моделей, поскольку скор OOF есть только у части `train`. Каждый вариант сохраняет свою модель и `val_predictions.parquet`. Для нейросетей сабмит вычисляет `neural.py predict`, а для смеси двух готовых сабмитов — `experiments.py blend`. Веса и имена моделей смеси записаны в `report.json`.

`--train-feeds 0 --val-feeds 0` использует обе выборки целиком. Сначала стоит измерить расход RAM на ограниченной выборке. CatBoost читает эмбеддинги через memmap, а развёрнутые признаки для обучения собирает в RAM. Для сравнения на GPU есть `--cat-device gpu` и `--neural-device cuda`. Вариант `lambda` обучается на CPU.

## Три семейства и стекинг

Команда ниже запускает сравнение на реальных данных с одинаковыми лентами для всех моделей. Для нейросетей на NVIDIA установите CUDA-сборку PyTorch. [Сборки PyTorch](https://pytorch.org/get-started/previous-versions/#v2100)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-experiments.txt -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install "torch==2.10.0+cu126" --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe experiments.py benchmark --data .\competition --output .\runs\avito-ensemble --gpu 0 --models history category-trend strong-gbdt avito-joint set-transformer neural-pairwise gru-din sasrec --train-feeds 6000 --val-feeds 1500 --iterations 60 --epochs 2 --family-blends --stack
```

`strong-gbdt` объединяет тренд категории, SIM, направленные переходы, Pixie, динамику популярности и прогноз категории, обученный на более ранних лентах. В ансамбль входят `strong-gbdt` и лучшие по ранней `val` представители listwise- и последовательного семейств. Optuna подбирает веса Z-нормировки и RRF на ранней `val`. Там же проверяется штраф за повтор категории в первых 20 позициях. Сетка штрафов включает значение 0.

`din-events` расширяет `din` признаками типа и источника события из истории, весом свежести и максимумами сходства по типам событий. Для прямого сравнения на 200 000 лентах используйте `--models din din-events --train-feeds 200000 --val-feeds 0 --iterations 300 --gpu 0`. Позиция `pos` доступна только в `train`, поэтому `din-events-pos` служит для диагностики.

Для стекинга `avito-joint` и `gru-din` предсказывают более поздние блоки `train` после обучения на предыдущих блоках. `stack-gbdt` получает два нормированных скора. Контроль `stack-base` использует те же обучающие ленты и обычные признаки `strong-gbdt`. Все итоговые сравнения выполняются по полной NDCG на поздней `val`.

### RRF и Z-бленд двух обученных моделей

Чтобы повторить полный эксперимент на одном компьютере, сначала запустите CatBoost. Когда появится окно с первой частью, выполните вторую команду в другом терминале. Очередь дождётся объединения шести частей, обучит `gru-din-plus` на 200 000 лент за 5 эпох и переключит то же окно на сравнение. В Windows очередь предотвращает автоматический сон от простоя на зарядке. Блокировка экрана допустима. Действие при закрытии крышки задаётся в настройках питания Windows.

```powershell
.\.venv\Scripts\python.exe sharded_runs.py train --data C:/study/AvitoData/competition --output runs/avito-real/din-events-v2-third-600 --models din-events-v2 --train-feeds 1050403 --val-feeds 0 --shards 6 --iterations 600 --epochs 5 --gpu 0 --person all --ui --report-round avito-third-v1
.\.venv\Scripts\python.exe training_queue.py --data C:/study/AvitoData/competition --catboost runs/avito-real/din-events-v2-third-600 --output runs/avito-real/gru-pair-third --gpu 0 --report-round avito-third-v1
```

`pair_blend.py` принимает две папки моделей с `val_predictions.parquet` и отчётом `report.json` в родительской папке. Слева можно передать результат объединения частей CatBoost, справа — результат одного обучения GRU. Обе модели должны покрывать полную валидацию соревнования.

```powershell
.\.venv\Scripts\python.exe pair_blend.py --data C:/study/AvitoData/competition --left runs/avito-real/din-events-v2-third-600/merged/din-events-v2 --right runs/avito-real/gru-pair-third/gru-fit/gru-din-plus --output runs/avito-real/gru-pair-third/comparison --gpu 0
```

Для каждого способа скрипт перебирает вес правой модели от 0 до 1 с шагом 0,1 на ранней половине `val`. RRF использует ранги внутри ленты и константу 60. Z-бленд стандартизирует оценки каждой модели внутри ленты. В новой папке результата сохраняются предсказания обеих смесей и `report.json` с весами, полной NDCG на каждом временном срезе и парными 95%-ми интервалами разницы NDCG по пользователям поздней валидации.

### LightGBM и Z-бленд с сохранёнными прогнозами CatBoost и GRU

Пример использует LightGBM LambdaRank, признаки `din-events-v2` и первые 200 000 лент. Для GPU-бустинга добавьте `--opencl PLATFORM DEVICE` к команде обучения и проверьте название видеокарты в строке `Using GPU Device`. Нумерация OpenCL в LightGBM зависит от машины. Выбор числа деревьев идёт по NDCG на ранней `val` с отсечением на длине её самой длинной ленты. Для итогового сравнения `lightgbm_blend.py` считает полную NDCG и использует CUDA при указании `--gpu`.

```powershell
$data = 'C:/study/AvitoData/competition'
$run = 'runs/avito-real/lightgbm-200k-blend'
.\.venv\Scripts\python.exe lgbm_ranker.py train --data $data --output "$run/lightgbm" --train-feeds 200000 --iterations 600 --threads 8
.\.venv\Scripts\python.exe lightgbm_blend.py --data $data --catboost runs/avito-real/din-events-v2-third-600/merged/din-events-v2 --gru runs/avito-real/gru-pair-third/gru-fit/gru-din-plus --lgbm "$run/lightgbm" --output "$run/blend" --gpu 0
```

Контрольная смесь сохраняет веса 70% CatBoost и 30% GRU из предыдущего прогона. Для CatBoost + LightGBM и CatBoost + LightGBM + GRU перебираются неотрицательные веса с шагом 0,05 и суммой 1. Веса подбираются только на ранней `val`. Отчёт включает полную NDCG трёх смесей и парные интервалы разниц на поздней `val`.

Команду для сабмишена можно запустить в отдельном терминале, когда очередь уже работает. Она дождётся сравнения, выберет лучший из четырёх вариантов по полной NDCG поздней `val` и применит его к `test`. Для CatBoost все части используют общую подготовку признаков каждого тестового батча. Итоговые `submission.parquet` и `recipe.json` появятся в указанной папке после проверки формата и совпадения всех ключей с образцом Авито.

```powershell
.\.venv\Scripts\python.exe best_submission.py --data C:/study/AvitoData/competition --catboost runs/avito-real/din-events-v2-third-600 --queue runs/avito-real/gru-pair-third --output runs/avito-real/gru-pair-third/best --gpu 0 --wait
```

## Распределение по нескольким компьютерам

Командный патч входит в эту версию проекта. В существующей копии обновитесь командой `git pull --ff-only`. Файл `team-training.patch`, который передавался отдельно, рассчитан на исходный коммит `46895ba` и применяется через `git apply team-training.patch`. Если патч уже применён локально, для командного запуска можно использовать новый клон `main`.

Для сравнения семейств каждая машина обучает свои модели на одинаковых лентах. Бюджет по умолчанию — 200 000 лент `train`, все 14 423 ленты `val`, 120 итераций CatBoost и 12 эпох для нейросетей. Это план примерно на 10 часов при последовательном выполнении трёх ролей и сборки на RTX 4060 Laptop. Такой бюджет можно менять флагами `--train-feeds`, `--val-feeds`, `--iterations` и `--epochs`. Для быстрой проверки добавьте `--train-feeds 50000 --epochs 3` ко всем трём командам. Используйте одинаковые параметры, версию кода и архив Авито. Индекс NVIDIA указывается через `--gpu`. Для полного `train` используйте запуск по частям ниже.

| Участник | Роль и модели | Что передаёт для сборки |
|---|---|---|
| Кирилл | `gbdt` — `history`, `category-trend`, `pixie-decays`, `strong-gbdt` | Отчёт и скоры на `val` |
| Савва | `listwise` — `avito-joint`, `set-transformer`, `neural-pairwise` | Отчёт, скоры на `val` и OOF-скоры `avito-joint` |
| Димон | `sequence` — `gru-din`, `sasrec` | Отчёт, скоры на `val` и OOF-скоры `gru-din` |

На каждой машине из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-experiments.txt
.\.venv\Scripts\python.exe -m pip install "torch==2.10.0+cu126" --index-url https://download.pytorch.org/whl/cu126
$data = 'C:\study\AvitoData\competition'
$round = 'avito-families-v1'
```

**Кирилл** запускает CatBoost.

```powershell
.\.venv\Scripts\python.exe team_runs.py job --role gbdt --data $data --output .\runs\team-gbdt --gpu 0 --report-round $round
```

**Савва** запускает модели взаимодействия кандидатов внутри ленты.

```powershell
.\.venv\Scripts\python.exe team_runs.py job --role listwise --data $data --output .\runs\team-listwise --gpu 0 --report-round $round
```

**Димон** запускает модели последовательности действий.

```powershell
.\.venv\Scripts\python.exe team_runs.py job --role sequence --data $data --output .\runs\team-sequence --gpu 0 --report-round $round
```

Если запуск прервался после создания отчёта, повторите ту же команду с `--resume`.

Для сборки передайте внутри команды `job.json`, `report.json`, файлы `<модель>/val_predictions.parquet` и каталоги `oof/` от двух нейросетевых ролей. Для последующего предсказания сохраните также веса моделей и файлы `behavior.npz`. Храните эти результаты отдельно от публичного репозитория.

`team_runs.py merge` можно запустить на любой из трёх машин после получения результатов. С флагом `--stack` эта машина дополнительно обучит финальный CatBoost на двух OOF-скорах.

```powershell
.\.venv\Scripts\python.exe team_runs.py merge --data $data --runs .\runs\team-gbdt .\runs\team-listwise .\runs\team-sequence --output .\runs\team-merged --gpu 0 --stack --report-round $round
```

Сборка проверяет состав и порядок выбранных лент, бюджет обучения и пары `(feed_id, item_id)`, затем сравнивает Z-блендинг, RRF, штраф за повтор категории и стекинг на общей `val`.

### Полный train по частям

Команды используют все 3 151 208 лент `train` и всю `val`. Каждая из 21 части содержит около 150 000 лент. Чтение идёт напрямую из архива в распакованном виде, общие эмбеддинги остаются в исходном memmap. Каждая часть обучает свой экземпляр выбранной модели. Сборка усредняет Z-нормализованные внутри ленты скоры этих экземпляров на одинаковых кандидатах.

| Кто | Части | Всего лент train |
|---|---|---:|
| Кирилл | 1, 4, 7, 10, 13, 16, 19 | 1 050 403 |
| Савва | 2, 5, 8, 11, 14, 17, 20 | 1 050 403 |
| Димон | 3, 6, 9, 12, 15, 18, 21 | 1 050 402 |

После установки выше каждый выполняет свою команду. `--ui` открывает окно с текущей частью и оставляет его после обучения. `--predict-test` сразу сохраняет скоры для будущего сабмита.

```powershell
$data = 'C:\study\AvitoData\competition'
$round = 'avito-full-v1'

# Кирилл
.\.venv\Scripts\python.exe sharded_runs.py train --data $data --output .\runs\avito-full --person kirill --predict-test --ui --report-round $round

# Савва
.\.venv\Scripts\python.exe sharded_runs.py train --data $data --output .\runs\avito-full --person savva --predict-test --ui --report-round $round

# Димон
.\.venv\Scripts\python.exe sharded_runs.py train --data $data --output .\runs\avito-full --person dimon --predict-test --ui --report-round $round
```

По умолчанию обучается `strong-gbdt`. Для сравнения нескольких моделей добавьте, например, `--models strong-gbdt avito-joint gru-din` ко всем командам. Чтобы распределить миллион лент, добавьте `--train-feeds 1000000 --shards 9` и укажите новую папку `--output` и новый раунд. Команда `sharded_runs.py plan --data $data --train-feeds 1000000 --shards 9` покажет назначения до запуска. При целых частях, не помещающихся в память, увеличьте `--shards` для всей команды и начните новую серию в новом каталоге.

На RTX 4060 Laptop пилот `strong-gbdt` на 50 000 лент занял 3 мин 39 с при 120 деревьях. На 200 000 лент подготовка и короткий прогон из 20 деревьев заняли 11 мин 22 с с пиком около 14,5 ГиБ RAM. Эпоха `gru-din` на том же размере заняла 5 мин 56 с. Оценка полного прогона `strong-gbdt` с предсказанием `test` — 4–6 часов на одном ПК или 1,5–2,5 часа на трёх похожих машинах. Для девяти основных моделей вместе оценка составляет 30–45 часов на одном ПК или 10–16 часов на трёх при трёх эпохах. Время передачи файлов зависит от сети. Для режима по частям используются отдельные веса, а OOF-стекинг выполняется в режиме сравнения семейств выше.

Для девяти моделей добавьте к своей команде этот аргумент:

```powershell
--models history category-trend pixie-decays strong-gbdt avito-joint set-transformer neural-pairwise gru-din sasrec
```

Для полного прогона на одном ПК запустите ту же команду с `--person all`. Она обходит части по очереди и сама запускает сборку. Повтор команды продолжает тот же план и пропускает завершённые модели и части.

```powershell
.\.venv\Scripts\python.exe sharded_runs.py train --data $data --output .\runs\avito-full --person all --predict-test --ui --report-round $round
```

Для прогона около 10 часов на одном ПК возьмите `strong-gbdt` и `gru-din` на всех лентах. Оценка составляет 9–12 часов при трёх эпохах `gru-din`. Этот прогон использует свой каталог и раунд.

```powershell
.\.venv\Scripts\python.exe sharded_runs.py train --data $data --output .\runs\avito-full-10h --person all --models strong-gbdt gru-din --epochs 3 --predict-test --ui --report-round avito-full-10h-v1
```

При запуске на трёх ПК передайте Кириллу каталоги `parts/part-NNN` с `part.json`, `fit/report.json` и файлами `fit/<модель>/val_predictions.parquet` и `test_predictions.parquet`. После сбора в одной папке Кирилл запускает:

```powershell
.\.venv\Scripts\python.exe sharded_runs.py merge --data $data --output .\runs\avito-full --predict-test --report-round $round
```

Сборка проверяет покрытие всех назначенных частей и соответствие кандидатов. Общие результаты будут в `merged/report.json`, файлы сабмита — в `merged/<модель>/submission.parquet` внутри папки запуска. В отчёте есть время по каждой части, сумма времени и наибольший из замеренных пиков RAM. Общая подготовка признаков для всех 3,15 млн лент одновременно потребовала бы около 31 ГиБ только для 34 столбцов, ещё до создания CatBoost Pool.

### Отчёты команды на GitHub

При общем `--report-round` завершённые отчёты публикуются в ветках `team-reports/<раунд>/<роль-или-часть>` и подтягиваются в `runs/team-reports/<раунд>/` при завершении каждого задания. Нужны авторизованный Git и право записи в репозиторий. Обновить отчёты вручную можно в любой момент.

```powershell
.\.venv\Scripts\python.exe team_reports.py pull --round $round
```

Готовые отчёты по первым прогонам на реальных данных Авито собраны в раунде `avito-pilots`. Команда ниже скачивает их в `runs/team-reports/avito-pilots/`. В сводках указаны размеры выборок, поэтому NDCG прогона на 1 500 лент и прогона на всей `val` можно различить.

```powershell
.\.venv\Scripts\python.exe team_reports.py pull --round avito-pilots
```

Для ручных комментариев и рекомендаций сохранены отдельные формы, которые можно отправить через PR в `main`.

- Кирилл — [форма GBDT](reports/gbdt.md)
- Савва — [форма listwise](reports/listwise.md)
- Димон — [форма последовательных моделей](reports/sequence.md)

Числа берите из `report.json`. Лимиты и фактические размеры указаны в `train_feed_limit` / `val_feed_limit` и `train_feeds` / `val_feeds`. Для полной NDCG используйте поля `tuning.ndcg`, `holdout.ndcg` и `val.ndcg` из `models`. Время и пик RAM указаны в `seconds` и `peak_ram_mib`. Если модель завершилась с ошибкой, запишите этап и сообщение вместо метрики. Савва и Димон также отмечают наличие OOF-файла и число лент с OOF-скорами.

К готовому отчёту можно приложить комплект для воспроизведения. В `REPRODUCE.md` укажите версию датасета, размер и смещение train, разбиение val, команды обучения и предсказания, чекпоинты и формулу смешивания. Передайте `--revision` с коммитом обучения и по одному `--checkpoint` на каждый нужный файл относительно папки `report.json`. Для моделей с дополнительными параметрами включите нужные `behavior.npz` или `centers.npy`. Публикуйте только те веса, которые готовы выложить в публичный репозиторий. Сырые данные и Parquet остаются локально.

```powershell
.\.venv\Scripts\python.exe team_reports.py publish --round avito-pilots --name example --report runs/example/report.json --revision b45cfc4 --reproduce runs/example/REPRODUCE.md --checkpoint din-events-v2/model.cbm --checkpoint gru-din-plus/model.pt
.\.venv\Scripts\python.exe team_reports.py pull --round avito-pilots --with-artifacts
```

Файлы и команды появятся в папке отчёта на GitHub. При скачивании комплект сохраняется рядом с локальной сводкой в каталоге с именем отчёта. Скрипт проверяет размер и SHA‑256 каждого чекпоинта. Лимит составляет 90 МиБ на файл.

## Окно обучения на GPU

Для ускорения моделей из `neural.py` включите «Предзагрузка батчей нейросети» в окне или добавьте `--prefetch-neural` к команде `experiments.py benchmark`. Один фоновый поток подготавливает следующий батч на CPU одновременно с вычислениями текущего батча на GPU. Размер батча и порядок данных задаются обычными параметрами запуска.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe dashboard.py
```

В окне выберите дискретную NVIDIA GPU по названию и объёму VRAM, папку данных, модели, лимиты лент и бюджет обучения. По умолчанию выбраны тот же синтетический набор и все 11 вариантов GPU-сравнения. Каждое нажатие «Обучить и проверить» создаёт новый запуск в `runs/dashboard/`. Прогресс по деревьям и батчам виден в ходе обучения. График показывает полную NDCG на ранней `val` при проверке чекпоинтов. Таблица обновляется после каждой модели и сортируется по полной NDCG на поздней `val`. Через «Открыть отчёт» можно посмотреть завершённый запуск и выбрать модель для её графика.

Команда `dashboard.py --report ПУТЬ_К_REPORT_JSON` открывает сохранённый отчёт сразу при запуске окна.

Тот же прогон из PowerShell. Индекс или UUID карты показывает `nvidia-smi`.

```powershell
.\.venv\Scripts\python.exe experiments.py benchmark --data .\runs\synthetic-v2\competition --output .\runs\gpu-01 --gpu 0 --train-feeds 0 --val-feeds 0 --iterations 100 --epochs 5 --threads 6 --by-category
```

Если запуск прервался после сохранения промежуточного отчёта, повторите ту же команду с `--resume`. Готовые модели будут взяты из отчёта, прерванный вариант запустится заново.

`--gpu` закрепляет CUDA за выбранной картой по UUID. На GPU идут обучение, предсказания, подбор чекпоинтов по полной NDCG, итоговая NDCG, обучение модели категории и кластеризация эмбеддингов. Для 100 деревьев проверяются первое дерево и затем шаг в 5 деревьев. CPU читает Parquet и memmap, формирует табличные признаки и графовые индексы, собирает ключи и записывает артефакты. `dual-history` использует CUDA через tinygrad. Для запуска прежних PyTorch-моделей через `--gpu` требуется CUDA-сборка PyTorch. В отчёт входят время, пиковая рабочая память процесса, Парето-фронт по этим двум затратам и NDCG, а также разрез поздней `val` по доминирующей положительной категории ленты. Разрез включает до восьми категорий с минимум 25 лентами в каждой.

Для предсказания CatBoost на GPU добавьте к `ranker.py predict` аргументы `--device gpu --gpu-id 0`. Для двух окон истории используйте `dual_history.py predict --gpu-id 0` с путём к `model.safetensors`.

## Варианты экспериментов

| Вариант | Что проверяет |
|---|---|
| `avito-catboost`, `avito-transformer` | Код из исходного `baseline.zip` с общими лимитами и ранней частью `val` для выбора чекпоинта. |
| `avito-features`, `history` | Признаки организаторов при настройках нашего CatBoost и исходный прототип с рангами `f1–f5`, категорией и историей. |
| `din` | Максимум и среднее трёх лучших косинусов с 32 свежими событиями, давность лучшего совпадения и наличие пары эмбеддингов. |
| `sim` | Для каждого кандидата выбирает до 16 релевантных событий из всей истории, агрегирует сходство с поправкой на давность и считает долю совпадающей категории. |
| `multi-interest` | До четырёх профилей по категориям истории и сходство с каждым интересом, по мотивам MIND. |
| `popularity`, `itemcf`, `als` | Независимо добавляют сглаженные частоты и редкость категории, сходство по совместным взаимодействиям или оценку implicit ALS. |
| `covis` | Направленные переходы между объявлениями в историях пользователей по мотивам разбора победителей. Повтор пары у одного пользователя учитывается один раз. |
| `category-trend` | Доли категории кандидата в событиях 1–8 и 9–64 и разница между ними. Категория `0` пропускается. Разница равна нулю, если одно из окон пусто. |
| `pixie` | Взвешенные пути от последних восьми объявлений через соседних пользователей к кандидату и число таких соседей. |
| `category-trend-pixie` | Обе группы признаков в одном CatBoost, чтобы модель могла учитывать их взаимодействие. |
| `category-trend-wide` | Расширение окон интереса до 1–16 и 17–128 событий. |
| `category-model` | Логит самостоятельной логистической модели интереса к категории и ранг этой категории среди уникальных категорий ленты. |
| `popularity-trend` | Сглаженные частоты положительных действий по объявлению за сутки и за предшествующую неделю, разница и объём наблюдений. |
| `category-covis` | Направленные переходы между объявлениями внутри одной категории. |
| `pixie-decays` | Два графовых скора с полураспадами в 2 и 16 событий и их ранги внутри ленты. |
| `covis-pixie` | Прямые переходы `covis` и связи через других пользователей `pixie` в одном реранкере. |
| `embedding-clusters` | До 32 сферических кластеров эмбеддингов, сходство с центром и доля совпадающего кластера в истории. |
| `history-distill` | CatBoost с признаками `history` и мягкими метками учителя `category-trend`. Учитель оценивает более поздние блоки `train`, доля мягкой метки — 30%. |
| `dual-history` | Отдельные нейросетевые энкодеры событий 1–8 и 9–64. Кандидат взаимодействует с каждым представлением отдельно. |
| `lambda` | Использует те же признаки, что `din`, с целью LambdaMART вместо YetiRank. |
| `transformer`, `transformer-listmle` | Один Transformer по множеству кандидатов с двумя разными функциями потерь. |
| `neural-din`, `bst`, `gru-din`, `sasrec` | Отдельные энкодеры истории. Реализованы адаптации к готовым кандидатам, включая GRU + внимание по мотивам DIEN. |
| `set-transformer`, `avito-joint` | Внимание через восемь обучаемых векторов или совместное кодирование истории и кандидатов с ListMLE по публикации AvitoTech. |
| `neural-pairwise`, `rrf` | Попарные потери с весом ΔNDCG или смесь порядков двух моделей с весом, выбранным на ранней `val`. |

Для моделей с глобальными статистиками четыре последовательных блока `train` получают признаки из строго более ранних блоков. Для `val` и `test` используется справочник со всего выбранного `train`. Команда `experiments.py diagnose` показывает покрытие известных ID и частоту положительных наблюдений по `pos`. Эти наблюдения не задают вероятность просмотра, поэтому позиционные IPS-веса требуют отдельной оценки. Для ESMM потребуются раздельные метки клика и конверсии.

Публикация [команды рекомендаций AvitoTech](https://habr.com/ru/companies/avito/articles/1090584/) описывает сравнение функций потерь и совместное внимание. Вариант `avito-joint` использует имеющиеся в соревновании тип, источник и порядок событий. `--history-window 50` позволяет отдельно проверить размер окна из статьи. Коды событий обезличены, поэтому окно включает все доступные типы.

Вариант `covis` использует направленные переходы для уже заданных кандидатов. `category-model` и `popularity-trend` проверяют ранг и логит интереса к категории и динамику популярности по отдельности. В разборе первого места весеннего AvitoTech ML CUP 2026 co-visitation показан на 4:22, блоки признаков реранкера на 14:12 и прогноз категории на 20:20. [Видео](https://vkvideo.ru/video-164555658_456242102)

`pixie` проверяет идею из [разбора второго места](https://www.youtube.com/watch?v=gKvlp8o76uY&t=160s). Из графа для каждой ленты исключается сам пользователь. Путь получает вес с поправкой на давность исходного события и число связей узлов. Для скоринга остаются до 256 соседних пользователей. Затухание использует номер события в истории, а граф строится по тем же хронологическим блокам, что `itemcf` и `als`. `category-trend` проверяет идею изменения интереса из [разбора третьего места](https://www.youtube.com/watch?v=EGr9n63j4nw&t=495s).

```powershell
.\.venv\Scripts\python.exe experiments.py benchmark --data .\runs\synthetic-v2\competition --output .\runs\category-pixie --models history covis category-trend pixie category-trend-pixie --train-feeds 0 --val-feeds 0 --iterations 100 --threads 6
```

## Источники

- [Схема данных Авито](https://ods.ai/competitions/avitotechrecsys2026/dataset) и [правила оценки](https://ods.ai/competitions/avitotechrecsys2026) задают формат, порядок разбиений и полную NDCG.
- [DIN](https://arxiv.org/abs/1706.06978), [SIM](https://arxiv.org/abs/2006.05639), [MIND](https://arxiv.org/abs/1904.08030), [BST](https://arxiv.org/abs/1905.06874) и [DIEN](https://arxiv.org/abs/1809.03672) описывают способы представления истории. Варианты в таблице используют соответствующие идеи с готовыми кандидатами. [EasyRec](https://easyrec.readthedocs.io/en/latest/models/din.html) показывает, как связать поля кандидата и истории.
- [SASRec](https://arxiv.org/abs/1808.09781), [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) и [ListMLE](https://icml.cc/Conferences/2008/papers/167.pdf) добавляют сравнение энкодеров последовательности, множества кандидатов и функций потерь.
- Обзоры [ЛАНИТ, часть 1](https://habr.com/ru/companies/lanit/articles/420499/), [ЛАНИТ, часть 2](https://habr.com/ru/companies/lanit/articles/421401/) и [ИТМО](https://neerc.ifmo.ru/wiki/index.php?title=Рекомендательные_системы) связывают холодный старт, item-based сходство и ансамбли с практическими признаками. Реализация ALS опирается на [работу Hu, Koren и Volinsky](https://yifanhu.net/PUB/cf.pdf). Для сравнения функций потерь и понимания позиционного смещения есть [LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) и [Unbiased Learning-to-Rank](https://ijcai.org/proceedings/2018/738).

## Статус обсуждавшихся идей

- В коде есть затухание и дедупликация истории, относительные ранги внутри ленты, признаки по эмбеддингам, сглаженная популярность, IDF категорий, `itemcf`, `covis`, ALS, изменение интереса к категории, графовый скор, сравнение функций потерь и RRF. Для валидации есть хронологическое разбиение и отчёт по длине истории. Парный бутстрап по пользователям выполнялся при сравнении DIN с `history`.
- DIN использован как табличные признаки и как внимание к истории. SIM, MIND, DIEN, BST, SASRec, Set Transformer и Pixie адаптированы к формату лент. В частности, GRU с вниманием использует основную задачу ранжирования, а Pixie оценивает только кандидатов данной ленты.
- В наборе GPU-сравнения добавлены отдельные варианты для динамики популярности, предиктора категории, матриц переходов по категориям, графовых скоров с двумя видами затухания, кластеров эмбеддингов, совместного `covis` и Pixie, двух окон истории и дистилляции. Есть проверка одного расширенного окна интереса, разрезы NDCG по времени, длине истории и категориям и Парето-фронт с учётом времени и RAM.
- Для BM25 по объявлениям нужны тексты, для ESMM — отдельные метки клика и конверсии, для позиционного IPS — оценка вероятности просмотра. Диагностика `pos` измеряет частоту действий по позиции в `train`.
