# GRU v4 с отбором старых событий

Нужны Avito train и val, исходная GRU v4 из valishinqa-v4-repro, прогнозы и train-feeds.parquet из v4-next-200k/adapt, прогнозы epochs и period из local-hypotheses-200k, исходные сигналы V4 и robust. Рабочая копия valishinqa-v4-worktree использует b47fdcb и исходные глобальные статистики.

Сравнение отдельных моделей проведено на шестой эпохе. Доля модели в первом периоде выбрана по наименьшему приросту на двух ранних отрезках. Поздняя val использована для сравнения контроля с лучшей положительной долей.

```powershell
.\.venv\Scripts\python.exe v4_tail.py train --data C:/study/AvitoData/competition --runs runs/avito-real --output runs/avito-real/gru-v4-tail-200k --gpu 0
```

```json
{
  "complete": true,
  "val_feeds": 14423,
  "tuning_cutoff_timestamp": 1790107648,
  "models": [
    {
      "model": "fixed-control",
      "val": {
        "ndcg": 0.3741509801015943,
        "ndcg_at_10": 0.2481355717828161,
        "ndcg_at_30": 0.3160213471350321
      },
      "tuning": {
        "ndcg": 0.371782220256612,
        "ndcg_at_10": 0.24634513307960978,
        "ndcg_at_30": 0.3136253850822606
      },
      "holdout": {
        "ndcg": 0.37657187723743313,
        "ndcg_at_10": 0.2499654187120863,
        "ndcg_at_30": 0.3184700452094506
      },
      "holdout_by_history": {
        "empty": 0.4057365975162252,
        "short": 0.38124329878112956,
        "long": 0.3758437057409881
      }
    },
    {
      "model": "tail-blend-challenger",
      "val": {
        "ndcg": 0.37412292504554606,
        "ndcg_at_10": 0.24819587998229017,
        "ndcg_at_30": 0.31587889551413656
      },
      "tuning": {
        "ndcg": 0.3718139341610899,
        "ndcg_at_10": 0.2464785946729149,
        "ndcg_at_30": 0.3134982070003428
      },
      "holdout": {
        "ndcg": 0.3764827376836655,
        "ndcg_at_10": 0.24995096338414866,
        "ndcg_at_30": 0.3183119838732666
      },
      "holdout_by_history": {
        "empty": 0.4060156782271562,
        "short": 0.38036872404650945,
        "long": 0.3758011382583667
      }
    },
    {
      "model": "plain-gru-v4-epoch6",
      "val": {
        "ndcg": 0.3680395345267577,
        "ndcg_at_10": 0.23954408344944492,
        "ndcg_at_30": 0.30914147591465624
      },
      "tuning": {
        "ndcg": 0.3653759166558914,
        "ndcg_at_10": 0.2371157082869463,
        "ndcg_at_30": 0.30578278723422453
      },
      "holdout": {
        "ndcg": 0.37076177962399814,
        "ndcg_at_10": 0.24202590805826688,
        "ndcg_at_30": 0.3125740905901771
      },
      "holdout_by_history": {
        "empty": 0.3800323799389395,
        "short": 0.37473559691999714,
        "long": 0.3703659463354542
      }
    },
    {
      "model": "gru-v4-tail-top8",
      "val": {
        "ndcg": 0.36776602495417254,
        "ndcg_at_10": 0.23893341579772934,
        "ndcg_at_30": 0.30902172785988397
      },
      "tuning": {
        "ndcg": 0.3656445370653264,
        "ndcg_at_10": 0.23700224067964412,
        "ndcg_at_30": 0.30652119547384155
      },
      "holdout": {
        "ndcg": 0.3699342075855541,
        "ndcg_at_10": 0.24090709680304884,
        "ndcg_at_30": 0.3115772978997525
      },
      "holdout_by_history": {
        "empty": 0.3849001629286117,
        "short": 0.3738251972921457,
        "long": 0.3694618905195657
      }
    }
  ],
  "train_feeds": 200000,
  "epochs": 6,
  "selected_tail_share": 0.0,
  "challenger_tail_share": 0.5,
  "early_trials": {
    "0.0": [
      0.3753802947732782,
      0.36820187024496204
    ],
    "0.25": [
      0.37544793006130855,
      0.3679229521285451
    ],
    "0.5": [
      0.3755036281958845,
      0.36814241595898634
    ],
    "1.0": [
      0.3757508579909328,
      0.3677198657601513
    ]
  },
  "early_split": 1790073856,
  "holdout_feeds": 7133,
  "holdout_paired_bootstrap": {
    "blend": {
      "delta": -8.913947324941322e-05,
      "ci95": [
        -0.00035492593311947223,
        0.00018369437220808454
      ],
      "user_clusters": 7126,
      "resamples": 2000
    },
    "model": {
      "delta": -0.0008275721890134529,
      "ci95": [
        -0.002265840646127682,
        0.0005829360847660919
      ],
      "user_clusters": 7126,
      "resamples": 2000
    }
  },
  "holdout_by_available_tail": {
    "True": {
      "feeds": 6099,
      "ndcg": {
        "plain": 0.3698821854997171,
        "tail": 0.36884686560024715,
        "control_blend": 0.37572091836407195,
        "tail_blend": 0.3756686487578846
      }
    },
    "False": {
      "feeds": 1034,
      "ndcg": {
        "plain": 0.37595002388319526,
        "tail": 0.37634784275809485,
        "control_blend": 0.3815912178260633,
        "tail_blend": 0.38128460263371605
      }
    }
  },
  "top_k": 8,
  "history_positions": [
    65,
    256
  ],
  "stats_feeds": 2929897,
  "selection_timestamps": [
    1789773056,
    1789851648
  ]
}
```
