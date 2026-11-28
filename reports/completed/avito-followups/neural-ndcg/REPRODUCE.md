# neural-ndcg

Рабочая копия V4 b47fdcb и артефакты предыдущих прогонов, включая XL №002 и №003, используются для контроля 0,376893.

```powershell
.\.venv\Scripts\python.exe gru_remaining.py neural-ndcg --data C:/study/AvitoData/competition --runs runs/avito-real --output runs/avito-real/gru-remaining-200k --gpu 0
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
        "ndcg": 0.3745274655950673,
        "ndcg_at_10": 0.24853325711637342,
        "ndcg_at_30": 0.31649659955711174
      },
      "tuning": {
        "ndcg": 0.3722127714248747,
        "ndcg_at_10": 0.24692179701915531,
        "ndcg_at_30": 0.3141810217250337
      },
      "holdout": {
        "ndcg": 0.3768931070503751,
        "ndcg_at_10": 0.25018018605354253,
        "ndcg_at_30": 0.3188631441240504
      },
      "holdout_by_history": {
        "empty": 0.4071253923341793,
        "short": 0.3798663975770847,
        "long": 0.3762617193236865
      }
    },
    {
      "model": "new-gru-in-blend",
      "val": {
        "ndcg": 0.37443322323815936,
        "ndcg_at_10": 0.24843929739348922,
        "ndcg_at_30": 0.3163505548484298
      },
      "tuning": {
        "ndcg": 0.3721522204092242,
        "ndcg_at_10": 0.2467537299548533,
        "ndcg_at_30": 0.3140598879593463
      },
      "holdout": {
        "ndcg": 0.3767644317931744,
        "ndcg_at_10": 0.2501619648024007,
        "ndcg_at_30": 0.3186916401731949
      },
      "holdout_by_history": {
        "empty": 0.40688203078598983,
        "short": 0.38052659793294036,
        "long": 0.376082596540032
      }
    },
    {
      "model": "fresh-pairwise-epoch6",
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
      "model": "neural-ndcg-epoch6",
      "val": {
        "ndcg": 0.36749785498894744,
        "ndcg_at_10": 0.23937749758879465,
        "ndcg_at_30": 0.3085267950455917
      },
      "tuning": {
        "ndcg": 0.36586464708351185,
        "ndcg_at_10": 0.23790579721558602,
        "ndcg_at_30": 0.3066417902337729
      },
      "holdout": {
        "ndcg": 0.3691670104117172,
        "ndcg_at_10": 0.2408815906381154,
        "ndcg_at_30": 0.3104532895189236
      },
      "holdout_by_history": {
        "empty": 0.37844701081927945,
        "short": 0.3677644470011078,
        "long": 0.36912610123394013
      }
    }
  ],
  "train_feeds": 200000,
  "epochs": 6,
  "epoch_seconds": [
    1307.58,
    727.64,
    717.31,
    796.07,
    1449.78,
    1493.62
  ],
  "periods": [
    {
      "feeds": 200000,
      "past_stats_feeds": 2929897
    }
  ],
  "early_trials": {
    "0.0": [
      0.37585914414027927,
      0.36858436113664506
    ],
    "0.25": [
      0.3757860504740871,
      0.36853629098508345
    ],
    "0.5": [
      0.37594093079111657,
      0.36850906580853965
    ],
    "1.0": [
      0.3750744927237534,
      0.3686419418889755
    ]
  },
  "selected_new_gru_share": 0.0,
  "challenger_new_gru_share": 0.25,
  "holdout_paired_bootstrap": {
    "delta": -0.0001286750835087529,
    "ci95": [
      -0.0004857352462904103,
      0.00023595094483266785
    ],
    "user_clusters": 7126,
    "resamples": 2000
  },
  "holdout_feeds": 7133,
  "peak_cuda_mib": 265.72412109375,
  "temperature": 1.0,
  "sinkhorn_steps": 10
}
```
