# mixed-periods

Рабочая копия V4 b47fdcb и артефакты предыдущих прогонов, включая XL №002 и №003, используются для контроля 0,376893.

```powershell
.\.venv\Scripts\python.exe gru_remaining.py mixed-periods --data C:/study/AvitoData/competition --runs runs/avito-real --output runs/avito-real/gru-remaining-200k --gpu 0
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
        "ndcg": 0.37447476733717666,
        "ndcg_at_10": 0.24850359299128566,
        "ndcg_at_30": 0.3163912957902673
      },
      "tuning": {
        "ndcg": 0.3722196477783532,
        "ndcg_at_10": 0.2469949115628484,
        "ndcg_at_30": 0.31415794389737717
      },
      "holdout": {
        "ndcg": 0.37677952292161926,
        "ndcg_at_10": 0.25004548106270247,
        "ndcg_at_30": 0.31867380459431877
      },
      "holdout_by_history": {
        "empty": 0.40696620744143364,
        "short": 0.37988728666122584,
        "long": 0.37613991067989805
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
      "model": "mixed-periods-epoch6",
      "val": {
        "ndcg": 0.3670165922441006,
        "ndcg_at_10": 0.23816633390247816,
        "ndcg_at_30": 0.3079379853975563
      },
      "tuning": {
        "ndcg": 0.3640510657950497,
        "ndcg_at_10": 0.2353087750119209,
        "ndcg_at_30": 0.30449240362226004
      },
      "holdout": {
        "ndcg": 0.3700473910403364,
        "ndcg_at_10": 0.24108678873386308,
        "ndcg_at_30": 0.31145940571748093
      },
      "holdout_by_history": {
        "empty": 0.3769098017280322,
        "short": 0.3740499234251539,
        "long": 0.36968431422496245
      }
    }
  ],
  "train_feeds": 200000,
  "epochs": 6,
  "epoch_seconds": [
    1140.68,
    633.35,
    638.7,
    633.66,
    739.85,
    1032.4
  ],
  "periods": [
    {
      "feeds": 100000,
      "history_cutoff": 1789773056,
      "past_stats_feeds": 2929897,
      "seed": 42,
      "first_timestamp": 1789773056,
      "last_timestamp": 1789851648
    },
    {
      "feeds": 60000,
      "history_cutoff": 1789707648,
      "past_stats_feeds": 2708510,
      "seed": 43,
      "first_timestamp": 1789707648,
      "last_timestamp": 1789772928
    },
    {
      "feeds": 40000,
      "history_cutoff": 1789631744,
      "past_stats_feeds": 2487331,
      "seed": 44,
      "first_timestamp": 1789631744,
      "last_timestamp": 1789707520
    }
  ],
  "early_trials": {
    "0.0": [
      0.37585914414027927,
      0.36858436113664506
    ],
    "0.25": [
      0.37585849507190494,
      0.3685987258409331
    ],
    "0.5": [
      0.37575253825756466,
      0.3685613437755005
    ],
    "1.0": [
      0.37573068225913675,
      0.36819361267782413
    ]
  },
  "selected_new_gru_share": 0.0,
  "challenger_new_gru_share": 0.25,
  "holdout_paired_bootstrap": {
    "delta": -0.00011358405941919476,
    "ci95": [
      -0.00030347660371424673,
      7.44929722079108e-05
    ],
    "user_clusters": 7126,
    "resamples": 2000
  },
  "holdout_feeds": 7133,
  "peak_cuda_mib": 144.625
}
```
