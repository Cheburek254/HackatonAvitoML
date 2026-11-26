Use the Avito validation and pretrained artifacts from v4-next-200k/adapt, valishinqa-v4-repro, valishinqa-v4-worktree, v3-robust-full and local-hypotheses-200k/period. Run stages third-period, adapt-v3, period-blend in this order.

`python local_hypotheses.py period-blend --data competition --runs runs/avito-real --output runs/avito-real/local-evening-200k --gpu 0`

Fixed selection and fold results

```json
{
  "complete": true,
  "val_feeds": 14423,
  "tuning_cutoff_timestamp": 1790107648,
  "models": [
    {
      "model": "two-periods-control",
      "val": {
        "ndcg": 0.37405020824710095,
        "ndcg_at_10": 0.2480615528800473,
        "ndcg_at_30": 0.31595496318056215
      },
      "tuning": {
        "ndcg": 0.37172076510860474,
        "ndcg_at_10": 0.24638275929003697,
        "ndcg_at_30": 0.31356618418376997
      },
      "holdout": {
        "ndcg": 0.37643092330102623,
        "ndcg_at_10": 0.24977729734538986,
        "ndcg_at_30": 0.31839632009725044
      },
      "holdout_by_history": {
        "empty": 0.4055651696457071,
        "short": 0.3805497174188348,
        "long": 0.37573968615499453
      }
    },
    {
      "model": "one-period-control",
      "val": {
        "ndcg": 0.37428432934838807,
        "ndcg_at_10": 0.2484131789827427,
        "ndcg_at_30": 0.3161482137521349
      },
      "tuning": {
        "ndcg": 0.3724129516223289,
        "ndcg_at_10": 0.24745374569000095,
        "ndcg_at_30": 0.3141531365696249
      },
      "holdout": {
        "ndcg": 0.37619689679868723,
        "ndcg_at_10": 0.24939372976140511,
        "ndcg_at_30": 0.318187203330241
      },
      "holdout_by_history": {
        "empty": 0.4057171535979065,
        "short": 0.3791530896589012,
        "long": 0.37557688400211486
      }
    },
    {
      "model": "early-selected",
      "val": {
        "ndcg": 0.37428432934838807,
        "ndcg_at_10": 0.2484131789827427,
        "ndcg_at_30": 0.3161482137521349
      },
      "tuning": {
        "ndcg": 0.3724129516223289,
        "ndcg_at_10": 0.24745374569000095,
        "ndcg_at_30": 0.3141531365696249
      },
      "holdout": {
        "ndcg": 0.37619689679868723,
        "ndcg_at_10": 0.24939372976140511,
        "ndcg_at_30": 0.318187203330241
      },
      "holdout_by_history": {
        "empty": 0.4057171535979065,
        "short": 0.3791530896589012,
        "long": 0.37557688400211486
      }
    }
  ],
  "selection": {
    "weights": [
      1.0,
      0.0,
      0.0,
      0.0
    ],
    "early_ndcg": [
      0.3757718161053299,
      0.36907063326979506
    ],
    "min_gain": 0.0
  },
  "early_trials": [
    {
      "weights": [
        1.0,
        0.0,
        0.0,
        0.0
      ],
      "early_ndcg": [
        0.3757718161053299,
        0.36907063326979506
      ],
      "min_gain": 0.0
    },
    {
      "weights": [
        1.0,
        0.0,
        0.0,
        0.5
      ],
      "early_ndcg": [
        0.37563196783288355,
        0.36901452768359017
      ],
      "min_gain": -0.00013984827244634124
    },
    {
      "weights": [
        1.0,
        0.0,
        0.0,
        1.0
      ],
      "early_ndcg": [
        0.37608753828117963,
        0.36894593128664543
      ],
      "min_gain": -0.0001247019831496221
    },
    {
      "weights": [
        0.5,
        0.5,
        0.0,
        0.0
      ],
      "early_ndcg": [
        0.3753831517248155,
        0.36807641980578365
      ],
      "min_gain": -0.0009942134640114064
    },
    {
      "weights": [
        0.5,
        0.5,
        0.0,
        0.5
      ],
      "early_ndcg": [
        0.3757080663783171,
        0.36784460320748313
      ],
      "min_gain": -0.0012260300623119225
    },
    {
      "weights": [
        0.5,
        0.5,
        0.0,
        1.0
      ],
      "early_ndcg": [
        0.37601327440112736,
        0.36787181403411573
      ],
      "min_gain": -0.001198819235679327
    },
    {
      "weights": [
        0.5,
        0.3,
        0.2,
        0.0
      ],
      "early_ndcg": [
        0.37521919174141805,
        0.3682640055251749
      ],
      "min_gain": -0.0008066277446201475
    },
    {
      "weights": [
        0.5,
        0.3,
        0.2,
        0.5
      ],
      "early_ndcg": [
        0.3758505026939555,
        0.3683394059480306
      ],
      "min_gain": -0.0007312273217644494
    },
    {
      "weights": [
        0.5,
        0.3,
        0.2,
        1.0
      ],
      "early_ndcg": [
        0.37562248383597796,
        0.3681436461495047
      ],
      "min_gain": -0.000926987120290379
    },
    {
      "weights": [
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.0
      ],
      "early_ndcg": [
        0.37528803689863965,
        0.3681126021469187
      ],
      "min_gain": -0.0009580311228763461
    },
    {
      "weights": [
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.5
      ],
      "early_ndcg": [
        0.3755325576599925,
        0.3680970191782293
      ],
      "min_gain": -0.0009736140915657665
    },
    {
      "weights": [
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        1.0
      ],
      "early_ndcg": [
        0.37573373107461117,
        0.3680637616850537
      ],
      "min_gain": -0.0010068715847413312
    }
  ],
  "split1_end": 1790073856,
  "weight_order": [
    "first-v4",
    "second-v4",
    "third-v4",
    "adapted-share-of-v3"
  ],
  "early_fold_feeds": [
    3636,
    3654
  ]
}
```
