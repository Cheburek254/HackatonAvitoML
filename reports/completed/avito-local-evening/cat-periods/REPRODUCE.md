After local-evening-200k and local-hypotheses-200k, run

`python local_hypotheses.py cat-periods --data competition --runs runs/avito-real --output runs/avito-real/local-cat-periods --gpu 0`

Selected early-fold recipes (weight is the CatBoost share)

```json
{
  "pair": {
    "period": "two",
    "cat": "qsm",
    "weight": 0.5,
    "early_ndcg": [
      0.374702427531215,
      0.36837427753859453
    ],
    "min_gain": 0.0002348954089021471
  },
  "full": {
    "period": "two",
    "cat": "qsm",
    "weight": 0.5,
    "early_ndcg": [
      0.37585083498739463,
      0.368209504723342
    ],
    "min_gain": -0.0008611285464530383
  }
}
```
