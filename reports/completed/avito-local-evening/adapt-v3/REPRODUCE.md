Use the Avito validation and pretrained artifacts from v4-next-200k/adapt, valishinqa-v4-repro, valishinqa-v4-worktree, v3-robust-full and local-hypotheses-200k/period. Run stages third-period, adapt-v3, period-blend in this order.

`python local_hypotheses.py adapt-v3 --data competition --runs runs/avito-real --output runs/avito-real/local-evening-200k --gpu 0`
