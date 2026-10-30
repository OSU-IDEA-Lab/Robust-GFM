# ULTRA zero-shot: expected summary

Output of `scripts/summarize.sh` (`results/summary.md`). One row per (dataset, checkpoint, variant) against the
reference variant, on the queries answerable on both. MRR is tie-aware; `floor` is the τ of the reference against
its repeat run; `held out` = graph not in the checkpoint's documented pretraining set (`unknown` for 50g).

| dataset | ckpt | held out | variant | queries | MRR ref | MRR var | dMRR | tau | floor | Hit@1 differs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WN18RR | 3g | no | all_inverse_edges | 3134 | 0.370 | 0.306 | -0.064 | 0.506 | 1.0 | 0.069 |
| WN18RR | 3g | no | transitive_edges | 3134 | 0.370 | 0.304 | -0.066 | 0.132 | 1.0 | 0.158 |
| WN18RR | 3g | no | universal_edges | 3134 | 0.370 | 0.372 | +0.002 | 0.060 | 1.0 | 0.192 |
| WN18RR | 4g | no | all_inverse_edges | 3134 | 0.485 | 0.435 | -0.050 | 0.581 | 1.0 | 0.097 |
| WN18RR | 4g | no | transitive_edges | 3134 | 0.485 | 0.345 | -0.141 | 0.327 | 1.0 | 0.174 |
| WN18RR | 4g | no | universal_edges | 3134 | 0.485 | 0.390 | -0.096 | 0.334 | 1.0 | 0.153 |
| WN18RR | 50g | unknown | all_inverse_edges | 3134 | 0.429 | 0.337 | -0.092 | 0.388 | 1.0 | 0.149 |
| WN18RR | 50g | unknown | transitive_edges | 3134 | 0.429 | 0.349 | -0.080 | 0.235 | 1.0 | 0.166 |
| WN18RR | 50g | unknown | universal_edges | 3134 | 0.429 | 0.378 | -0.051 | 0.225 | 1.0 | 0.157 |
| FB15k237 | 3g | no | inverse | 20466 | 0.369 | 0.365 | -0.004 | 0.819 | 1.0 | 0.043 |
| FB15k237 | 3g | no | reified_cvt_light | 20412 | 0.369 | 0.367 | -0.002 | 0.866 | 1.0 | 0.015 |
| FB15k237 | 3g | no | reified_cvt_strong | 14995 | 0.403 | 0.364 | -0.039 | 0.537 | 1.0 | 0.061 |
| FB15k237 | 3g | no | shortcut2_c200k | 20466 | 0.369 | 0.296 | -0.073 | 0.538 | 1.0 | 0.115 |
| FB15k237 | 3g | no | shortcut2_c50k | 20466 | 0.369 | 0.322 | -0.047 | 0.590 | 1.0 | 0.088 |
| FB15k237 | 3g | no | shortcut2_c5k | 20466 | 0.369 | 0.356 | -0.013 | 0.800 | 1.0 | 0.039 |
| FB15k237 | 4g | no | inverse | 20466 | 0.368 | 0.359 | -0.009 | 0.808 | 1.0 | 0.051 |
| FB15k237 | 4g | no | reified_cvt_light | 20412 | 0.368 | 0.365 | -0.003 | 0.886 | 1.0 | 0.020 |
| FB15k237 | 4g | no | reified_cvt_strong | 14995 | 0.400 | 0.379 | -0.021 | 0.712 | 1.0 | 0.053 |
| FB15k237 | 4g | no | shortcut2_c200k | 20466 | 0.368 | 0.285 | -0.083 | 0.494 | 1.0 | 0.122 |
| FB15k237 | 4g | no | shortcut2_c50k | 20466 | 0.368 | 0.315 | -0.053 | 0.573 | 1.0 | 0.094 |
| FB15k237 | 4g | no | shortcut2_c5k | 20466 | 0.368 | 0.355 | -0.014 | 0.785 | 1.0 | 0.043 |
| FB15k237 | 50g | unknown | inverse | 20466 | 0.367 | 0.352 | -0.015 | 0.798 | 1.0 | 0.054 |
| FB15k237 | 50g | unknown | reified_cvt_light | 20412 | 0.367 | 0.366 | -0.002 | 0.814 | 1.0 | 0.017 |
| FB15k237 | 50g | unknown | reified_cvt_strong | 14995 | 0.399 | 0.361 | -0.038 | 0.586 | 1.0 | 0.068 |
| FB15k237 | 50g | unknown | shortcut2_c200k | 20466 | 0.367 | 0.300 | -0.068 | 0.542 | 1.0 | 0.112 |
| FB15k237 | 50g | unknown | shortcut2_c50k | 20466 | 0.367 | 0.327 | -0.041 | 0.657 | 1.0 | 0.091 |
| FB15k237 | 50g | unknown | shortcut2_c5k | 20466 | 0.367 | 0.358 | -0.010 | 0.810 | 1.0 | 0.042 |
| Hetionet | 3g | yes | inverse | 5000 | 0.254 | 0.243 | -0.012 | 0.848 | 1.0 | 0.051 |
| Hetionet | 3g | yes | reified_regulation | 4872 | 0.259 | 0.256 | -0.003 | 0.916 | 1.0 | 0.019 |
| Hetionet | 3g | yes | shortcut2_c50k | 5000 | 0.254 | 0.242 | -0.013 | 0.736 | 1.0 | 0.078 |
| Hetionet | 4g | yes | inverse | 5000 | 0.274 | 0.255 | -0.019 | 0.820 | 1.0 | 0.054 |
| Hetionet | 4g | yes | reified_regulation | 4872 | 0.279 | 0.280 | +0.001 | 0.887 | 1.0 | 0.025 |
| Hetionet | 4g | yes | shortcut2_c50k | 5000 | 0.274 | 0.232 | -0.042 | 0.723 | 1.0 | 0.081 |
| Hetionet | 50g | unknown | inverse | 5000 | 0.394 | 0.326 | -0.068 | 0.803 | 1.0 | 0.121 |
| Hetionet | 50g | unknown | reified_regulation | 4872 | 0.400 | 0.393 | -0.007 | 0.886 | 1.0 | 0.036 |
| Hetionet | 50g | unknown | shortcut2_c50k | 5000 | 0.394 | 0.274 | -0.120 | 0.747 | 1.0 | 0.168 |
