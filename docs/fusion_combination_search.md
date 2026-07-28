# Fusion combination search

This stage searches the trained model inventory. It does not train new models and it does not open the final IID/OOD splits.

## Search space

For every complete cohort, the search enumerates all 31 non-empty subsets of the five operator specialists. Each subset is evaluated with:

- raw base-relative addition
- vocabulary-centered, RMS-equalized addition
- alpha values `0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0`

`bias_mean` is not searched independently. For a subset of cardinality `k`, `bias_mean(alpha=a)` is exactly `raw_sum(alpha=a/k)`.

The primary confirmation cohort is `gpt_bias_fusion_factory_surface_v3`, because its specialists share a trained parent Base and are available for three seeds. The four-size bias-factory models are included as a scaling diagnostic. Those specialists were trained from scratch, so their output-logit deltas are not interpreted as shared-parent parameter deltas.

## Metrics

The broad sweep uses teacher-forced validation traces and records:

- active-operator token accuracy
- active-operator token NLL
- inactive-operator agreement with Base
- inactive centered-delta RMS
- matched-Joint JSD for the all-five fusion-factory subset
- cross-seed mean and standard deviation

Winners are selected within each fixed subset. Accuracy is optimized first, then NLL, inactive drift, and matched-Joint JSD. The cross-seed selector penalizes unstable accuracy.

## Run

Install the current checkout once after pulling:

```bash
.venv/bin/pip install -e .
```

Detached execution:

```bash
bash scripts/run_fusion_combination_search.sh detach
```

Status:

```bash
bash scripts/run_fusion_combination_search.sh status
```

Direct execution with a custom grid:

```bash
.venv/bin/opfusion-search-fusion-combinations \
  --source all \
  --alpha-grid 0.125,0.25,0.5,0.75,1,1.25,1.5,2 \
  --examples-per-operator 64 \
  --evaluation-seed 703000 \
  --device cuda
```

Primary output:

```text
evaluations/fusion_combination_search/summary.json
```

The report contains per-candidate rows, winners per cohort and subset, three-seed aggregate rows, three-seed winners, and the recommended all-five validation candidate.

## Claim boundary

This is calibration on the validation partition. It may identify candidates for a later preregistered evaluation, but it cannot authorize production and cannot support final IID/OOD claims. `production_go` is therefore always `false`.

A candidate containing `scalar.neg` must be interpreted against the known NEG specialist failure. A large bias norm or apparent specialization is not evidence that the specialist computes negation correctly.
