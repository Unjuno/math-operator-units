# Preregistered Experiment Plan v2

This plan extends v1 before the target-GPU pilot and before final IID/OOD inspection. Raw all-five sum at `alpha=1.0` remains confirmatory; every fallback is secondary.

## Fixed sequence

After the four-condition pilot selects model construction, train production seeds 0, 1, and 2. Freeze every `selected.pt`. Before final data, run fusion calibration on `validation` with evaluation seed `703000` and 128 examples per operator. Final evaluation remains seed `700000`, 64 examples per operator, `subset_31`, and splits `iid_test`, `operand_ood`, and `length_ood`.

Calibration activates when raw-minus-Relevant-Specialist mean trace gap is below `-0.02`, worst-operator trace gap is below `-0.05`, or mean EOS gap is below `-0.02`. Final results cannot trigger tuning.

## Leakage control

Use three leave-one-training-seed-out folds. Fit on two seeds and score on the third. The held-out seed contributes no fitted weight, RMS, median, clipping threshold, or other statistic. Problems are paired across seeds. Never tune per seed or operator.

## Mixing ladder

F0 is `raw_sum` and fixed `bias_mean`.

F1 uses `z=z_base+alpha*sum(B_k)` with `alpha` in `{0.10,0.20,0.25,0.50,0.75,1.00}`, shared everywhere.

F2 centers each field over vocabulary. RMS equalization computes one unit RMS over fitting seeds, operators, examples, response positions, and vocabulary, rescales to the median unit RMS, then averages fields. RMS clipping pools per-position centered-field RMS over fitting data and clips at quantile `0.90`, `0.95`, or `0.99`. Use epsilon `1e-8`; no operator-specific scale.

F3 uses `z=z_base+alpha*sum(pi_k*B_k)`, `pi=softmax(a)`. Fit constant weights from uniform initialization by deterministic L-BFGS for at most 500 iterations, gradient tolerance `1e-9`, minimizing mean gold-token NLL plus `lambda*sum((pi_k-1/K)^2)`, with `lambda` in `{0,0.01,0.10,1.0}`.

F4 computes `d_k=JSD(p_k,p_bar)`, `u=softmax(-beta*d)`, and `w=K*u`, then uses `z=z_base+alpha*sum(w_k*B_k)`, with `beta` in `{0.5,1.0,2.0}`. Since weights sum to `K`, `beta -> 0` recovers raw sum before alpha. This is consensus-tempered decoding, not raw composition.

Use the first eligible family in order F1, F2, F3, F4. Eligibility requires mean trace-gap improvement over raw of at least `0.02`, EOS and final-value gaps to Relevant Specialist at least `-0.02`, worst-operator trace gap at least `-0.05`, and improvement on at least two held-out seeds. Refit only the selected family on all validation data. If none qualifies, stop the non-router search. Learned routers or correctors require a new plan and calibration partition.

## Final lock

Before generating any final split, calibration must create `evaluations/fusion_calibration/final_authorization.json`. The evaluator verifies authorization ABI 1, active-plan hash, current Git commit, experiment fingerprint, all three production seeds and folds, calibration settings, and final settings. Allowed states are `raw_preserved_no_rescue`, `rescue_selected`, and `no_eligible_nonrouter_rescue`; a selected rescue also records its family and mixer-contract SHA-256. Missing or mismatched authorization fails closed.

Final reporting always preserves Relevant Specialist, raw sum, bias mean, and matched Joint as the primary table. A frozen rescue is labeled secondary. Any method selected after final data are viewed is post hoc.
