from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SRC = Path(__file__).with_name("experiment_neural_private_tokenizer_adapter.py")
spec = importlib.util.spec_from_file_location("private_tokenizer_adapter", SRC)
v = importlib.util.module_from_spec(spec)
sys.modules["private_tokenizer_adapter"] = v
assert spec.loader is not None
spec.loader.exec_module(v)


def invert_adapter(ad):
    return {d: s for s, d in ad["sym_to_digit"].items()}


def encode_by_adapter(ad, y):
    ds = v.digits_base(y, ad["base"])
    if ad["reverse"]:
        ds = list(reversed(ds))
    inv = invert_adapter(ad)
    try:
        return [inv[d] for d in ds]
    except KeyError:
        return None


@torch.no_grad()
def action_logprobs(model, ad, a, b, device):
    seqs = []
    valid = []
    maxlen = 0
    for y in range(v.N):
        seq = encode_by_adapter(ad, y)
        if seq is None:
            seqs.append(None)
            continue
        target = seq + [model.eos]
        seqs.append(target)
        maxlen = max(maxlen, len(target))
        valid.append(y)

    if not valid:
        return torch.full((v.N,), float("-inf"))

    pair = torch.tensor([model.codec.pair_id(a, b)] * len(valid), device=device)
    target = torch.full((len(valid), maxlen), -100, dtype=torch.long, device=device)
    for i, y in enumerate(valid):
        target[i, : len(seqs[y])] = torch.tensor(seqs[y], device=device)

    logits = model.teacher_logits(pair, target)
    logp = F.log_softmax(logits, -1)
    scores = []
    for i, y in enumerate(valid):
        toks = target[i]
        mask = toks >= 0
        pos = torch.arange(maxlen, device=device)[mask]
        scores.append(logp[i, pos, toks[mask]].sum())

    out = torch.full((v.N,), float("-inf"), device=device)
    out[torch.tensor(valid, device=device)] = torch.stack(scores)
    out = out - torch.logsumexp(out, 0)
    return out.cpu()


def fuse(lp_primary, lp_secondary, alpha, method):
    if method == "logpool":
        return alpha * lp_primary + (1 - alpha) * lp_secondary
    if method == "probmix":
        return torch.logaddexp(
            lp_primary + torch.log(torch.tensor(alpha)),
            lp_secondary + torch.log(torch.tensor(1 - alpha)),
        )
    raise KeyError(method)


def eval_programs(models, adapters, device, alpha, method, nprog=60, seed=0, cache=None):
    rng = random.Random(seed)
    result = {}
    if cache is None:
        cache = {}

    def get_logp(source_i, a, b):
        key = (source_i, a, b)
        if key not in cache:
            cache[key] = action_logprobs(models[source_i], adapters[source_i], a, b, device)
        return cache[key]

    for depth in range(1, 6):
        e2e = 0
        local = 0
        stages = 0
        for _ in range(nprog):
            pred = truth = rng.randrange(v.N)
            program = [(rng.choice(v.OPS), rng.randrange(v.N)) for _ in range(depth)]
            for op, b in program:
                primary = v.OPS.index(op)
                secondary = (primary + 1) % len(v.OPS)
                lp1 = get_logp(primary, pred, b)
                lp2 = get_logp(secondary, pred, b)
                score = lp1 if alpha >= 1 else fuse(lp1, lp2, alpha, method)
                y = int(score.argmax())
                expected = v.op_apply(op, pred, b)
                local += int(y == expected)
                stages += 1
                pred = y
                truth = v.op_apply(op, truth, b)
            e2e += int(pred == truth)
        result[str(depth)] = {"e2e": e2e / nprog, "stage_local": local / stages}
    return result


def run(out, pool_seed=120, steps=260, nprog=60):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    models = []
    rows = []
    train_exact = []
    started = time.time()

    for i, op in enumerate(v.OPS):
        model, source_rows, _ = v.train_one(op, pool_seed * 100 + i * 19 + 3, device, steps)
        models.append(model)
        rows.append(source_rows)
        train_exact.append(v.exact_acc(model, source_rows, device))

    adapters = []
    for model, source_rows in zip(models, rows):
        idx = v.active_anchor_indices(source_rows)
        sequences = v.gen_rows(model, source_rows, idx, device)
        adapters.append(v.infer_adapter([(source_rows[k][3], seq) for k, seq in zip(idx, sequences)]))

    runs = {}
    for method in ("logpool", "probmix"):
        runs[method] = {}
        cache = {}
        for alpha in (1.0, 0.95, 0.8, 0.6, 0.5):
            runs[method][str(alpha)] = eval_programs(
                models,
                adapters,
                device,
                alpha,
                method,
                nprog=nprog,
                seed=77123,
                cache=cache,
            )

    result = {
        "experiment": "heterogeneous_action_space_soft_fusion",
        "pool_seed": pool_seed,
        "source_train_exact": train_exact,
        "private_output_head_sizes": [m.out_vocab for m in models],
        "raw_token_logit_fusion_defined_for_all_sources": len({m.out_vocab for m in models}) == 1,
        "adapter_anchor_count": 20,
        "fusion": "map each private sequence likelihood into a normalized common-action distribution, then fuse there",
        "runs": runs,
        "elapsed_s": time.time() - started,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool-seed", type=int, default=120)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--nprog", type=int, default=60)
    args = parser.parse_args()
    run(args.out, args.pool_seed, args.steps, args.nprog)


if __name__ == "__main__":
    main()
