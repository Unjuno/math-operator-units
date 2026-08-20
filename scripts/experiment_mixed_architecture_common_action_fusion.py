from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).with_name("experiment_neural_private_tokenizer_adapter.py")
spec = importlib.util.spec_from_file_location("private_tokenizer_adapter_mixed", BASE)
v = importlib.util.module_from_spec(spec)
sys.modules["private_tokenizer_adapter_mixed"] = v
assert spec.loader is not None
spec.loader.exec_module(v)

ARCHS = ("gru", "nar_mlp", "gru", "nar_mlp", "gru")


class NARSpecialist(nn.Module):
    """Non-autoregressive sequence specialist with position-specific output heads."""

    def __init__(self, codec, hidden=160, max_steps=4):
        super().__init__()
        self.codec = codec
        self.base = codec.base
        self.start = self.base
        self.eos = self.base + 1
        self.out_vocab = self.base + 2
        self.max_steps = max_steps
        self.pair_emb = nn.Embedding(v.N * v.N, hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.ModuleList([nn.Linear(hidden, self.out_vocab) for _ in range(max_steps)])

    def teacher_logits(self, pair, target):
        h = self.net(self.pair_emb(pair))
        return torch.stack([self.head[t](h) for t in range(target.size(1))], dim=1)

    @torch.no_grad()
    def generate(self, pair, max_len=5):
        h = self.net(self.pair_emb(pair))
        steps = min(self.max_steps, max_len)
        logits = torch.stack([self.head[t](h) for t in range(steps)], dim=1)
        rows = logits.argmax(-1).tolist()
        out = []
        for row in rows:
            seq = []
            for token in row:
                if token == self.eos:
                    break
                seq.append(token)
            out.append(seq)
        return out


def train_one(op, seed, arch, device, steps=260, batch=512):
    rng = random.Random(seed)
    codec = v.make_codec(rng)
    torch.manual_seed(seed)
    model = v.Specialist(codec) if arch == "gru" else NARSpecialist(codec)
    model = model.to(device)
    rows, pair_ids, target = v.make_rows(op, codec, device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 77)
    for _ in range(steps):
        idx = torch.randint(0, len(rows), (batch,), generator=gen, device=device)
        logits = model.teacher_logits(pair_ids[idx], target[idx])
        loss = F.cross_entropy(
            logits.reshape(-1, model.out_vocab),
            target[idx].reshape(-1),
            ignore_index=-100,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, rows


@torch.no_grad()
def action_logprobs(model, adapter, a, b, device):
    inv = {digit: symbol for symbol, digit in adapter["sym_to_digit"].items()}
    seqs = []
    valid = []
    maxlen = 0
    for y in range(v.N):
        digits = v.digits_base(y, adapter["base"])
        if adapter["reverse"]:
            digits = list(reversed(digits))
        try:
            seq = [inv[d] for d in digits]
        except KeyError:
            seqs.append(None)
            continue
        target = seq + [model.eos]
        seqs.append(target)
        valid.append(y)
        maxlen = max(maxlen, len(target))

    pair = torch.tensor([model.codec.pair_id(a, b)] * len(valid), device=device)
    target = torch.full((len(valid), maxlen), -100, dtype=torch.long, device=device)
    for i, y in enumerate(valid):
        target[i, : len(seqs[y])] = torch.tensor(seqs[y], device=device)

    logp = F.log_softmax(model.teacher_logits(pair, target), -1)
    scores = []
    for i in range(len(valid)):
        mask = target[i] >= 0
        pos = torch.arange(maxlen, device=device)[mask]
        scores.append(logp[i, pos, target[i][mask]].sum())

    out = torch.full((v.N,), float("-inf"), device=device)
    out[torch.tensor(valid, device=device)] = torch.stack(scores)
    out -= torch.logsumexp(out, 0)
    return out.cpu()


def fuse(lp_primary, lp_secondary, alpha, method):
    if method == "probmix":
        return torch.logaddexp(
            lp_primary + torch.log(torch.tensor(alpha)),
            lp_secondary + torch.log(torch.tensor(1 - alpha)),
        )
    if method == "logpool":
        return alpha * lp_primary + (1 - alpha) * lp_secondary
    raise KeyError(method)


def eval_soft(models, adapters, device, alpha, method, nprog=40, seed=0):
    rng = random.Random(seed)
    cache = {}
    result = {}

    def get_logp(source_i, a, b):
        key = (source_i, a, b)
        if key not in cache:
            cache[key] = action_logprobs(models[source_i], adapters[source_i], a, b, device)
        return cache[key]

    for depth in range(1, 6):
        e2e = local = stages = 0
        for _ in range(nprog):
            pred = truth = rng.randrange(v.N)
            program = [(rng.choice(v.OPS), rng.randrange(v.N)) for _ in range(depth)]
            for op, b in program:
                primary = v.OPS.index(op)
                secondary = (primary + 1) % len(v.OPS)
                score = fuse(
                    get_logp(primary, pred, b),
                    get_logp(secondary, pred, b),
                    alpha,
                    method,
                )
                y = int(score.argmax())
                expected = v.op_apply(op, pred, b)
                local += int(y == expected)
                stages += 1
                pred = y
                truth = v.op_apply(op, truth, b)
            e2e += int(pred == truth)
        result[str(depth)] = {"e2e": e2e / nprog, "stage_local": local / stages}
    return result


def eval_hard(models, adapters, device, nprog=200, seed=0):
    rng = random.Random(seed)
    result = {}
    for depth in range(1, 6):
        e2e = local = stages = 0
        for _ in range(nprog):
            pred = truth = rng.randrange(v.N)
            ok = True
            program = [(rng.choice(v.OPS), rng.randrange(v.N)) for _ in range(depth)]
            for op, b in program:
                truth = v.op_apply(op, truth, b)
                if not (0 <= pred < v.N):
                    ok = False
                    continue
                source_i = v.OPS.index(op)
                pair = torch.tensor([models[source_i].codec.pair_id(pred, b)], device=device)
                seq = models[source_i].generate(pair)[0]
                y = v.decode(adapters[source_i], seq)
                expected = v.op_apply(op, pred, b)
                local += int(y == expected)
                stages += 1
                if y is None or not (0 <= y < v.N):
                    pred = -1
                    ok = False
                else:
                    pred = y
            e2e += int(ok and pred == truth)
        result[str(depth)] = {"e2e": e2e / nprog, "stage_local": local / max(stages, 1)}
    return result


def run_pool(pool_seed, device):
    models = []
    rows_all = []
    train = []
    for i, (op, arch) in enumerate(zip(v.OPS, ARCHS)):
        model, rows = train_one(op, pool_seed * 100 + i * 19 + 3, arch, device)
        models.append(model)
        rows_all.append(rows)
        train.append({
            "op": op,
            "arch": arch,
            "exact": v.exact_acc(model, rows, device),
            "head": model.out_vocab,
        })

    adapters = []
    structural = []
    full_decode = []
    for model, rows in zip(models, rows_all):
        idx = v.active_anchor_indices(rows)
        seqs = v.gen_rows(model, rows, idx, device)
        adapter = v.infer_adapter([(rows[k][3], seq) for k, seq in zip(idx, seqs)])
        adapters.append(adapter)
        codec = model.codec
        structural.append(
            adapter["base"] == codec.base
            and adapter["reverse"] == codec.reverse
            and all(adapter["sym_to_digit"].get(codec.digit_to_symbol[d]) == d for d in range(codec.base))
        )
        all_idx = list(range(len(rows)))
        all_seqs = v.gen_rows(model, rows, all_idx, device)
        full_decode.append(
            sum(v.decode(adapter, seq) == rows[k][3] for k, seq in zip(all_idx, all_seqs)) / len(all_idx)
        )

    return {
        "pool_seed": pool_seed,
        "architectures": list(ARCHS),
        "train": train,
        "adapter_structural": structural,
        "full_decode": full_decode,
        "hard": eval_hard(models, adapters, device, nprog=200, seed=pool_seed + 7000),
        "soft": {
            "probmix_0.6": eval_soft(models, adapters, device, 0.6, "probmix", 40, pool_seed + 8000),
            "probmix_0.5": eval_soft(models, adapters, device, 0.5, "probmix", 40, pool_seed + 8000),
            "logpool_0.8": eval_soft(models, adapters, device, 0.8, "logpool", 40, pool_seed + 8000),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool-seeds", type=int, nargs="+", default=[210, 211, 212])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    started = time.time()
    pools = []
    for seed in args.pool_seeds:
        print("POOL", seed, flush=True)
        pool = run_pool(seed, device)
        pools.append(pool)
        print(json.dumps(pool, indent=2), flush=True)
    result = {
        "experiment": "mixed_architecture_common_action_fusion",
        "scope": "mixed GRU autoregressive and non-autoregressive MLP sources with private input tokenizers and private variable-length output codecs",
        "pools": pools,
        "elapsed_s": time.time() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
