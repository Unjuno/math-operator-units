from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Any, Optional
import math
import numpy as np

@dataclass
class RegisteredSource:
    key: str
    relation_prob: Callable[[Any], float]
    execute_common: Callable[[Any], Any]
    bridge_prob: Optional[Callable[[Any], float]] = None

@dataclass
class Identification:
    key: Optional[str]
    posterior: dict[str, float]
    relation_queries: int
    bridge_queries: int
    unresolved: bool

class BehaviorRegistry:
    def __init__(self, sources: Iterable[RegisteredSource]):
        self.sources = list(sources)
        if not self.sources:
            raise ValueError('empty registry')
        keys = [s.key for s in self.sources]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate keys')
        self.by_key = {s.key: s for s in self.sources}

    @staticmethod
    def _normalize(logp: np.ndarray) -> np.ndarray:
        m = float(np.max(logp))
        w = np.exp(logp - m)
        z = float(w.sum())
        return w / z if z > 0 else np.ones_like(w) / len(w)

    @staticmethod
    def _entropy(p: float) -> float:
        p = min(max(float(p), 1e-9), 1 - 1e-9)
        return -p * math.log(p) - (1-p) * math.log(1-p)

    def _score_query(self, probs: np.ndarray, posterior: np.ndarray) -> float:
        mix = float(np.dot(posterior, probs))
        return self._entropy(mix) - float(np.dot(posterior, [self._entropy(x) for x in probs]))

    def _pick_query(self, proposals: list[Any], posterior: np.ndarray, bridge: bool=False):
        best_q, best_score, best_probs = None, -1.0, None
        for q in proposals:
            vals = []
            valid = True
            for s in self.sources:
                fn = s.bridge_prob if bridge else s.relation_prob
                if fn is None:
                    valid = False
                    break
                vals.append(float(np.clip(fn(q), 1e-4, 1-1e-4)))
            if not valid:
                continue
            probs = np.asarray(vals, dtype=np.float64)
            score = self._score_query(probs, posterior)
            if score > best_score:
                best_q, best_score, best_probs = q, score, probs
        return best_q, best_score, best_probs

    def identify(
        self,
        relation_observe: Callable[[Any], int],
        relation_proposals: Callable[[int], list[Any]],
        *,
        max_relation_queries: int = 16,
        proposal_count: int = 256,
        accept_posterior: float = 0.985,
        accept_margin: float = 0.90,
        bridge_observe: Optional[Callable[[Any], int]] = None,
        bridge_proposals: Optional[Callable[[int], list[Any]]] = None,
        max_bridge_queries: int = 2,
        obs_error_floor: float = 1e-4,
    ) -> Identification:
        n = len(self.sources)
        logp = np.zeros(n, dtype=np.float64) - math.log(n)
        rq = 0
        for _ in range(max_relation_queries):
            post = self._normalize(logp)
            order = np.argsort(post)[::-1]
            if post[order[0]] >= accept_posterior and (post[order[0]] - post[order[1]] if n > 1 else 1.0) >= accept_margin:
                return Identification(self.sources[int(order[0])].key, {s.key: float(p) for s,p in zip(self.sources, post)}, rq, 0, False)
            q, score, probs = self._pick_query(relation_proposals(proposal_count), post, bridge=False)
            if q is None or score < 1e-7:
                break
            y = int(relation_observe(q))
            probs = np.clip(probs, obs_error_floor, 1-obs_error_floor)
            logp += np.log(probs if y else (1-probs))
            rq += 1

        bq = 0
        if bridge_observe is not None and bridge_proposals is not None and all(s.bridge_prob is not None for s in self.sources):
            for _ in range(max_bridge_queries):
                post = self._normalize(logp)
                order = np.argsort(post)[::-1]
                if post[order[0]] >= accept_posterior and (post[order[0]] - post[order[1]] if n > 1 else 1.0) >= accept_margin:
                    return Identification(self.sources[int(order[0])].key, {s.key: float(p) for s,p in zip(self.sources, post)}, rq, bq, False)
                q, score, probs = self._pick_query(bridge_proposals(proposal_count), post, bridge=True)
                if q is None or score < 1e-7:
                    break
                y = int(bridge_observe(q))
                probs = np.clip(probs, obs_error_floor, 1-obs_error_floor)
                logp += np.log(probs if y else (1-probs))
                bq += 1

        post = self._normalize(logp)
        order = np.argsort(post)[::-1]
        ok = post[order[0]] >= accept_posterior and (post[order[0]] - post[order[1]] if n > 1 else 1.0) >= accept_margin
        key = self.sources[int(order[0])].key if ok else None
        return Identification(key, {s.key: float(p) for s,p in zip(self.sources, post)}, rq, bq, not ok)

class Composer:
    def __init__(self, registry: BehaviorRegistry):
        self.registry = registry

    def run(self, source_keys: list[str], initial_state: Any, stage_inputs: list[Any]):
        state = initial_state
        trace = [state]
        for key, external in zip(source_keys, stage_inputs):
            source = self.registry.by_key[key]
            state = source.execute_common((state, external))
            trace.append(state)
        return state, trace
