import torch

from opfusion import fusion_unseen_source_behavioral_admission as behavioral
from opfusion import fusion_unseen_source_generalization as unseen


def test_behavioral_consensus_uses_semantic_view_modal_fraction(monkeypatch):
    outputs = iter([7, 7, 3])
    monkeypatch.setattr(
        behavioral.sequential,
        'prompt_ids_for_values',
        lambda **kwargs: [1, 2, 3],
    )
    monkeypatch.setattr(
        behavioral.rollout,
        '_generate_model_value',
        lambda *args, **kwargs: next(outputs),
    )
    value = behavioral.behavioral_consensus(
        torch.nn.Identity(),
        operator='aggregation.sum',
        values=(1, 2, 3),
        factory=object(),
        tokenizer=object(),
        max_new_tokens=8,
        device=torch.device('cpu'),
    )
    assert value == 2 / 3


def test_behavioral_consensus_counts_parse_failure_against_score(monkeypatch):
    outputs = iter([5, None])
    monkeypatch.setattr(
        behavioral.sequential,
        'prompt_ids_for_values',
        lambda **kwargs: [1],
    )
    monkeypatch.setattr(
        behavioral.rollout,
        '_generate_model_value',
        lambda *args, **kwargs: next(outputs),
    )
    value = behavioral.behavioral_consensus(
        torch.nn.Identity(),
        operator='scalar.add',
        values=(2, 3),
        factory=object(),
        tokenizer=object(),
        max_new_tokens=8,
        device=torch.device('cpu'),
    )
    assert value == 0.5


def test_aggregate_mode_preserves_subset_accounting():
    reports = []
    for cohort_index in range(2):
        aggregate = behavioral.sequential._empty_counter()
        pairs = {}
        for inner in behavioral.FUNCTIONAL_OPERATORS:
            for outer in behavioral.FUNCTIONAL_OPERATORS:
                counter = behavioral.sequential._empty_counter()
                counter['cases'] = 1
                counter['end_to_end_correct'] = int(inner == 'scalar.add')
                behavioral.sequential._merge_counter(aggregate, counter)
                pairs[f'{inner}->{outer}'] = behavioral.sequential._finalize_counter(counter)
        reports.append({
            'aggregate': behavioral.sequential._finalize_counter(aggregate),
            'pairs': pairs,
            'diagnostics': {
                'mean_matching_heldout_consensus': 0.8,
                'mean_nonmatching_heldout_consensus': 0.2,
                'mean_admission_factor': 0.3,
                'mean_heldout_raw_gate': 0.4,
                'mean_heldout_effective_gate': 0.1,
            },
        })
    report = behavioral.aggregate_mode(reports, heldout='scalar.add')
    assert report['aggregate']['cases'] == 32
    assert report['subsets']['heldout_involved']['cases'] == 14
    assert report['subsets']['heldout_neither']['cases'] == 18
    assert report['diagnostics']['mean_matching_heldout_consensus'] == 0.8
