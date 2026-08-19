from types import SimpleNamespace

import torch

from opfusion import fusion_mixed_novel_plugin_admission as mixed


def test_selected_logits_keeps_seen_sources_and_appends_two_novel_handles(monkeypatch):
    vocab = 7
    base = SimpleNamespace(logits=torch.zeros(vocab))
    units = {
        source: SimpleNamespace(logits=torch.full((vocab,), float(index + 1)))
        for index, source in enumerate((*mixed.FUNCTIONAL_OPERATORS, mixed.NUISANCE_SOURCE))
    }
    monkeypatch.setattr(mixed, '_next_logits', lambda model, ids: model.logits)
    heldout = 'scalar.add'
    names, logits = mixed._selected_logits(
        base=base,
        units=units,
        ids=torch.tensor([[1]]),
        heldout=heldout,
        appended=(heldout, mixed.NUISANCE_SOURCE),
    )
    assert names[0] == 'Base'
    assert heldout not in names[1:4]
    assert names[-2:] == [heldout, mixed.NUISANCE_SOURCE]
    assert logits.shape == (6, vocab)


def test_selected_logits_useful_only_has_variable_source_count(monkeypatch):
    vocab = 5
    base = SimpleNamespace(logits=torch.zeros(vocab))
    units = {
        source: SimpleNamespace(logits=torch.ones(vocab))
        for source in (*mixed.FUNCTIONAL_OPERATORS, mixed.NUISANCE_SOURCE)
    }
    monkeypatch.setattr(mixed, '_next_logits', lambda model, ids: model.logits)
    names, logits = mixed._selected_logits(
        base=base,
        units=units,
        ids=torch.tensor([[1]]),
        heldout='scalar.max',
        appended=('scalar.max',),
    )
    assert len(names) == 5
    assert logits.shape == (5, vocab)
    assert mixed.NUISANCE_SOURCE not in names


def test_correct_count_reads_subset_counter():
    block = {
        'subsets': {
            'heldout_involved': {'end_to_end_correct': 11},
            'heldout_neither': {'end_to_end_correct': 17},
        }
    }
    assert mixed._correct_count(block, 'heldout_involved') == 11
    assert mixed._correct_count(block, 'heldout_neither') == 17
