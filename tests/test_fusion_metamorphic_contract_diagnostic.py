from opfusion import fusion_metamorphic_contract_diagnostic as meta


def test_additive_contract_perturbs_each_operand_with_same_output_delta():
    probes = meta.contract_probes("aggregation.sum", (2, 5, 16))
    assert len(probes) == 3
    assert probes[0].values == (3, 5, 16)
    assert probes[0].output_delta == 1
    assert probes[2].values == (2, 5, 15)
    assert probes[2].output_delta == -1


def test_min_contract_contains_effectful_and_inert_probe():
    probes = meta.contract_probes("scalar.min", (1, 5, 3))
    keyed = {probe.probe_id: probe for probe in probes}
    assert keyed["lower_min"].values == (0, 5, 3)
    assert keyed["lower_min"].output_delta == -1
    assert keyed["raise_non_min"].output_delta == 0


def test_max_contract_contains_effectful_and_inert_probe():
    probes = meta.contract_probes("scalar.max", (1, 5, 3))
    keyed = {probe.probe_id: probe for probe in probes}
    assert keyed["raise_max"].values == (1, 6, 3)
    assert keyed["raise_max"].output_delta == 1
    assert keyed["lower_non_max"].output_delta == 0


def test_contract_summary_uses_output_relations_not_absolute_answer():
    rows = [
        {"modal_value": 101, "consensus": 1.0, "output_delta": 1},
        {"modal_value": 100, "consensus": 0.5, "output_delta": 0},
    ]
    summary = meta.summarize_contract(100, 1.0, rows)
    assert summary["contract_probes"] == 2
    assert summary["contract_passes"] == 2
    assert summary["contract_rate"] == 1.0
    assert summary["contract_weighted"] == 0.75


def test_contract_summary_rejects_stable_wrong_relation():
    rows = [
        {"modal_value": 100, "consensus": 1.0, "output_delta": 1},
        {"modal_value": 102, "consensus": 1.0, "output_delta": 0},
    ]
    summary = meta.summarize_contract(100, 1.0, rows)
    assert summary["contract_passes"] == 0
    assert summary["contract_rate"] == 0.0
