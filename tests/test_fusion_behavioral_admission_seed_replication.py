import torch

from opfusion import fusion_behavioral_admission_seed_replication as replication


def test_deterministic_runtime_configuration():
    report = replication.configure_deterministic_runtime()
    assert report['num_threads'] == 1
    assert report['deterministic_algorithms'] is True
    assert report['mkldnn_enabled'] is False


def test_same_seed_reproduces_torch_initialization_after_runtime_config():
    replication.configure_deterministic_runtime()
    torch.manual_seed(12345)
    first = torch.nn.Linear(7, 5).weight.detach().clone()
    torch.manual_seed(12345)
    second = torch.nn.Linear(7, 5).weight.detach().clone()
    assert torch.equal(first, second)
