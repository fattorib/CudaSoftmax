import numpy as np
import pytest
import torch
from softmax_cuda import fusedSoftmax


def relative_error(x: torch.FloatTensor, y: torch.FloatTensor) -> float:
    """Computes ||x-y||_2 / ||y||_2.
    We expect this to be < 1e-06 (fp32 is accurate to 7-8 digits)
    """
    return torch.linalg.norm(x - y) / torch.linalg.norm(y)


def ref_softmax(x: torch.FloatTensor) -> torch.FloatTensor:
    """PyTorch reference softmax."""
    return torch.nn.functional.softmax(x, dim=-1)


@pytest.mark.parametrize(
    "dims",
    [
        (r, c)
        for r in np.random.randint(128, 8192, size=10)
        for c in [1024, 2048, 2560, 3072, 4096, 5120, 7680, 8192, 16384]
    ],
)
def test_softmax_fwd_fp32(dims):
    rows, cols = dims

    x = torch.randn(
        (rows, cols), device="cuda:0", dtype=torch.float32, requires_grad=False
    )

    out_torch = ref_softmax(x)

    out_cuda = fusedSoftmax(x)

    # check forward
    assert relative_error(out_cuda, out_torch) < 1e-06
