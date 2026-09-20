import numpy as np
import pytest

from uipc_manip.action_selector_audit import heldout_choices, make_bank


def test_selection_obeys_executed_dimensions_box_and_common_radius():
    base = np.array([[0.99, -0.98, 0.0, 0.8, 0.1, -0.3]], dtype=np.float32)
    weights = np.array([1, -2, 3, 1000, -4, 5])
    bank, q, chosen = make_bank(base, lambda a: a @ weights,
                               lambda a: np.broadcast_to(weights, a.shape).astype(float).copy(),
                               np.random.default_rng(1))
    assert np.allclose(bank[..., 3], base[..., 3])
    assert np.max(np.abs(bank)) <= 1
    assert np.max(np.linalg.norm(bank - base[:, None], axis=-1)) <= 0.500001
    assert q[0, 1] > q[0, 0]
    assert q[0, chosen[0]] >= q[0, 0]
    assert chosen[0] != 1  # The sampled arm must not reuse the gradient proposal.


def test_holdout_selection_does_not_use_its_scoring_repeat():
    # Candidate 1 wins only in repeat 0; selecting on that repeat would leak.
    scores = np.array([[[1.0, 1.0, 1.0], [100.0, 0.0, 0.0]]])
    assert heldout_choices(scores).tolist() == [[0, 1, 1]]
    scores[0, 1, 0] = 1000
    assert heldout_choices(scores)[0, 0] == 0
    with pytest.raises(ValueError):
        heldout_choices(scores[..., :1])
