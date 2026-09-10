"""CPU tests for the explicit squared-distance ball query and neighbourhood reuse."""

import torch

from uipc_manip.models import ball_query, reuse_neighbourhoods


def _cdist_reference(pos, valid, centers, center_valid, radius, k):
    d = torch.cdist(centers, pos, compute_mode="donot_use_mm_for_euclid_dist")
    within = (d <= radius) & valid[:, None, :] & center_valid[:, :, None]
    score = torch.where(within, d, torch.full_like(d, float("inf")))
    values, idx = score.topk(min(int(k), pos.shape[1]), dim=-1, largest=False)
    return idx, torch.isfinite(values)


def _neighbour_sets(idx, ok):
    return torch.where(ok, idx, torch.full_like(idx, -1)).sort(dim=-1).values


def test_ball_query_matches_the_cdist_reference():
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(3, 400, 3, generator=g) * 0.4
    valid = torch.rand(3, 400, generator=g) > 0.1
    for radius, k in ((0.05, 8), (0.1, 16), (0.3, 64)):
        i0, ok0 = _cdist_reference(pos, valid, pos, valid, radius, k)
        i1, ok1 = ball_query(pos, valid, pos, valid, radius, k)
        assert torch.equal(ok0, ok1)
        # The aggregation is a masked max, so the neighbour set is what must agree.
        assert torch.equal(_neighbour_sets(i0, ok0), _neighbour_sets(i1, ok1))


def test_ball_query_with_distinct_centres_and_no_neighbours():
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(2, 50, 3, generator=g)
    valid = torch.ones(2, 50, dtype=torch.bool)
    centers = torch.rand(2, 7, 3, generator=g) + 5.0          # far from every point
    center_valid = torch.ones(2, 7, dtype=torch.bool)
    _, ok = ball_query(pos, valid, centers, center_valid, 0.1, 4)
    assert not ok.any()


def test_neighbourhood_reuse_serves_repeats_only_inside_the_block():
    pos = torch.rand(2, 64, 3)
    valid = torch.ones(2, 64, dtype=torch.bool)
    outside = ball_query(pos, valid, pos, valid, 0.2, 8)
    assert ball_query(pos, valid, pos, valid, 0.2, 8)[0] is not outside[0]
    with reuse_neighbourhoods():
        first = ball_query(pos, valid, pos, valid, 0.2, 8)
        assert ball_query(pos, valid, pos, valid, 0.2, 8)[0] is first[0]
        assert ball_query(pos, valid, pos, valid, 0.1, 8)[0] is not first[0]
        pos.add_(0.0)                                          # an in-place change bumps the version
        assert ball_query(pos, valid, pos, valid, 0.2, 8)[0] is not first[0]
    assert torch.equal(first[0], outside[0]) and torch.equal(first[1], outside[1])
