"""Wang RSS 2023 garment curriculum for batched slots.

Wang's multi-garment launcher (``curl/train_multi_garments.py``) raises
``curriculum_step = step // curriculum_update_freq + 1`` and samples the next
episode's garment only among the first ``curriculum_step`` garments of its
order. A batched simulator binds each slot to one garment for the life of the
environment, so, as in the Newton port, the schedule instead gates which slots
may write to replay. Every slot keeps stepping so the world stays in sync.
"""

from __future__ import annotations

# Wang's launcher order for the five garments of the cached distribution.
WANG_GARMENT_ORDER: tuple[str, ...] = ("hospital_gown", "tshirt_26", "tshirt_68", "tshirt_4", "tshirt_392")


def garment_curriculum_stage(step: int, *, interval: int, garment_count: int) -> int:
    """Number of garments whose slots feed replay at ``step``.

    A non-positive ``interval`` disables the curriculum and activates every
    garment at once. Otherwise garment ``k`` (zero-based) joins at step
    ``k * interval``, which is Wang's ``step // curriculum_update_freq + 1``.
    """
    if interval <= 0:
        return int(garment_count)
    return max(1, min(int(garment_count), int(step) // int(interval) + 1))


def curriculum_order(requested: list[str] | tuple[str, ...] | None, present: list[str] | tuple[str, ...]) -> list[str]:
    """Order the garments actually present, easiest first.

    ``requested`` is a preference list: names present in this run come first in
    that order, names absent are ignored (the default names all five reference
    garments), and garments present but not named are appended in their own order.
    """
    present_list = list(dict.fromkeys(present))
    requested_list = [name for name in (requested or ()) if name]
    order = [name for name in requested_list if name in present_list]
    order += [name for name in present_list if name not in order]
    return order
