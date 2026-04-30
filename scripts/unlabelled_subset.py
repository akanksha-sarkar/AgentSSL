"""Shared COCO unlabelled JSON subsampling (images + optional annotations filter)."""
from __future__ import annotations

import random
from typing import Any


def subset_unlabelled_coco(
    data: dict[str, Any],
    rng_seed: int,
    *,
    fraction: float | None = None,
    count: int | None = None,
) -> dict[str, Any]:
    """
    Return a copy of ``data`` with a random subset of ``images``.

    Provide exactly one of ``fraction`` or ``count``:
    - ``count``: keep ``min(count, n)`` images (``count`` may be 0).
    - ``fraction``: keep ``int(n * fraction)`` images, except if that is 0 and
      ``fraction > 0`` and ``n > 0``, keep 1 image (tiny-fraction guard).
      If ``fraction == 0``, keep 0 images.
    """
    if (fraction is None) == (count is None):
        raise ValueError("Provide exactly one of fraction= or count=")

    images = data.get("images")
    if not isinstance(images, list):
        raise ValueError("Input JSON must have a list field 'images'")

    n = len(images)
    if count is not None:
        if count < 0:
            raise ValueError("count must be >= 0")
        k = min(count, n)
    else:
        assert fraction is not None
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("fraction must be in [0, 1]")
        if n == 0:
            k = 0
        else:
            k = max(0, int(n * fraction))
            if k == 0 and fraction > 0:
                k = 1

    rng = random.Random(rng_seed)
    indices = sorted(rng.sample(range(n), k)) if k > 0 else []
    new_images = [images[i] for i in indices]

    kept_ids = {img["id"] for img in new_images if "id" in img}

    out: dict[str, Any] = {}
    for key, val in data.items():
        if key == "images":
            out[key] = new_images
        elif key == "annotations":
            anns = val if isinstance(val, list) else []
            if not anns:
                out[key] = anns
            else:
                out[key] = [a for a in anns if a.get("image_id") in kept_ids]
        else:
            out[key] = val

    return out
