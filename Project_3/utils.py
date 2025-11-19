import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def generate_clicks_random(mask: torch.Tensor, num_pos_clicks: int = 5, num_neg_clicks: int = 5) -> torch.Tensor:
    """Run simulations of user clicks by sampling points from a full segmentation mask.

    Args:
        mask (torch.Tensor): The ground truth mask, shape [1, H, W].
        num_pos_clicks (int): Number of positive clicks to generate.
        num_neg_clicks (int): Number of negative clicks to generate.

    Returns:
        torch.Tensor: A "click mask" of the same shape as the input, with:
                      - 1 at positive click locations
                      - 0 at negative click locations
                      - -100 (an ignore value) everywhere else

    """
    # Create an empty click mask with the ignore value
    click_mask = torch.full_like(mask, -100)
    pos_coords = (mask == 1).nonzero(as_tuple=False)
    neg_coords = (mask == 0).nonzero(as_tuple=False)

    if len(pos_coords) > 0:
        # Randomly choose indices for the positive clicks
        pos_indices = np.random.choice(len(pos_coords), size=num_pos_clicks, replace=True)  # noqa: NPY002
        for idx in pos_indices:
            coord = pos_coords[idx]
            click_mask[coord[0], coord[1], coord[2]] = 1

    # --- Sample Negative Clicks ---
    if len(neg_coords) > 0:
        neg_indices = np.random.choice(len(neg_coords), size=num_neg_clicks, replace=True)  # noqa: NPY002
        for idx in neg_indices:
            coord = neg_coords[idx]
            click_mask[coord[0], coord[1], coord[2]] = 0

    return click_mask


def _sample_with_min_distance(
    coords: np.ndarray,
    num_to_sample: int,
    min_dist: float,
    weights: np.ndarray | None = None,
    existing_points: list | np.ndarray | None = None,
    max_tries_per_point: int = 200,
    relax_factor: float = 0.9,
) -> np.ndarray:
    """Randomly sample points from coords with minimum distance constraint.

    Samples points from coords such that they are at least min_dist pixels
    apart (approximately). If it becomes impossible to find more points,
    the distance is gradually relaxed.

    Args:
        coords: Array of shape (N, 2) with [y, x]
        num_to_sample: Number of points to sample
        min_dist: Minimum distance between sampled points
        weights: Optional sampling weights (same length as coords)
        existing_points: List/array of [y, x] already chosen points to
                        enforce distance against
        max_tries_per_point: Maximum attempts to find valid point before relaxing
        relax_factor: Factor to reduce minimum distance when relaxing constraint

    Returns:
        Array of sampled points with shape (num_sampled, 2)

    """
    coords = np.asarray(coords)
    n_candidates = coords.shape[0]

    if n_candidates == 0 or num_to_sample <= 0:
        return np.empty((0, 2), dtype=np.int64)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.maximum(weights, 1e-6)
        weights = weights / weights.sum()
    else:
        # uniform
        weights = np.full(n_candidates, 1.0 / n_candidates, dtype=np.float64)

    current_min = float(min_dist)

    all_fixed = [] if existing_points is None else [np.array(p, dtype=np.float64) for p in existing_points]

    chosen = []

    for _ in range(num_to_sample):
        success = False
        tries = 0

        while not success and tries < max_tries_per_point:
            rng = np.random.default_rng()
            idx = rng.choice(n_candidates, p=weights)
            cand = coords[idx].astype(np.float64)

            all_points = all_fixed + chosen
            if len(all_points) > 0:
                dists = np.linalg.norm(np.stack(all_points) - cand, axis=1)
                if np.all(dists >= current_min):
                    success = True
                else:
                    tries += 1
            else:
                success = True

            if success:
                chosen.append(cand)

        if not success:
            # could not find a new point with this min distance -> relax
            current_min *= relax_factor

    if len(chosen) == 0:
        return np.empty((0, 2), dtype=np.int64)

    return np.round(np.stack(chosen)).astype(np.int64)


# ---------------------------------------------------------------------
# Main spaced strategy
# ---------------------------------------------------------------------
def generate_clicks_spaced(mask: torch.Tensor, num_pos_clicks: int = 5, num_neg_clicks: int = 5) -> torch.Tensor:
    """Generate human-like spaced click samples from segmentation mask.

    Positives:
      - 1 central click (with jitter) inside the lesion
      - remaining clicks inside the lesion, biased towards the boundary,
        spaced out by a minimum distance, with some randomness & jitter.

    Negatives:
      - random background clicks anywhere, but with a minimum distance
        so they don't clump, and with relaxed distance if needed.

    Args:
        mask: Ground truth segmentation mask
        num_pos_clicks: Number of positive clicks to generate
        num_neg_clicks: Number of negative clicks to generate

    Returns:
        Click mask with 1 at positive clicks, 0 at negative clicks, -100 elsewhere

    """
    # Hyperparameters (you can tweak these)
    pos_min_dist = 35  # min distance between positive clicks
    neg_min_dist = 60  # min distance between negative clicks
    pos_jitter = 5  # max jitter in pixels for positive click
    neg_jitter = 5  # max jitter in pixels for negative click
    jitter_attempts = 10  # tries to keep jittered point valid

    click_mask = torch.full_like(mask, -100)

    # assume shape (1, H, W) or (H, W)
    mask_np = mask.squeeze().cpu().numpy().astype(bool)
    h, w = mask_np.shape

    # -------------------- POSITIVE CLICKS --------------------
    pos_coords = np.argwhere(mask_np)  # (N_pos, 2), [y, x]

    pos_points = []

    if pos_coords.shape[0] > 0 and num_pos_clicks > 0:
        # Distance transform to get a "most central" pixel
        dist_to_bg = distance_transform_edt(mask_np)
        central_flat_idx = np.argmax(dist_to_bg)
        cy, cx = np.unravel_index(central_flat_idx, mask_np.shape)
        central = np.array([cy, cx], dtype=np.int64)

        # Helper: jitter a lesion point but keep it inside the lesion
        def jitter_and_snap(coord: np.ndarray) -> np.ndarray:
            if pos_jitter <= 0:
                return np.array(coord, dtype=np.int64)

            coord = np.array(coord, dtype=np.int64)
            rng = np.random.default_rng()
            for _ in range(jitter_attempts):
                dy = rng.integers(-pos_jitter, pos_jitter + 1)
                dx = rng.integers(-pos_jitter, pos_jitter + 1)
                yy = np.clip(coord[0] + dy, 0, h - 1)
                xx = np.clip(coord[1] + dx, 0, w - 1)
                if mask_np[yy, xx]:
                    return np.array([yy, xx], dtype=np.int64)

            # fall back: nearest lesion pixel
            dists = np.linalg.norm(pos_coords - coord, axis=1)
            return pos_coords[np.argmin(dists)]

        # First central click (jittered)
        central = jitter_and_snap(central)
        pos_points.append(central)

        # Remaining positives: prefer boundary -> weight = 1 / (dist + 1)
        remaining = max(num_pos_clicks - 1, 0)
        if remaining > 0:
            # Remove the central pixel from candidates to avoid duplicates
            not_central = ~((pos_coords[:, 0] == central[0]) & (pos_coords[:, 1] == central[1]))
            pos_candidates = pos_coords[not_central]

            if pos_candidates.shape[0] > 0:
                candidate_dists = dist_to_bg[pos_candidates[:, 0], pos_candidates[:, 1]]
                # small distance -> near boundary -> larger weight
                pos_weights = 1.0 / (candidate_dists + 1.0)

                extra_pos = _sample_with_min_distance(
                    pos_candidates,
                    num_to_sample=remaining,
                    min_dist=pos_min_dist,
                    weights=pos_weights,
                    existing_points=[central],
                )

                # apply jitter + snap to lesion for each extra point
                pos_points.extend(jitter_and_snap(p) for p in extra_pos)

    # write positives into click_mask
    for y, x in pos_points:
        click_mask[0, int(y), int(x)] = 1

    # -------------------- NEGATIVE CLICKS --------------------
    neg_coords = np.argwhere(~mask_np)  # all background pixels

    # Helper: jitter a background point but keep it in background
    def jitter_bg(coord: np.ndarray) -> np.ndarray:
        if neg_jitter <= 0:
            return np.array(coord, dtype=np.int64)

        coord = np.array(coord, dtype=np.int64)
        rng = np.random.default_rng()
        for _ in range(jitter_attempts):
            dy = rng.integers(-neg_jitter, neg_jitter + 1)
            dx = rng.integers(-neg_jitter, neg_jitter + 1)
            yy = np.clip(coord[0] + dy, 0, h - 1)
            xx = np.clip(coord[1] + dx, 0, w - 1)
            # stay in background
            if not mask_np[yy, xx]:
                return np.array([yy, xx], dtype=np.int64)

        # fall back: nearest background pixel
        dists = np.linalg.norm(neg_coords - coord, axis=1)
        return neg_coords[np.argmin(dists)]

    neg_points = []
    if neg_coords.shape[0] > 0 and num_neg_clicks > 0:
        base_neg_points = _sample_with_min_distance(
            neg_coords,
            num_to_sample=num_neg_clicks,
            min_dist=neg_min_dist,
            weights=None,
            existing_points=None,
        )

        # apply jitter to each negative point
        neg_points.extend(jitter_bg(p) for p in base_neg_points)

    for y, x in neg_points:
        click_mask[0, int(y), int(x)] = 0

    return click_mask


def generate_clicks(
    mask: torch.Tensor,
    num_pos_clicks: int = 5,
    num_neg_clicks: int = 5,
    strategy: str = "random",
) -> torch.Tensor:
    """Choose and apply click generation strategy to segmentation mask.

    Args:
        mask: Ground truth segmentation mask
        num_pos_clicks: Number of positive clicks to generate
        num_neg_clicks: Number of negative clicks to generate
        strategy: Click generation strategy ("random" or "spaced")

    Returns:
        Click mask with 1 at positive clicks, 0 at negative clicks, -100 elsewhere

    Raises:
        ValueError: If unknown strategy is specified

    """
    if strategy == "random":
        return generate_clicks_random(mask, num_pos_clicks, num_neg_clicks)
    if strategy == "spaced":
        return generate_clicks_spaced(mask, num_pos_clicks, num_neg_clicks)
    msg = f"Unknown click strategy: {strategy}"
    raise ValueError(msg)
