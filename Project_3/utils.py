
import torch
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt 




def generate_clicks_random(mask, num_pos_clicks=5, num_neg_clicks=5):
    """Random sampling strategy."""
    click_mask = torch.full_like(mask, -100)
    pos_coords = (mask == 1).nonzero(as_tuple=False)
    neg_coords = (mask == 0).nonzero(as_tuple=False)

    if len(pos_coords) > 0:
        pos_indices = np.random.choice(len(pos_coords), size=num_pos_clicks, replace=True)
        for idx in pos_indices:
            coord = pos_coords[idx]
            click_mask[coord[0], coord[1], coord[2]] = 1
    
    if len(neg_coords) > 0:
        neg_indices = np.random.choice(len(neg_coords), size=num_neg_clicks, replace=True)
        for idx in neg_indices:
            coord = neg_coords[idx]
            click_mask[coord[0], coord[1], coord[2]] = 0
            
    return click_mask


def _jitter_points(coords, region_mask, max_jitter=2, num_attempts=5):
    """
    Small random shift around each chosen point while staying inside the same region
    (region_mask True = allowed).
    """
    if len(coords) == 0:
        return coords

    h, w = region_mask.shape
    jittered = []

    for (y, x) in coords:
        yy, xx = int(y), int(x)
        for _ in range(num_attempts):
            dy = np.random.randint(-max_jitter, max_jitter + 1)
            dx = np.random.randint(-max_jitter, max_jitter + 1)
            ny, nx = yy + dy, xx + dx
            if 0 <= ny < h and 0 <= nx < w and region_mask[ny, nx]:
                yy, xx = ny, nx
                break
        jittered.append([yy, xx])

    return np.array(jittered, dtype=np.int32)

def _sample_with_min_distance(coords,
                              num_to_sample,
                              min_dist,
                              weights=None,
                              existing_points=None,
                              max_tries_per_point=200,
                              relax_factor=0.9):
    """
    Randomly sample points from `coords` such that they are at least
    `min_dist` pixels apart (approximately). If it becomes impossible
    to find more points, the distance is gradually relaxed.

    coords: array of shape (N, 2) with [y, x]
    weights: optional sampling weights (same length as coords)
    existing_points: list/array of [y, x] already chosen points to
                     enforce distance against.
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

    if existing_points is None:
        all_fixed = []
    else:
        all_fixed = [np.array(p, dtype=np.float64) for p in existing_points]

    chosen = []

    for _ in range(num_to_sample):
        success = False
        tries = 0

        while not success and tries < max_tries_per_point:
            idx = np.random.choice(n_candidates, p=weights)
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
def generate_clicks_spaced(mask, num_pos_clicks=5, num_neg_clicks=5):
    """
    Human-ish spaced sampling:

    Positives:
      - 1 central click (with jitter) inside the lesion
      - remaining clicks inside the lesion, biased towards the boundary,
        spaced out by a minimum distance, with some randomness & jitter.

    Negatives:
      - random background clicks anywhere, but with a minimum distance
        so they don't clump, and with relaxed distance if needed.
    """

    # Hyperparameters (you can tweak these)
    pos_min_dist = 35      # min distance between positive clicks
    neg_min_dist = 60      # min distance between negative clicks
    pos_jitter = 5         # max jitter in pixels for positive click
    neg_jitter = 5         # max jitter in pixels for negative click
    jitter_attempts = 10   # tries to keep jittered point valid

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
        def jitter_and_snap(coord):
            if pos_jitter <= 0:
                return np.array(coord, dtype=np.int64)

            coord = np.array(coord, dtype=np.int64)
            for _ in range(jitter_attempts):
                dy = np.random.randint(-pos_jitter, pos_jitter + 1)
                dx = np.random.randint(-pos_jitter, pos_jitter + 1)
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
            not_central = ~((pos_coords[:, 0] == central[0]) &
                            (pos_coords[:, 1] == central[1]))
            pos_candidates = pos_coords[not_central]

            if pos_candidates.shape[0] > 0:
                candidate_dists = dist_to_bg[
                    pos_candidates[:, 0], pos_candidates[:, 1]
                ]
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
                for p in extra_pos:
                    pos_points.append(jitter_and_snap(p))

    # write positives into click_mask
    for (y, x) in pos_points:
        click_mask[0, int(y), int(x)] = 1

    # -------------------- NEGATIVE CLICKS --------------------
    neg_coords = np.argwhere(~mask_np)  # all background pixels

    # Helper: jitter a background point but keep it in background
    def jitter_bg(coord):
        if neg_jitter <= 0:
            return np.array(coord, dtype=np.int64)

        coord = np.array(coord, dtype=np.int64)
        for _ in range(jitter_attempts):
            dy = np.random.randint(-neg_jitter, neg_jitter + 1)
            dx = np.random.randint(-neg_jitter, neg_jitter + 1)
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
        for p in base_neg_points:
            neg_points.append(jitter_bg(p))

    for (y, x) in neg_points:
        click_mask[0, int(y), int(x)] = 0

    return click_mask







def generate_clicks(mask, num_pos_clicks=5, num_neg_clicks=5, strategy='random'):
    """Main wrapper function to choose the click strategy."""
    if strategy == 'random':
        return generate_clicks_random(mask, num_pos_clicks, num_neg_clicks)
    elif strategy == 'spaced':
        return generate_clicks_spaced(mask, num_pos_clicks, num_neg_clicks)
    else:
        raise ValueError(f"Unknown click strategy: {strategy}")