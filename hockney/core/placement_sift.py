"""
placement_sift.py — SIFT-based reference placement engine.

Alternative to DISK+LightGlue that uses OpenCV's SIFT feature detector
with FLANN-based matching.  SIFT is older but handles:
  - Low-quality / noisy images better (more robust descriptors)
  - Scale differences natively (Scale-Invariant Feature Transform)
  - No GPU required, no model download

Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing
to boost local contrast before feature extraction — critical for low-contrast
camera images like the Kodak Charmer.

Matching strategy:
  1. CLAHE + SIFT extract on reference at preview resolution
  2. CLAHE + SIFT extract on each detail at multiple scales
  3. FLANN knnMatch with Lowe's ratio test (0.7) for quality filtering
  4. RANSAC geometric verification on surviving matches
  5. Centroid of RANSAC inliers → canvas position
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

from hockney.core.placement import _snap_orientation

log = logging.getLogger(__name__)

# ── CLAHE helpers ─────────────────────────────────────────────────────────────

def clahe_preprocess(img_rgb: np.ndarray, clip_limit: float = 3.0,
                     tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE to the L channel of a LAB-converted image.
    Returns RGB.  Dramatically boosts local contrast for feature detection
    on flat/noisy images.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def light_denoise(img_rgb: np.ndarray, h: int = 3) -> np.ndarray:
    """Light denoising — preserve edges while reducing sensor noise."""
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, h, h, 7, 21)


# ── SIFT matching engine ─────────────────────────────────────────────────────

DETAIL_SCALES = [1.0, 0.5, 0.33, 0.25, 0.17]
LOWE_RATIO = 0.75         # Lowe's ratio test threshold
RANSAC_REPROJ = 5.0
MIN_GOOD_MATCHES = 6      # after ratio test + RANSAC
MIN_GOOD_RELAXED = 4      # second-pass


def extract_sift(img_rgb: np.ndarray, n_features: int = 0,
                 use_clahe: bool = True) -> tuple[list, Optional[np.ndarray]]:
    """
    Extract SIFT keypoints and descriptors from an RGB image.
    Returns (keypoints, descriptors) or ([], None) on failure.
    n_features=0 means unlimited (SIFT default).
    """
    if use_clahe:
        img_rgb = clahe_preprocess(img_rgb)

    grey = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp, desc = sift.detectAndCompute(grey, None)
    return kp, desc


def extract_sift_spatial(img_rgb: np.ndarray, n_features: int = 2000,
                         grid_cells: int = 4, use_clahe: bool = True
                         ) -> tuple[list, Optional[np.ndarray]]:
    """
    Extract SIFT with spatial binning — divides the image into a grid and
    extracts features from each cell separately.  This forces keypoints to
    spread across the image instead of clustering on high-contrast features
    (chrome, gauges, text) which causes the diagonal cascade problem.

    Returns (keypoints, descriptors).
    """
    if use_clahe:
        img_rgb = clahe_preprocess(img_rgb)

    grey = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = grey.shape
    features_per_cell = max(50, n_features // (grid_cells * grid_cells))
    sift = cv2.SIFT_create(nfeatures=features_per_cell)

    all_kp = []
    all_desc = []
    cell_h = h // grid_cells
    cell_w = w // grid_cells

    for row in range(grid_cells):
        for col in range(grid_cells):
            y0 = row * cell_h
            x0 = col * cell_w
            y1 = h if row == grid_cells - 1 else (row + 1) * cell_h
            x1 = w if col == grid_cells - 1 else (col + 1) * cell_w

            # Create mask for this cell
            mask = np.zeros(grey.shape, dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255

            kp, desc = sift.detectAndCompute(grey, mask)
            if kp and desc is not None:
                all_kp.extend(kp)
                all_desc.append(desc)

    if all_desc:
        return all_kp, np.vstack(all_desc)
    return [], None


def match_sift(desc_a: np.ndarray, desc_b: np.ndarray,
               ratio: float = LOWE_RATIO) -> list[cv2.DMatch]:
    """
    FLANN-based kNN matching with Lowe's ratio test.
    Returns list of good DMatch objects.
    """
    if desc_a is None or desc_b is None:
        return []
    if len(desc_a) < 2 or len(desc_b) < 2:
        return []

    # FLANN parameters for SIFT (float descriptors)
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return []

    # Lowe's ratio test
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def ransac_filter(kp_a, kp_b, matches,
                  reproj: float = RANSAC_REPROJ
                  ) -> tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    RANSAC filter on matched keypoints.
    Returns (M, inlier_kp_a, inlier_kp_b, inlier_mask) or (None, ...) on failure.
    M is the 2x3 affine matrix.
    """
    if len(matches) < 4:
        return None, np.array([]), np.array([]), np.array([])

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(pts_a, pts_b,
                                           method=cv2.RANSAC,
                                           ransacReprojThreshold=reproj)
    if M is None or mask is None:
        return None, np.array([]), np.array([]), np.array([])

    inlier_idx = mask.ravel().astype(bool)
    inlier_a = pts_a[inlier_idx].reshape(-1, 2)
    inlier_b = pts_b[inlier_idx].reshape(-1, 2)
    return M, inlier_a, inlier_b, inlier_idx


def place_with_sift(
    store,
    config,
    records,
    thumb_edge: int,
    min_matches: int = MIN_GOOD_MATCHES,
    min_matches_relaxed: int = MIN_GOOD_RELAXED,
    progress_cb=None,
    cancel_check=None,
) -> tuple[dict, list, int, dict]:
    """
    SIFT-based reference placement.

    Returns:
      placements: dict[image_id → ImagePlacement]
      odds_and_ends: list of unplaced (id, gx, gy, gr)
      placed_count: int
      log_data: dict of per-image match stats for the results log
    """
    from PIL import Image as PILImage, ImageOps
    from hockney.core.models import ImagePlacement

    PREVIEW_LONG_EDGE = 1500

    refs = config.references
    n_refs = len(refs)
    n_detail = len(records)
    total_work = n_refs + n_detail + n_detail * n_refs
    done = 0
    log_data = {}  # image_id → {engine, inliers, scale, slot, confidence, ...}

    # ── 1. Extract SIFT from references ───────────────────────────────────
    ref_kp = {}     # slot → keypoints
    ref_desc = {}   # slot → descriptors
    ref_sizes = {}  # slot → (w, h)

    for ref in refs:
        if cancel_check and cancel_check():
            return {}, [], 0, log_data
        try:
            pil = PILImage.open(ref.source_path)
            pil = ImageOps.exif_transpose(pil).convert("RGB")
            w, h = pil.size
            scale = PREVIEW_LONG_EDGE / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
            ref_sizes[ref.slot] = (new_w, new_h)

            arr = np.array(pil)
            kp, desc = extract_sift(arr, n_features=0, use_clahe=True)
            if desc is not None and len(kp) > 0:
                ref_kp[ref.slot] = kp
                ref_desc[ref.slot] = desc
                log.info("SIFT ref %s: %d keypoints", ref.slot, len(kp))
        except Exception as e:
            log.warning("SIFT ref extraction failed %s: %s", ref.slot, e)

        done += 1
        if progress_cb:
            progress_cb(int(done / total_work * 100))

    if not ref_kp:
        return {}, [(r.id, 0.0, 0.0, 0.0) for r in records], 0, log_data

    # ── 2. Extract SIFT from details at multiple scales ───────────────────
    detail_multiscale = {}  # image_id → [(scale, kp, desc, (w,h)), ...]

    for record in records:
        if cancel_check and cancel_check():
            return {}, [], 0, log_data
        try:
            preview_arr = store.get_preview(record.id)
            if preview_arr is None:
                preview_arr = store.get_thumbnail(record.id)
            if preview_arr is None:
                continue

            # Light denoise + CLAHE will be done inside extract_sift
            preview_arr = light_denoise(preview_arr, h=3)

            h_full, w_full = preview_arr.shape[:2]
            scales_data = []

            for ds in DETAIL_SCALES:
                new_w = max(64, int(w_full * ds))
                new_h = max(64, int(h_full * ds))

                if ds < 1.0:
                    resized = cv2.resize(preview_arr, (new_w, new_h),
                                         interpolation=cv2.INTER_AREA)
                else:
                    resized = preview_arr
                    new_w, new_h = w_full, h_full

                kp, desc = extract_sift(resized, n_features=0, use_clahe=True)
                if desc is not None and len(kp) > 0:
                    scales_data.append((ds, kp, desc, (new_w, new_h)))

            detail_multiscale[record.id] = scales_data
        except Exception as e:
            log.warning("SIFT detail extraction failed %s: %s", record.id[:6], e)
        done += 1
        if progress_cb:
            progress_cb(int(done / total_work * 100))

    # ── 3. Match each detail against references ───────────────────────────
    placements = {}
    odds_and_ends = []
    placed_count = 0

    for record in records:
        if cancel_check and cancel_check():
            return {}, [], 0, log_data

        if record.id not in detail_multiscale:
            odds_and_ends.append((record.id, 0, None, 0.0))
            done += n_refs
            if progress_cb:
                progress_cb(int(done / total_work * 100))
            continue

        best_slot = None
        best_inliers = 0
        best_ref_centroid = None
        best_rot = 0.0
        best_scale = 1.0

        for slot in ref_kp:
            for ds, d_kp, d_desc, (dw, dh) in detail_multiscale[record.id]:
                try:
                    good = match_sift(ref_desc[slot], d_desc, ratio=LOWE_RATIO)
                    if len(good) < 4:
                        continue

                    M, inlier_a, inlier_b, mask = ransac_filter(
                        ref_kp[slot], d_kp, good, reproj=RANSAC_REPROJ
                    )
                    if M is None:
                        continue

                    n_inliers = len(inlier_a)
                    if n_inliers > best_inliers and n_inliers >= min_matches:
                        from hockney.core.placement import _homography_center
                        rw, rh = ref_sizes.get(slot, (PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE))
                        # Wider margin for SIFT — its homographies are noisier
                        center = _homography_center(
                            inlier_b, inlier_a, dw, dh, rw, rh,
                            margin=0.5)
                        if center is None:
                            # Still use centroid rather than skipping entirely
                            cx = float(inlier_a[:, 0].mean())
                            cy = float(inlier_a[:, 1].mean())
                        else:
                            cx, cy = center

                        rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))

                        best_slot = slot
                        best_inliers = n_inliers
                        best_ref_centroid = (cx, cy)
                        best_rot = rot
                        best_scale = ds
                except Exception as e:
                    log.debug("SIFT match %s↔%s@%.0f%% failed: %s",
                              slot, record.id[:6], ds * 100, e)

            done += 1
            if progress_cb:
                progress_cb(int(done / total_work * 100))

        # Log match result regardless of success
        log_data[record.id] = {
            "engine": "sift",
            "inliers": best_inliers,
            "scale": best_scale,
            "slot": best_slot or "none",
            "placed": best_slot is not None and best_ref_centroid is not None,
        }

        if best_slot and best_ref_centroid:
            cx, cy = best_ref_centroid

            # Perspective offset for multi-ref projects
            if config.project_type == "perspective":
                offsets = {
                    "standing": (0, 0), "left": (-1, 0), "right": (1, 0),
                    "down_low": (0, 1), "up_high": (0, -1),
                }
                ox, oy = offsets.get(best_slot, (0, 0))
                rw, rh = ref_sizes.get(best_slot,
                                       (PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE))
                cx += ox * rw
                cy += oy * rh

            final_x = cx
            final_y = cy

            # Centre tile on match point
            aspect = record.width / max(record.height, 1)
            if aspect >= 1:
                tile_w = thumb_edge
                tile_h = thumb_edge / aspect
            else:
                tile_w = thumb_edge * aspect
                tile_h = thumb_edge
            final_x -= tile_w / 2
            final_y -= tile_h / 2

            p = ImagePlacement(
                image_id=record.id,
                x=final_x, y=final_y, rotation=_snap_orientation(best_rot),
                z_order=placed_count,
                auto_x=final_x, auto_y=final_y, auto_rotation=_snap_orientation(best_rot),
            )
            placements[record.id] = p
            placed_count += 1
            log.info("SIFT placed %s in %s (%d inliers, scale=%.0f%%)",
                     record.id[:6], best_slot, best_inliers, best_scale * 100)
        else:
            odds_and_ends.append((record.id, best_inliers,
                                  best_ref_centroid, best_rot))

    # ── 3b. Second pass at relaxed threshold ──────────────────────────────
    still_unmatched = []
    for uid, n_inliers, ref_centroid, rot in odds_and_ends:
        if n_inliers >= min_matches_relaxed and ref_centroid:
            cx, cy = ref_centroid
            record = store.get_record(uid)
            if record:
                aspect = record.width / max(record.height, 1)
                tile_w = thumb_edge if aspect >= 1 else thumb_edge * aspect
                tile_h = thumb_edge / aspect if aspect >= 1 else thumb_edge
            else:
                tile_w = tile_h = thumb_edge
            final_x = cx - tile_w / 2
            final_y = cy - tile_h / 2
            p = ImagePlacement(
                image_id=uid,
                x=final_x, y=final_y, rotation=_snap_orientation(rot),
                z_order=placed_count,
                auto_x=final_x, auto_y=final_y, auto_rotation=_snap_orientation(rot),
            )
            placements[uid] = p
            placed_count += 1
            log_data[uid]["placed"] = True
            log_data[uid]["pass"] = "relaxed"
            log.info("SIFT second-pass placed %s (%d inliers)", uid[:6], n_inliers)
        else:
            gx, gy, gr = 0.0, 0.0, 0.0
            if ref_centroid:
                gx, gy = ref_centroid
                gr = rot
            still_unmatched.append((uid, gx, gy, gr))

    odds_and_ends_final = [(uid, gx, gy, gr) for uid, gx, gy, gr in still_unmatched]
    return placements, odds_and_ends_final, placed_count, log_data
