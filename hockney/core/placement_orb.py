"""
placement_orb.py — ORB-based reference placement engine.

ORB (Oriented FAST and Rotated BRIEF) is OpenCV's lightweight, patent-free
keypoint detector.  It's very fast but less accurate than SIFT or learned
methods.  Good for:
  - Quick previews / drafts before running a heavier engine
  - Low-powered machines without GPU
  - Images with strong corners and edges (buildings, text, vehicles)

Uses BFMatcher with Hamming distance (ORB produces binary descriptors).
No CLAHE — ORB's FAST detector already handles contrast variation.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

DETAIL_SCALES = [1.0, 0.5, 0.33, 0.25]
RANSAC_REPROJ = 5.0
LOWE_RATIO = 0.75
MIN_GOOD_MATCHES = 8
MIN_GOOD_RELAXED = 5


def extract_orb(img_rgb: np.ndarray, n_features: int = 3000
                ) -> tuple[list, Optional[np.ndarray]]:
    """Extract ORB keypoints and descriptors from an RGB image."""
    grey = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8)
    kp, desc = orb.detectAndCompute(grey, None)
    return kp, desc


def match_orb(desc_a: np.ndarray, desc_b: np.ndarray,
              ratio: float = LOWE_RATIO) -> list[cv2.DMatch]:
    """
    BFMatcher with Hamming distance + Lowe's ratio test for ORB.
    ORB descriptors are binary, so we use NORM_HAMMING.
    """
    if desc_a is None or desc_b is None:
        return []
    if len(desc_a) < 2 or len(desc_b) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return []

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
    """RANSAC filter on matched ORB keypoints."""
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


def place_with_orb(
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
    ORB-based reference placement.

    Returns:
      placements: dict[image_id -> ImagePlacement]
      odds_and_ends: list of unplaced (id, gx, gy, gr)
      placed_count: int
      log_data: dict of per-image match stats
    """
    from PIL import Image as PILImage, ImageOps
    from hockney.core.models import ImagePlacement
    from hockney.core.placement_sift import light_denoise

    PREVIEW_LONG_EDGE = 1500

    refs = config.references
    n_refs = len(refs)
    n_detail = len(records)
    total_work = n_refs + n_detail + n_detail * n_refs
    done = 0
    log_data = {}

    # ── 1. Extract ORB from references ─────────────────────────────────
    ref_kp = {}
    ref_desc = {}
    ref_sizes = {}

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
            kp, desc = extract_orb(arr, n_features=5000)
            if desc is not None and len(kp) > 0:
                ref_kp[ref.slot] = kp
                ref_desc[ref.slot] = desc
                log.info("ORB ref %s: %d keypoints", ref.slot, len(kp))
        except Exception as e:
            log.warning("ORB ref extraction failed %s: %s", ref.slot, e)

        done += 1
        if progress_cb:
            progress_cb(int(done / total_work * 100))

    if not ref_kp:
        return {}, [(r.id, 0.0, 0.0, 0.0) for r in records], 0, log_data

    # ── 2. Extract ORB from details at multiple scales ─────────────────
    detail_multiscale = {}

    for record in records:
        if cancel_check and cancel_check():
            return {}, [], 0, log_data
        try:
            preview_arr = store.get_preview(record.id)
            if preview_arr is None:
                preview_arr = store.get_thumbnail(record.id)
            if preview_arr is None:
                continue

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

                kp, desc = extract_orb(resized, n_features=3000)
                if desc is not None and len(kp) > 0:
                    scales_data.append((ds, kp, desc, (new_w, new_h)))

            detail_multiscale[record.id] = scales_data
        except Exception as e:
            log.warning("ORB detail extraction failed %s: %s", record.id[:6], e)
        done += 1
        if progress_cb:
            progress_cb(int(done / total_work * 100))

    # ── 3. Match each detail against references ────────────────────────
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
                    good = match_orb(ref_desc[slot], d_desc, ratio=LOWE_RATIO)
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
                        center = _homography_center(
                            inlier_b, inlier_a, dw, dh, rw, rh,
                            margin=0.5)
                        if center is None:
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
                    log.debug("ORB match %s<>%s@%.0f%% failed: %s",
                              slot, record.id[:6], ds * 100, e)

            done += 1
            if progress_cb:
                progress_cb(int(done / total_work * 100))

        log_data[record.id] = {
            "engine": "orb",
            "inliers": best_inliers,
            "scale": best_scale,
            "slot": best_slot or "none",
            "placed": best_slot is not None and best_ref_centroid is not None,
        }

        if best_slot and best_ref_centroid:
            cx, cy = best_ref_centroid

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
                x=final_x, y=final_y, rotation=best_rot,
                z_order=placed_count,
                auto_x=final_x, auto_y=final_y, auto_rotation=best_rot,
            )
            placements[record.id] = p
            placed_count += 1
            log.info("ORB placed %s in %s (%d inliers, scale=%.0f%%)",
                     record.id[:6], best_slot, best_inliers, best_scale * 100)
        else:
            odds_and_ends.append((record.id, best_inliers,
                                  best_ref_centroid, best_rot))

    # ── 3b. Second pass at relaxed threshold ───────────────────────────
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
                x=final_x, y=final_y, rotation=rot,
                z_order=placed_count,
                auto_x=final_x, auto_y=final_y, auto_rotation=rot,
            )
            placements[uid] = p
            placed_count += 1
            log_data[uid]["placed"] = True
            log_data[uid]["pass"] = "relaxed"
        else:
            gx, gy, gr = 0.0, 0.0, 0.0
            if ref_centroid:
                gx, gy = ref_centroid
                gr = rot
            still_unmatched.append((uid, gx, gy, gr))

    return placements, still_unmatched, placed_count, log_data
