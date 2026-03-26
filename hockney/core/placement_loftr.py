"""
placement_loftr.py — LoFTR dense matching engine.

LoFTR (Detector-Free Local Feature Matching with Transformers) is a dense
matcher that doesn't rely on keypoint detection.  Instead it matches every
patch of the image against every patch of the reference using a transformer.

This makes it dramatically better than SIFT or DISK on:
  - Low-quality / noisy images (Kodak Charmer)
  - Textureless surfaces (smooth metal, plastic)
  - Blurry or soft-focus images
  - Repetitive textures

Uses kornia.feature.LoFTR with the "outdoor" pretrained weights.
Input: greyscale images as 1x1xHxW float tensors, normalized [0,1].
Output: matched keypoint pairs + confidence scores.

Like SIFT, this uses homography-based center mapping for placement:
the detail image's center point is projected through the homography
to find where it lands on the reference.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

from hockney.core.placement import _snap_orientation

log = logging.getLogger(__name__)

DETAIL_SCALES = [1.0, 0.5, 0.33, 0.25]   # fewer scales — LoFTR is slower
RANSAC_REPROJ = 5.0
MIN_GOOD_MATCHES = 8
MIN_GOOD_RELAXED = 5
LOFTR_LONG_EDGE = 840    # LoFTR works best at moderate resolution; 840 balances quality/speed


def _to_loftr_tensor(img_rgb: np.ndarray, device, long_edge: int = LOFTR_LONG_EDGE):
    """
    Convert RGB numpy array to greyscale 1x1xHxW float tensor for LoFTR.
    Resizes so the long edge is at most `long_edge` pixels.
    Returns (tensor, scale_x, scale_y) where scale factors map back to original coords.
    """
    import torch

    h, w = img_rgb.shape[:2]
    scale = min(long_edge / max(h, w), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # LoFTR wants dimensions divisible by 8
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    if new_w < 64 or new_h < 64:
        new_w = max(new_w, 64)
        new_h = max(new_h, 64)

    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    t = torch.from_numpy(grey.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0)   # 1x1xHxW
    return t.to(device), w / new_w, h / new_h


def place_with_loftr(
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
    LoFTR-based reference placement.

    Returns:
      placements: dict[image_id → ImagePlacement]
      odds_and_ends: list of unplaced (id, gx, gy, gr)
      placed_count: int
      log_data: dict of per-image match stats
    """
    try:
        import torch
        from kornia.feature import LoFTR
    except ImportError as e:
        log.warning("LoFTR import failed: %s", e)
        return {}, [(r.id, 0.0, 0.0, 0.0) for r in records], 0, {}

    from PIL import Image as PILImage
    from hockney.core.models import ImagePlacement
    from hockney.core.placement_sift import clahe_preprocess, light_denoise

    PREVIEW_LONG_EDGE = 1500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("LoFTR placement on device: %s", device)

    # Load LoFTR matcher
    matcher = LoFTR(pretrained="outdoor").eval().to(device)

    refs = config.references
    n_refs = len(refs)
    n_detail = len(records)
    total_work = n_refs + n_detail + n_detail * n_refs
    done = 0
    log_data = {}

    # ── 1. Prepare reference images ───────────────────────────────────────
    ref_tensors = {}    # slot → (tensor, scale_x, scale_y)
    ref_sizes = {}      # slot → (w, h) at preview resolution

    for ref in refs:
        if cancel_check and cancel_check():
            return {}, [], 0, log_data
        try:
            pil = PILImage.open(ref.source_path).convert("RGB")
            w, h = pil.size
            scale = PREVIEW_LONG_EDGE / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
            ref_sizes[ref.slot] = (new_w, new_h)

            arr = np.array(pil)
            tensor, sx, sy = _to_loftr_tensor(arr, device)
            ref_tensors[ref.slot] = (tensor, sx, sy)
            log.info("LoFTR ref %s: tensor shape %s, scale_back=(%.2f, %.2f)",
                     ref.slot, list(tensor.shape), sx, sy)
        except Exception as e:
            log.warning("LoFTR ref extraction failed %s: %s", ref.slot, e)

        done += 1
        if progress_cb:
            progress_cb(int(done / total_work * 100))

    if not ref_tensors:
        return {}, [(r.id, 0.0, 0.0, 0.0) for r in records], 0, log_data

    # ── 2. Prepare detail images at multiple scales ───────────────────────
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

                tensor, sx, sy = _to_loftr_tensor(resized, device)
                scales_data.append((ds, tensor, sx, sy, (new_w, new_h)))

            detail_multiscale[record.id] = scales_data
        except Exception as e:
            log.warning("LoFTR detail prep failed %s: %s", record.id[:6], e)
        done += 1
        if progress_cb:
            progress_cb(int(done / total_work * 100))

    # ── 3. Match each detail against references ───────────────────────────
    import torch

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
        best_ref_point = None
        best_rot = 0.0
        best_scale = 1.0

        for slot, (ref_t, ref_sx, ref_sy) in ref_tensors.items():
            for ds, det_t, det_sx, det_sy, (dw, dh) in detail_multiscale[record.id]:
                try:
                    with torch.no_grad():
                        input_dict = {"image0": ref_t, "image1": det_t}
                        result = matcher(input_dict)

                    kp_ref = result["keypoints0"].cpu().numpy()  # Nx2
                    kp_det = result["keypoints1"].cpu().numpy()  # Nx2
                    confidence = result["confidence"].cpu().numpy()  # N

                    # Filter by confidence
                    conf_mask = confidence > 0.5
                    kp_ref = kp_ref[conf_mask]
                    kp_det = kp_det[conf_mask]

                    if len(kp_ref) < 4:
                        continue

                    # Scale keypoints back to preview resolution
                    kp_ref_scaled = kp_ref.copy()
                    kp_ref_scaled[:, 0] *= ref_sx
                    kp_ref_scaled[:, 1] *= ref_sy
                    kp_det_scaled = kp_det.copy()
                    kp_det_scaled[:, 0] *= det_sx
                    kp_det_scaled[:, 1] *= det_sy

                    # RANSAC
                    M, mask = cv2.estimateAffinePartial2D(
                        kp_ref_scaled.reshape(-1, 1, 2).astype(np.float32),
                        kp_det_scaled.reshape(-1, 1, 2).astype(np.float32),
                        method=cv2.RANSAC,
                        ransacReprojThreshold=RANSAC_REPROJ,
                    )
                    if M is None or mask is None:
                        continue

                    inlier_idx = mask.ravel().astype(bool)
                    n_inliers = int(inlier_idx.sum())

                    if n_inliers > best_inliers and n_inliers >= min_matches:
                        # Homography: map detail center → reference
                        inlier_det = kp_det_scaled[inlier_idx]
                        inlier_ref = kp_ref_scaled[inlier_idx]

                        H, _ = cv2.findHomography(
                            inlier_det.reshape(-1, 1, 2).astype(np.float32),
                            inlier_ref.reshape(-1, 1, 2).astype(np.float32),
                            cv2.RANSAC, RANSAC_REPROJ,
                        )
                        if H is not None:
                            det_center = np.array([[[dw / 2.0, dh / 2.0]]],
                                                   dtype=np.float32)
                            ref_point = cv2.perspectiveTransform(det_center, H)
                            cx = float(ref_point[0, 0, 0])
                            cy = float(ref_point[0, 0, 1])
                        else:
                            cx = float(inlier_ref[:, 0].mean())
                            cy = float(inlier_ref[:, 1].mean())

                        rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))

                        best_slot = slot
                        best_inliers = n_inliers
                        best_ref_point = (cx, cy)
                        best_rot = rot
                        best_scale = ds

                except Exception as e:
                    log.debug("LoFTR match %s↔%s@%.0f%% failed: %s",
                              slot, record.id[:6], ds * 100, e)

            done += 1
            if progress_cb:
                progress_cb(int(done / total_work * 100))

        # Log
        log_data[record.id] = {
            "engine": "loftr",
            "inliers": best_inliers,
            "scale": best_scale,
            "slot": best_slot or "none",
            "placed": best_slot is not None and best_ref_point is not None,
        }

        if best_slot and best_ref_point:
            cx, cy = best_ref_point

            if config.project_type == "perspective":
                offsets = {
                    "standing": (0, 0), "left": (-1, 0), "right": (1, 0),
                    "down_low": (0, 1), "up_high": (0, -1),
                }
                ox, oy = offsets.get(best_slot, (0, 0))
                rw, rh = ref_sizes.get(best_slot, (PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE))
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
                x=final_x, y=final_y, rotation=_snap_orientation(best_rot),
                z_order=placed_count,
                auto_x=final_x, auto_y=final_y, auto_rotation=_snap_orientation(best_rot),
            )
            placements[record.id] = p
            placed_count += 1
            log.info("LoFTR placed %s in %s (%d inliers, scale=%.0f%%)",
                     record.id[:6], best_slot, best_inliers, best_scale * 100)
        else:
            odds_and_ends.append((record.id, best_inliers,
                                  best_ref_point, best_rot))

    # ── 3b. Second pass at relaxed threshold ──────────────────────────────
    still_unmatched = []
    for uid, n_inliers, ref_point, rot in odds_and_ends:
        if n_inliers >= min_matches_relaxed and ref_point:
            cx, cy = ref_point
            rec = store.get_record(uid)
            if rec:
                aspect = rec.width / max(rec.height, 1)
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
            if uid in log_data:
                log_data[uid]["placed"] = True
                log_data[uid]["pass"] = "relaxed"
        else:
            gx, gy, gr = 0.0, 0.0, 0.0
            if ref_point:
                gx, gy = ref_point
                gr = rot
            still_unmatched.append((uid, gx, gy, gr))

    return placements, still_unmatched, placed_count, log_data
