"""
placement_director.py — LLM-assisted placement (Phase 2).

Two optional, cloud-powered placement modes. Both are opt-in engine choices in
the Matching Engine selector; the offline feature-matching engines are untouched.

  • direct_full()  — "AI does all the placement."
        Shows the model a numbered contact sheet (plus any reference images) and
        asks it to arrange every tile on the canvas. The model returns a
        normalized centre (cx, cy in 0..1) + relative scale per tile, which we
        map into the same thumbnail-pixel coordinate space the canvas already
        uses. No feature matching involved.

  • finetune()     — "AI fine-tunes the local placement."
        Runs the normal local engine first, renders the resulting arrangement
        (numbered), and asks the model to nudge only the tiles that look
        misaligned. Tiles the model doesn't mention keep their local position.

Design:
  * Provider is injected (a VisionProvider), so all the coordinate math is pure
    and unit-testable without network or Qt.
  * Coordinates are thumbnail-pixel units (tile ≈ store.thumb_long_edge), matching
    placement.py so results drop straight onto the canvas / into export.
  * The model only ever decides *layout*. It never sees or alters the originals.
"""

from __future__ import annotations

import logging
import math

from hockney.core.models import ImagePlacement

log = logging.getLogger(__name__)

# Spacing factor: how far apart tile centres spread across the canvas relative to
# tile size. >1 leaves room so default-scale tiles overlap modestly (Hockney look).
SPREAD = 1.3
DEFAULT_THUMB_EDGE = 300


# ── Small numeric helpers (pure) ────────────────────────────────────────────────

def _as_int(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _clamp01(v):
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    return min(1.0, max(0.0, f))


def _clampf(v, lo, hi, default):
    try:
        f = float(v)
    except (ValueError, TypeError):
        return default
    return min(hi, max(lo, f))


def _tile_size(record, thumb_edge: float, scale: float = 1.0) -> tuple[float, float]:
    """Rendered tile size in placement (thumbnail-pixel) units, preserving aspect."""
    w = getattr(record, "width", 0) or 0
    h = getattr(record, "height", 0) or 0
    if w <= 0 or h <= 0:
        return thumb_edge * scale, thumb_edge * scale
    if w >= h:
        tw, th = thumb_edge, thumb_edge * h / w
    else:
        tw, th = thumb_edge * w / h, thumb_edge
    return tw * scale, th * scale


def _p(progress_cb, pct: int):
    if progress_cb:
        try:
            progress_cb(int(pct))
        except Exception:
            pass


def _cancelled(cancel_cb) -> bool:
    try:
        return bool(cancel_cb()) if cancel_cb else False
    except Exception:
        return False


# ── Contact sheet / arrangement rendering ───────────────────────────────────────

def build_contact_sheet(records, store):
    """Numbered contact sheet where tile (i+1) == records[i]."""
    from hockney.core.export import render_contact_sheet
    placeholders = [
        ImagePlacement(image_id=r.id, z_order=i) for i, r in enumerate(records)
    ]
    return render_contact_sheet(placeholders, store)


def _render_arrangement(ordered, rec_by_id, store, thumb_edge, frame, target=1400.0):
    """
    Render the current placement arrangement as one image with each tile numbered
    in reading/z order. Self-contained so we know exactly how canvas coords map.
    """
    from PIL import Image, ImageDraw

    minx, miny, maxx, maxy = frame
    span_w = max(1.0, maxx - minx)
    span_h = max(1.0, maxy - miny)
    s = min(1.0, target / max(span_w, span_h))
    cw = max(1, int(span_w * s))
    ch = max(1, int(span_h * s))

    canvas = Image.new("RGB", (cw, ch), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for idx, p in enumerate(ordered, start=1):
        rec = rec_by_id.get(p.image_id)
        tw, th = _tile_size(rec, thumb_edge, 1.0) if rec else (thumb_edge, thumb_edge)
        x = (p.x - minx) * s
        y = (p.y - miny) * s
        tile_w = max(1, int(tw * s))
        tile_h = max(1, int(th * s))
        arr = store.get_thumbnail(p.image_id) if store else None
        if arr is not None:
            try:
                tile = Image.fromarray(arr).resize((tile_w, tile_h))
                canvas.paste(tile, (int(x), int(y)))
            except Exception as e:
                log.debug("arrangement paste failed for %s: %s", idx, e)
        draw.rectangle([x, y, x + 26, y + 16], fill=(0, 0, 0))
        draw.text((x + 3, y + 2), str(idx), fill=(255, 220, 0))

    return canvas


# ── Coordinate mapping (pure) ───────────────────────────────────────────────────

def _canvas_size(n: int, thumb_edge: float) -> tuple[float, float]:
    cols = max(1, math.ceil(math.sqrt(max(n, 1))))
    rows = max(1, math.ceil(max(n, 1) / cols))
    return cols * thumb_edge * SPREAD, rows * thumb_edge * SPREAD


def _layout_to_placements(layout, records, thumb_edge):
    """
    Map an LLM layout {"tiles":[{"i","cx","cy","scale"?,"rotation"?}]} into
    ImagePlacements (thumbnail-pixel space). Returns (placements_dict, placed_ids).
    """
    tiles = layout.get("tiles", []) if isinstance(layout, dict) else []
    n = len(records)
    canvas_w, canvas_h = _canvas_size(n, thumb_edge)

    placements: dict[str, ImagePlacement] = {}
    placed: set[str] = set()

    for t in tiles:
        if not isinstance(t, dict):
            continue
        i = _as_int(t.get("i", t.get("index")))
        if i is None or not (1 <= i <= n):
            continue
        rec = records[i - 1]
        if rec.id in placed:
            continue
        cx = _clamp01(t.get("cx"))
        cy = _clamp01(t.get("cy"))
        if cx is None or cy is None:
            continue
        scale = _clampf(t.get("scale", 1.0), 0.3, 3.0, 1.0)
        rot = _clampf(t.get("rotation", 0.0), -45.0, 45.0, 0.0)

        tw, th = _tile_size(rec, thumb_edge, scale)
        x = cx * canvas_w - tw / 2.0
        y = cy * canvas_h - th / 2.0
        z = len(placed)
        placements[rec.id] = ImagePlacement(
            image_id=rec.id, x=x, y=y, rotation=rot, z_order=z,
            auto_x=x, auto_y=y, auto_rotation=rot,
        )
        placed.add(rec.id)

    return placements, placed


def _fill_missing(placements, records, placed, thumb_edge) -> int:
    """Grid out any tiles the model didn't return, below the arrangement."""
    missing = [r for r in records if r.id not in placed]
    if not missing:
        return 0
    ys = [p.y for p in placements.values()]
    y0 = (max(ys) if ys else 0.0) + thumb_edge * 2.0
    base_z = len(placements)
    for i, rec in enumerate(missing):
        x = float(i * (thumb_edge + 20))
        y = y0
        placements[rec.id] = ImagePlacement(
            image_id=rec.id, x=x, y=y, rotation=0.0, z_order=base_z + i,
            auto_x=x, auto_y=y, auto_rotation=0.0,
        )
    return len(missing)


def _frame_of(ordered, rec_by_id, thumb_edge):
    """Bounding box (minx, miny, maxx, maxy) over the placed tiles' extents."""
    xs, ys, xe, ye = [], [], [], []
    for p in ordered:
        rec = rec_by_id.get(p.image_id)
        tw, th = _tile_size(rec, thumb_edge, 1.0) if rec else (thumb_edge, thumb_edge)
        xs.append(p.x)
        ys.append(p.y)
        xe.append(p.x + tw)
        ye.append(p.y + th)
    if not xs:
        return 0.0, 0.0, float(thumb_edge), float(thumb_edge)
    return min(xs), min(ys), max(xe), max(ye)


def _current_norm(ordered, rec_by_id, thumb_edge, frame):
    """Normalized centres of the current arrangement, for the fine-tune prompt."""
    minx, miny, maxx, maxy = frame
    span_w = max(1.0, maxx - minx)
    span_h = max(1.0, maxy - miny)
    out = []
    for idx, p in enumerate(ordered, start=1):
        rec = rec_by_id.get(p.image_id)
        tw, th = _tile_size(rec, thumb_edge, 1.0) if rec else (thumb_edge, thumb_edge)
        cx = (p.x + tw / 2.0 - minx) / span_w
        cy = (p.y + th / 2.0 - miny) / span_h
        out.append({"i": idx, "cx": round(cx, 3), "cy": round(cy, 3)})
    return out


def _apply_corrections(ordered, rec_by_id, thumb_edge, frame, layout) -> int:
    """Mutate placements in place from corrected normalized centres. Returns count moved."""
    minx, miny, maxx, maxy = frame
    span_w = max(1.0, maxx - minx)
    span_h = max(1.0, maxy - miny)
    tiles = layout.get("tiles", []) if isinstance(layout, dict) else []
    moved = 0
    for t in tiles:
        if not isinstance(t, dict):
            continue
        i = _as_int(t.get("i", t.get("index")))
        if i is None or not (1 <= i <= len(ordered)):
            continue
        cx = _clamp01(t.get("cx"))
        cy = _clamp01(t.get("cy"))
        if cx is None or cy is None:
            continue
        p = ordered[i - 1]
        rec = rec_by_id.get(p.image_id)
        tw, th = _tile_size(rec, thumb_edge, 1.0) if rec else (thumb_edge, thumb_edge)
        nx = minx + cx * span_w - tw / 2.0
        ny = miny + cy * span_h - th / 2.0
        if abs(nx - p.x) > 0.5 or abs(ny - p.y) > 0.5:
            p.x, p.y = nx, ny
            p.auto_x, p.auto_y = nx, ny
            moved += 1
    return moved


# ── Prompts ─────────────────────────────────────────────────────────────────────

def _full_prompt(n: int, has_refs: bool) -> str:
    ref_line = (
        "After the contact sheet you are given one or more wide REFERENCE photos "
        "of the whole scene — use them to decide where each tile belongs.\n"
        if has_refs else ""
    )
    return (
        "You are assembling a David Hockney style 'joiner': a single scene rebuilt "
        f"from {n} overlapping photo fragments. The FIRST image is a numbered "
        f"contact sheet of all {n} tiles (labelled 1..{n} in the corner).\n"
        + ref_line +
        "Decide where every tile sits on the final canvas. Use a normalized "
        "coordinate system where (0,0) is the top-left corner and (1,1) is the "
        "bottom-right. Tiles showing the same part of the scene should sit close "
        "and overlap slightly; keep the overall layout true to the scene, not a "
        "tidy grid.\n\n"
        "Return JSON ONLY (no markdown, no prose) in exactly this form:\n"
        '{"tiles":[{"i":1,"cx":0.0,"cy":0.0,"scale":1.0}, ...]}\n'
        "  • i      = tile number (1-based)\n"
        "  • cx,cy  = normalized centre of that tile (0..1)\n"
        "  • scale  = relative size, 1.0 default (0.3..3.0)\n"
        f"Include every tile from 1 to {n} exactly once."
    )


def _finetune_prompt(n: int, current, has_refs: bool) -> str:
    import json as _json
    ref_line = (
        "Wide REFERENCE photo(s) of the scene follow the arrangement image.\n"
        if has_refs else ""
    )
    return (
        "The image is the CURRENT assembled Hockney joiner; each tile is numbered. "
        "It was placed automatically by feature matching and some tiles may be "
        "misaligned, overlapping wrongly, or out of place.\n"
        + ref_line +
        "Current normalized tile centres (0..1, top-left origin):\n"
        f"{_json.dumps(current)}\n\n"
        "Suggest corrected normalized centres ONLY for tiles that should move to "
        "better align the scene. Preserve the loose, slightly-overlapping Hockney "
        "look — do not snap to a grid. Leave good tiles alone.\n\n"
        "Return JSON ONLY in this form (omit tiles that are already fine):\n"
        '{"tiles":[{"i":<tile number>,"cx":<0..1>,"cy":<0..1>}, ...]}'
    )


# ── Orchestrators (need a provider; return a PlacementResult) ────────────────────

def direct_full(records, store, provider, progress_cb=None, cancel_cb=None, refs=None):
    """LLM decides the entire layout from a contact sheet (+ optional references)."""
    from hockney.core.placement import PlacementResult

    n = len(records)
    if n == 0:
        return PlacementResult([], False, 0, "No images to place.")

    thumb_edge = getattr(store, "thumb_long_edge", DEFAULT_THUMB_EDGE)
    _p(progress_cb, 5)
    sheet = build_contact_sheet(records, store)
    if _cancelled(cancel_cb):
        return PlacementResult([], False, 0, "Cancelled.")

    images = [sheet] + list(refs or [])
    prompt = _full_prompt(n, bool(refs))
    _p(progress_cb, 15)
    layout = provider.query_json(prompt, images, max_tokens=min(8000, 300 + n * 45))
    _p(progress_cb, 80)

    placements, placed = _layout_to_placements(layout, records, thumb_edge)
    fallback = _fill_missing(placements, records, placed, thumb_edge)
    _p(progress_cb, 100)

    msg = f"AI Director placed {len(placed)}/{n} tiles."
    if fallback:
        msg += f" {fallback} weren't returned by the model — gridded below for you to move."
    return PlacementResult(list(placements.values()), False, fallback, msg)


def finetune(records, store, provider, base, progress_cb=None, cancel_cb=None, refs=None):
    """Run on top of a local PlacementResult: LLM nudges misaligned tiles."""
    from hockney.core.placement import PlacementResult

    base_list = list(base.placements)
    if not base_list:
        return base

    thumb_edge = getattr(store, "thumb_long_edge", DEFAULT_THUMB_EDGE)
    rec_by_id = {r.id: r for r in records}
    ordered = sorted(base_list, key=lambda p: p.z_order)

    _p(progress_cb, 5)
    frame = _frame_of(ordered, rec_by_id, thumb_edge)
    sheet = _render_arrangement(ordered, rec_by_id, store, thumb_edge, frame)
    if _cancelled(cancel_cb):
        return base

    current = _current_norm(ordered, rec_by_id, thumb_edge, frame)
    images = [sheet] + list(refs or [])
    prompt = _finetune_prompt(len(ordered), current, bool(refs))
    _p(progress_cb, 20)
    layout = provider.query_json(prompt, images, max_tokens=min(8000, 300 + len(ordered) * 30))
    _p(progress_cb, 85)

    moved = _apply_corrections(ordered, rec_by_id, thumb_edge, frame, layout)
    _p(progress_cb, 100)

    msg = (
        f"AI fine-tune adjusted {moved} tile(s) on top of the "
        f"{'reference' if refs else 'local'} placement."
        if moved else
        "AI fine-tune reviewed the placement and left it unchanged."
    )
    return PlacementResult(base_list, base.used_lightglue, base.fallback_count, msg)
