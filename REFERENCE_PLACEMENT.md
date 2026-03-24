# Reference-Based Placement — Design Spec

## Problem

The current placement engine compares every detail photo against every other
detail photo (N² pairwise matching).  With 114 images that's ~6,400 pairs,
most of which don't overlap.  Errors accumulate through BFS chaining, and the
result looks random.

## Solution: The Puzzle Box Lid

The user provides **reference images** — wide shots that show the whole scene.
The engine matches each detail photo against the references instead of against
every other detail photo.  This gives each image an **absolute** position
(no error accumulation) and reduces matching from N² to N × R, where R is the
number of references (typically 1–6).

---

## New Project Flow

### Step 1 — New Project

Triggered via **File → New Project…** or on first launch.

| Field        | Type       | Notes                           |
|--------------|------------|---------------------------------|
| Project Name | text       | e.g. "Downtown Portland Spring" |
| Project Type | choice     | Perspective / Time of Day / Seasonal |
| Subject Type | choice     | Landscape / Skylife / Urban / Indoor / People |

Subject Type helps the matching engine weight features appropriately:
- **Landscape** — large features, horizon lines, natural textures
- **Skylife** — open sky, clouds, atmospheric gradients
- **Urban** — geometric edges, signage, repeating structures
- **Indoor** — smaller scale, repetitive textures, controlled lighting
- **People** — moving subjects, face-aware matching weight reduction

### Step 2 — Load Reference Images

Based on **Project Type**, the UI shows reference slots.  Each slot is a
labelled drop zone.  The user loads one image per slot.  Empty slots are
fine — use what you have.

#### Perspective (standard joiner)

Default slots (always visible):

| Slot     | Description                          |
|----------|--------------------------------------|
| Standing | Eye-level reference — the main view  |
| Left     | Shifted left of centre               |
| Right    | Shifted right of centre              |

Advanced slots (hidden behind "Show Advanced"):

| Slot           | Description                        |
|----------------|------------------------------------|
| Down Low       | Low angle / ground level           |
| Up High        | Elevated vantage point             |

#### Time of Day

Single perspective, four time slots:

| Slot      | Description                          |
|-----------|--------------------------------------|
| Morning   | Early light, long shadows            |
| Afternoon | Midday / flat light                  |
| Evening   | Golden hour, warm tones              |
| Night     | Dark, artificial light               |

Only one perspective (frontal/standing).

#### Seasonal

Single perspective, four season slots:

| Slot   | Description                            |
|--------|----------------------------------------|
| Spring | New growth, blossoms                   |
| Summer | Full foliage, bright light             |
| Fall   | Colour change, warm palette            |
| Winter | Bare branches, snow, cool tones        |

Only one perspective (frontal/standing).

---

## Placement Engine (Revised)

### Phase 1 — Feature Extraction

Extract DISK features from:
1. All reference images (at full PREVIEW_LONG_EDGE = 1500px for accuracy)
2. All detail photos (at THUMB_LONG_EDGE = 300px as before)

References get higher-resolution extraction because they're the map.

### Phase 2 — Reference Matching

For each detail photo:
1. Match against **every** reference image via LightGlue
2. Pick the reference with the **highest inlier count** (best match)
3. Compute the similarity transform (translation + rotation) relative
   to that reference
4. Store: which reference it matched, and the transform

If a detail photo doesn't meet MIN_MATCHES against any reference, it goes
to the unplaced pile.

### Phase 3 — Position Mapping

Each reference occupies a known position in the canvas coordinate system.
The first reference (Standing / Morning / Spring) anchors at the origin.
Additional references are offset based on their slot:

- **Perspective** — Left/Right references offset horizontally by a fraction
  of the reference width.  Down Low / Up High offset vertically.
- **Time of Day** — References tile left-to-right: Morning | Afternoon |
  Evening | Night.
- **Seasonal** — References tile left-to-right: Spring | Summer | Fall |
  Winter.

Each detail photo's final canvas position = reference anchor position +
matched transform offset.

### Phase 4 — Odds & Ends Tray

Detail photos that match no reference go to the **Odds & Ends** tray — a
separate area below the main composition.  This replaces the old unnamed
grid fallback.

Behaviour:
- Odds & Ends images are arranged in a labelled row beneath the composition
- Hovering over an image in the Odds & Ends tray highlights its **best-guess
  position** on the tabletop (ghost outline at reduced opacity showing where
  the engine thinks it would go, even though the match wasn't strong enough
  to place automatically)
- The user can drag an image from Odds & Ends onto the table manually, or
  leave it in the tray
- The ghost position helps the user decide: "close enough, I'll place it
  there" or "nope, that's a dud"

---

## Subject Type Tuning

The Subject Type adjusts matching parameters:

| Subject   | MAX_KEYPOINTS | MIN_MATCHES | Notes                              |
|-----------|---------------|-------------|------------------------------------|
| Landscape | 1024          | 12          | Default — good balance             |
| Skylife   | 512           | 8           | Fewer features in sky regions      |
| Urban     | 2048          | 15          | Dense features, need more to filter|
| Indoor    | 1024          | 10          | Moderate, watch for repetition     |
| People    | 1024          | 12          | Standard, but down-weight faces    |

These are starting points — will need tuning with real images.

---

## Project File Changes (v2)

```json
{
  "version": 2,
  "project_name": "Downtown Portland Spring",
  "project_type": "perspective",
  "subject_type": "urban",
  "references": [
    {
      "slot": "standing",
      "source_path": "/path/to/ref_standing.jpg",
      "id": "abc123"
    },
    {
      "slot": "left",
      "source_path": "/path/to/ref_left.jpg",
      "id": "def456"
    }
  ],
  "images": [ ... ],
  "placements": [ ... ],
  "removed_ids": [ ... ],
  "processing": { ... }
}
```

v1 projects (no references) continue to work — they use the old pairwise
engine as fallback.

---

## UI Changes

### New Project Dialog

Three-step wizard or single dialog with sections:
1. Project Name (text field)
2. Project Type (radio buttons: Perspective / Time of Day / Seasonal)
3. Subject Type (dropdown: Landscape / Skylife / Urban / Indoor / People)

### Reference Panel

New section in the **Tools** sidebar dock, above the image list:

- Shows labelled slots based on project type
- Each slot: thumbnail preview + "Load…" button + "Clear" button
- "Show Advanced" toggle for Down Low / Up High (Perspective only)
- Empty slots show a dashed border placeholder

### Toolbar

- "Auto-Place" button now reads "Auto-Place [References]" when references
  are loaded, "Auto-Place [Grid]" when no references and no LightGlue

---

## Implementation Order

1. **Data model** — Add `ProjectConfig` to `models.py` with project_name,
   project_type, subject_type, references list
2. **Project file v2** — Update `project.py` to save/load references and
   config; maintain v1 backward compat
3. **New Project dialog** — New `ui/new_project_dialog.py`
4. **Reference panel in sidebar** — New section in `ui/sidebar.py`
5. **Revised placement engine** — New `_place_with_references()` method in
   `placement.py` that matches details against references
6. **Subject type tuning** — Wire MAX_KEYPOINTS / MIN_MATCHES to subject type
7. **Reference position mapping** — Compute canvas anchors from slot positions

Steps 1–4 are UI/data plumbing.  Step 5 is the core algorithm change.

---

## Reference Backdrop (Puzzle Box Lid on Canvas)

The reference images render **on the canvas as a faint backdrop** so the user
can see the "puzzle box lid" while placing detail photos on top.

- **Toggle**: View → Show Reference Backdrop (on/off, default on)
- **Opacity slider**: adjustable from 5% to 50% (default ~20%)
- References are drawn at z-order below all detail photos, tiled according
  to their slot positions (same layout as Phase 3 position mapping)
- When toggled off, the canvas shows only detail photos on the dark background
- The backdrop is purely visual — it doesn't affect export or placement math

This gives the user spatial context while arranging, especially useful when
manually placing images from the Odds & Ends tray.

---

## Resolved Design Decisions

1. **Reference backdrop**: Yes — shown on canvas with adjustable opacity,
   toggle on/off via View menu.
2. **Time/Season slot assignment**: Best match wins. The engine matches each
   detail photo against all reference slots and assigns it to whichever
   reference has the highest inlier count.  No manual tagging required.
3. **Reference resolution**: 1500px (PREVIEW_LONG_EDGE) is sufficient for
   matching.  Full-res originals are only touched at export.
4. **Unplaced photos**: Named "Odds & Ends" tray.  Hovering over an
   unplaced image shows its best-guess ghost position on the tabletop.

---

## Resolved Design Decisions (continued)

5. **Odds & Ends interaction**: Draggable directly onto the table. No
   right-click required — just grab and place.
6. **Reference backdrop in export**: Always excluded. The backdrop is a
   working aid only, never part of the final output.
7. **Ghost position preview**: Solid outline (not a translucent image copy).
   Shows the bounding rectangle at the guessed position and rotation so
   the user can see where the engine thinks it goes without visual clutter.
