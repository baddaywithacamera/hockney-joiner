# Hockney Joiner — Keyboard & Mouse Controls

## Image Movement
| Control | Action |
|---------|--------|
| Arrow ←/→ | Rotate active image ±0.5° |
| Shift + Arrow ←/→ | Rotate active image ±0.1° (fine) |
| Arrow ↑/↓ | Nudge active image ±1px |
| R | Reset active image to auto-placed position |
| Delete | Remove image from composition (undoable) |

## Z-Order (Pile) Controls
| Control | Action |
|---------|--------|
| Z | Send active image backward one layer |
| X | Bring active image forward one layer |
| Ctrl + hover | Highlight the pile of images under cursor |
| Ctrl + click | Cycle the pile — shuffle top to bottom, exposing buried images |
| Right-click → Bring Forward | Move image up one step in pile |
| Right-click → Send Backward | Move image down one step in pile |
| Right-click → Bring to Front | Move image to top of entire stack |
| Right-click → Send to Back | Move image to bottom of entire stack |

## Navigation
| Control | Action |
|---------|--------|
| Scroll wheel | Zoom in/out |
| Trackpad pinch | Zoom in/out |
| Middle-mouse drag | Pan canvas |
| Click image | Activate image (keyboard controls apply) |
| Click empty canvas | Deactivate all |
| F | Fit all images in view |
| G | Toggle grid overlay |

## Undo / Redo
| Control | Action |
|---------|--------|
| Ctrl+Z | Undo last operation |
| Ctrl+Y | Redo last undone operation |

## Composition Chat
| Control | Action |
|---------|--------|
| Type in chat → Ask | Send question to AI about the composite |
| Clear highlights | Remove AI highlight from flagged images |

## Deal Mode (Shoot Replay)
| Control | Action |
|---------|--------|
| D | Enter Deal Mode — images hide, then appear one-by-one |
| Spacebar (1st tap) | Show next photo in lower-right corner preview with EXIF info |
| Spacebar (2nd tap) | Send photo from preview to its calculated table position |
| ESC | Exit Deal Mode — all remaining images become visible |

Deal Mode sorts images by filename (sequential camera numbering) and reveals
them one at a time. On entry, a dialog lets you type batch shooting info
(shutter, aperture, ISO) that applies to every photo in the set — useful for
cameras without EXIF. Leave fields blank to use per-file EXIF when available;
batch values override per-file EXIF. A progress indicator (e.g. "12 / 47")
is shown while dealing. Useful for building joiners photo-by-photo on camera
and for YouTube videos showing the assembly process in real time.

All z-order and movement operations are undoable via Ctrl+Z.
