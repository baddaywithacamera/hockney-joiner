# Building Hockney Joiner

## Development (run from source)

```bash
pip install -r requirements.txt
python -m hockney
```

## Standalone build (PyInstaller)

```bash
pip install pyinstaller
pyinstaller hockney-joiner.spec
```

Output lands in `dist/hockney-joiner/`. The models directory inside it
is empty — LightGlue and Moondream are downloaded on first run.

### Windows notes
- Build on Windows to get a Windows binary (PyInstaller is not cross-compiler)
- The `console=False` flag suppresses the terminal window
- To add an app icon, put `icon.ico` in the project root and update the spec

### Linux notes

**System dependencies** (needed for PyQt6 and OpenCV):

```bash
# Ubuntu / Debian
sudo apt install python3-dev python3-venv libxcb-cursor0 libgl1 libegl1

# Fedora
sudo dnf install python3-devel mesa-libGL mesa-libEGL xcb-util-cursor

# Arch
sudo pacman -S python mesa xcb-util-cursor
```

**Running from source on Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m hockney
```

**Building a standalone binary:**

- Build on Linux for a Linux binary (PyInstaller does not cross-compile)
- May need `--hidden-import` additions for some distros
- `upx` compression is optional — remove from the spec if UPX is not installed
- RAW camera support: `pip install rawpy` (optional, needs libraw-dev on Debian/Ubuntu)

## Adding an app icon

1. Create `assets/icon.ico` (Windows) and `assets/icon.png` (Linux)
2. In the spec, set `icon='assets/icon.ico'`
3. Rebuild

## Model notes

Models are NOT bundled in the installer. They are downloaded on first use:

| Model | Size | Trigger |
|-------|------|---------|
| LightGlue (DISK) | ~55 MB | Help → Download LightGlue |
| Moondream2 | ~1.7 GB | Help → Download Moondream |

Both run fully offline after download.
