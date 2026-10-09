#!/usr/bin/env bash
# Point the artifact's configuration at wherever you unpacked it.
#
# The shipped configuration files use an anonymous placeholder prefix
# ("/opt/artifact"). Running this script rewrites that prefix to the absolute
# path of this checkout so the example configs, dataset roots and the
# reachability-baseline manifests resolve to the files that ship with the
# artifact.
#
# Usage:  bash configure_paths.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 - "$ROOT" <<'PY'
import pathlib, sys

root = pathlib.Path(sys.argv[1]).resolve()
placeholder = "/opt/artifact"
SKIP_SUFFIX = {".elf", ".bin", ".so", ".o", ".a", ".png", ".jpg", ".jpeg",
               ".pdf", ".gz", ".zip", ".xz", ".whl"}
SKIP_DIRS = {".git", "__pycache__"}

changed = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if any(part in SKIP_DIRS for part in path.parts):
        continue
    if path.suffix.lower() in SKIP_SUFFIX:
        continue
    if path.stat().st_size > 8_000_000:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    if placeholder not in text:
        continue
    path.write_text(text.replace(placeholder, str(root)), encoding="utf-8")
    changed.append(path.relative_to(root))

print(f"rewrote {len(changed)} file(s): {placeholder} -> {root}")
for rel in changed[:20]:
    print(f"  {rel}")
if len(changed) > 20:
    print(f"  ... and {len(changed) - 20} more")
PY

cat <<'EOF'

Next steps
----------
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r Requirements.txt
  python3 -m lsgemu.lsgemu --config configs/lsgemu_config.yaml <firmware.elf> --time 60
EOF
