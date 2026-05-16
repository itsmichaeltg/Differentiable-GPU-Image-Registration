#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 FRAME_DIR OUTPUT.mp4 [fps]" >&2
  exit 2
fi

frame_dir="$1"
output="$2"
fps="${3:-12}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to encode video frames" >&2
  exit 1
fi

ffmpeg -y -framerate "$fps" -i "$frame_dir/frame_%05d.pgm" \
  -pix_fmt yuv420p "$output"
