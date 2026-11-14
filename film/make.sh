#!/bin/bash
# Build the film from film/film.json.
#
#   film/make.sh v2                      the no-gloves cut, with the stand-in shot 2
#   film/make.sh v3 --shot2 hand.mp4     the same film with a real shot 2 (film/README.md says what the clip must be)
#
# Only what changed is made again: a voice line whose words changed, a take that is missing.
# The cut itself renders in a drun sandbox. Result: /bulk3/runs/robotteleop/film/kaiku-nogloves-VERSION.mp4
set -euo pipefail
work=${KAIKU_FILM_WORK:-/bulk3/runs/robotteleop/film/work}
ver=v1; shot2=""
while [ $# -gt 0 ]; do
  case "$1" in
    --shot2) shot2=$2; shift 2 ;;
    *) ver=$1; shift ;;
  esac
done
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
out=$(dirname "$work")/kaiku-$(/usr/bin/python -c "import json;print(json.load(open('film/film.json'))['cut'])")-$ver.mp4

hsec exec --only ELEVENLABS_API_KEY -- python3 film/vo.py "$work"
nice -n 10 /usr/bin/python film/capture.py "$work"

if [ -n "$shot2" ] && [ "$shot2" != standin ]; then            # a real hand: fitted to the frame, sound dropped
  nice -n 10 ffmpeg -y -loglevel error -i "$shot2" -an -t 15 -r 30 \
    -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1" \
    -c:v libx264 -preset medium -crf 14 -pix_fmt yuv420p "$work/clips/shot2.mp4"
  echo real > "$work/clips/shot2.kind"
else
  cp "$work/clips/shot2-standin.mp4" "$work/clips/shot2.mp4"
  echo standin > "$work/clips/shot2.kind"
fi

nice -n 10 /usr/bin/python film/edit.py plan "$work"
(cd "$work/edit" && drun -- bash run.sh)
cp "$work/edit/out/film.mp4" "$out"
cp "$work/edit/out/film.srt" "${out%.mp4}.srt"
cp "$work/edit/out/report.json" "${out%.mp4}.report.json"
# one frame every two seconds, to look the whole film over at a glance
nice -n 10 ffmpeg -y -loglevel error -i "$out" -vf "fps=1/2,scale=480:270,tile=8x8" -frames:v 1 -q:v 3 "${out%.mp4}-contact.jpg"
echo "$out  $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out") s"
cat "${out%.mp4}.report.json"
