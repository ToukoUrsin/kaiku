#!/bin/bash
# Build the film from film/film.json.
#
#   film/make.sh v4                          the no-gloves film: the cold open, then film of a real hand played into the page as its camera
#   film/make.sh v4 --no-open                the body alone, without the cold open and the end card (kaiku-nogloves-v4-noopen.mp4)
#   film/make.sh v5 --shot2 glove.mp4        the same film with a phone clip of a real gloved hand as its first shot, with the gloves words
#       [--from 3]                           start 3 s into the clip (default 0)
#       [--turn 90|180|270]                  turn the clip clockwise, if it plays on its side
#       [--hand 0.5,0.45,0.5]                where the hand is in the (turned) clip when it starts: centre x, centre y, height,
#                                            as shares of the picture. It is put where the robot's hand is in the open's last frame.
#
# The cold open and the end card are read from the folder film.json names (open.dir) every time this runs, at whatever
# length they have then. The build stops if the whole film would pass 2:00.
# Only what changed is made again: a voice line whose words changed, a take that is missing.
# The cut itself renders in a drun sandbox.
# Result: /bulk3/runs/robotteleop/film/kaiku-nogloves-VERSION.mp4, or kaiku-gloves-VERSION.mp4 with --shot2,
# with captions (.srt), a contact sheet and a report beside it. film/README.md says what --shot2 changes.
set -euo pipefail
work=${KAIKU_FILM_WORK:-/bulk3/runs/robotteleop/film/v4/work}
films=${KAIKU_FILM_OUT:-/bulk3/runs/robotteleop/film}
ver=v4; shot2=""; from=0; turn=0; hand=0.5,0.45,0.5; open=""; tag=""
while [ $# -gt 0 ]; do
  case "$1" in
    --shot2) shot2=$2; shift 2 ;;
    --from) from=$2; shift 2 ;;
    --turn) turn=$2; shift 2 ;;
    --hand) hand=$2; shift 2 ;;
    --no-open) open=--no-open; tag=-noopen; shift ;;
    *) ver=$1; shift ;;
  esac
done
root=$(cd "$(dirname "$0")/.." && pwd)
[ -z "$shot2" ] || [ "$shot2" = standin ] || shot2=$(realpath "$shot2")
cd "$root"
cut=$(/usr/bin/python -c "import json;print(json.load(open('film/film.json'))['cut'])")
mkdir -p "$work/clips"

if [ -n "$shot2" ] && [ "$shot2" != standin ]; then
  # A phone clip of a real gloved hand, any size, upright or sideways: the whole picture is kept (edit.py puts it over
  # a blurred copy of itself where it does not fill 16:9). Sound is dropped. An HDR clip is brought down to ordinary video.
  [ -f "$shot2" ] || { echo "STOP: no such clip: $shot2"; exit 1; }
  cut=gloves
  len=$(/usr/bin/python -c "import json;print(json.load(open('film/film.json'))['shots'][0]['dur_real'])")
  have=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$shot2")
  /usr/bin/python -c "import sys;sys.exit(0 if float('$have') - float('$from') >= float('$len') - 0.05 else 1)" \
    || { echo "STOP: shot 1 uses $len s of the clip from $from s, and the clip is $have s long"; exit 1; }
  case "$turn" in
    0) rot="" ;; 90) rot="transpose=1," ;; 180) rot="hflip,vflip," ;; 270) rot="transpose=2," ;;
    *) echo "STOP: --turn is 90, 180 or 270"; exit 1 ;;
  esac
  hdr=""
  case "$(ffprobe -v error -select_streams v:0 -show_entries stream=color_transfer -of csv=p=0 "$shot2")" in
    arib-std-b67|smpte2084) hdr="zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv," ;;
  esac
  /usr/bin/python -c "import sys;v=[float(x) for x in '$hand'.split(',')];sys.exit(0 if len(v)==3 and all(0<x<=1 for x in v) else 1)" \
    || { echo "STOP: --hand is three shares of the picture, like 0.5,0.45,0.5 (centre x, centre y, height)"; exit 1; }
  nice -n 10 ffmpeg -y -loglevel error -ss "$from" -i "$shot2" -an -t "$len" \
    -vf "${hdr}${rot}fps=30,scale=1920:1080:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,format=yuv420p" \
    -c:v libx264 -preset medium -crf 14 -pix_fmt yuv420p "$work/clips/shot2.mp4"
  echo real > "$work/clips/shot2.kind"
  echo "$hand" > "$work/clips/shot2.hand"
else
  echo standin > "$work/clips/shot2.kind"                      # the hand film: edit.py puts it together from take hand
fi
out=$films/kaiku-$cut-$ver$tag.mp4

hsec exec --only ELEVENLABS_API_KEY,JUNCTION_GEMINI_API_KEY,DEEPGRAM_API_KEY -- python3 film/vo.py "$work"
nice -n 10 /usr/bin/python film/capture.py "$work"
nice -n 10 /usr/bin/python film/edit.py plan "$work" $open
(cd "$work/edit" && drun -- bash run.sh)
cp "$work/edit/out/film.mp4" "$out"
cp "$work/edit/out/film.srt" "${out%.mp4}.srt"
cp "$work/edit/out/report.json" "${out%.mp4}.report.json"
# one frame every two seconds, to look the whole film over at a glance
nice -n 10 ffmpeg -y -loglevel error -i "$out" -vf "fps=1/2,scale=480:270,tile=8x8" -frames:v 1 -q:v 3 "${out%.mp4}-contact.jpg"
echo "$out  $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out") s"
cat "${out%.mp4}.report.json"
