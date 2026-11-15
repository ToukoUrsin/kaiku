#!/usr/bin/env bash
# Put the team's own two clips around a finished film: the hello in front, the thank-you last.
#
#   film/team.sh FILM.mp4 OUT.mp4 [DIR]
#
# DIR holds intro.mp4 and thanks.mp4, already cut to 1920x1080, 30 fps, SDR, 48 kHz stereo
# (default /bulk3/runs/robotteleop/film/team). The film is not changed in between. Stops if the
# result is over 2:00. If FILM has an .srt beside it, OUT gets one with its times moved on.
set -euo pipefail
film="$1"; out="$2"; dir="${3:-/bulk3/runs/robotteleop/film/team}"
for f in "$dir/intro.mp4" "$dir/thanks.mp4" "$film"; do [ -f "$f" ] || { echo "STOP: $f is not there"; exit 1; }; done
len() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
lead=$(len "$dir/intro.mp4")
nice -n 10 ffmpeg -y -loglevel error -nostdin -i "$dir/intro.mp4" -i "$film" -i "$dir/thanks.mp4" -filter_complex \
  "[0:v]fps=30,format=yuv420p,setsar=1[v0];[1:v]fps=30,format=yuv420p,setsar=1[v1];[2:v]fps=30,format=yuv420p,setsar=1[v2];
   [0:a]aresample=48000,aformat=channel_layouts=stereo[a0];[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];[2:a]aresample=48000,aformat=channel_layouts=stereo[a2];
   [v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -c:v libx264 -crf 20 -preset medium -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart "$out"
got=$(len "$out")
python3 - "$got" <<'EOF'
import sys
got = float(sys.argv[1])
if got > 120.0:
    raise SystemExit(f"STOP: the film with the team's clips is {got:.2f} s, over 2:00")
print(f"{got:.2f} s with the team's clips")
EOF
srt="${film%.mp4}.srt"
if [ -f "$srt" ]; then
  python3 - "$srt" "${out%.mp4}.srt" "$lead" <<'EOF'
import re, sys
src, dst, lead = sys.argv[1], sys.argv[2], float(sys.argv[3])
def move(m):
    h, mi, s, ms = (int(x) for x in m.groups())
    t = h * 3600 + mi * 60 + s + ms / 1000 + lead
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(round(t % 1 * 1000)) % 1000:03d}"
open(dst, "w").write(re.sub(r"(\d\d):(\d\d):(\d\d),(\d\d\d)", move, open(src).read()))
EOF
fi
