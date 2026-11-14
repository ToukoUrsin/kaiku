# Kaiku film capture

Record the hosted stream (HTTP assets, HUD/events, binary deltas, and ACKs):

```bash
mkdir -p /bulk2/runs/robotteleop/film
.venv/bin/python tools/film/record.py https://touko-ursin--kaiku.us-east.modal.run /bulk2/runs/robotteleop/film/kaiku-150s --seconds 150 --at '60:mode:"policy"' --at '120:mode:"gloves"'
.venv/bin/python tools/film/capture.py /bulk2/runs/robotteleop/film/kaiku-150s --out /bulk2/runs/robotteleop/film/proof/gloves-overview.mp4 --start 20 --end 30 --fps 30 --size 1920x1080 --query 'clean=1&shot=overview&quality=high' --jobs 16
.venv/bin/python tools/film/capture.py /bulk2/runs/robotteleop/film/kaiku-150s --out /bulk2/runs/robotteleop/film/proof/gloves-lead.mp4 --start 30 --end 40 --fps 30 --size 1920x1080 --query 'clean=1&shot=lead&quality=high' --jobs 8
.venv/bin/python tools/film/capture.py /bulk2/runs/robotteleop/film/kaiku-150s --out /bulk2/runs/robotteleop/film/proof/switch-policy.mp4 --start 55 --end 65 --fps 30 --size 1920x1080 --query 'shot=overview&quality=high' --jobs 8
.venv/bin/python tools/film/capture.py /bulk2/runs/robotteleop/film/kaiku-150s --out /bulk2/runs/robotteleop/film/proof/phone.mp4 --start 58 --end 63 --fps 30 --size 390x844 --scale 1 --query 'clean=1&shot=lead&quality=high'
ffmpeg -i proof.mp4 -vf "select=not(mod(n\,30)),scale=480:-1,tile=5x2" /bulk2/runs/robotteleop/film/contact.jpg
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_read_frames proof.mp4
```

The capture server replays recorded HTTP responses and WebSocket messages; CDP virtual time advances by exactly one frame interval before each screenshot. `--jobs` splits frame-aligned segments, prefixes workers with `nice -n 10`, and joins them with ffmpeg stream copy. No page files were changed.
