"""What was on the hand in a booth session, written where python -m kaiku.app.sitting reads it."""
import json
import os
import sys
import time

from kaiku.rgmp import GloveStream

folder, port = sys.argv[1], int(sys.argv[2])
stream = GloveStream("127.0.0.1", port).start()
time.sleep(1.5)
frames = [stream.latest(side) for side in ("left", "right")]
emulated = bool(stream.emulated)
stream.stop()
source = open(os.path.join(folder, "source")).read().strip()
with open(os.path.join(folder, "demos", "session.json"), "w") as f:
    json.dump({"format": 1, "source": "emulated" if emulated else source,
               "start_time": os.path.getctime(os.path.join(folder, "started")),
               "glove_count": sum(frame is not None for frame in frames),
               "coil_present": any(bool(frame and frame.coil_present) for frame in frames)}, f, indent=2)
