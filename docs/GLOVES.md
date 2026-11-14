# Real Smartgloves II

What you need: both Smartgloves II, a Coil Pro (it gives the hands their position; without it they only
turn), and the Rokoko Device SDK on the machine the gloves are plugged into.

## 1. Rokoko's driver and hand solver

On the machine with the gloves:

```bash
rkk-hand-solver --emit world --driver-arg=--auto-stream-usb
```

That starts Rokoko's driver (`rokoko-sdk`, RGMP on port 12276) and Rokoko's own hand solver (port 12277).
`--auto-stream-usb` matters: without it the driver accepts connections and sends nothing from gloves on USB.
`--emit world` is what Kaiku reads the fingers from.

If the driver is already running, add `--no-driver`.

## 2. Kaiku

Same machine (a laptop does 16 robots):

```bash
python -m kaiku.app.run --n 16 --gloves 127.0.0.1 --record demos      # then open http://localhost:8651
```

Or all 48 robots on a bigger machine, with the gloves' two ports forwarded to it:

```bash
ssh -N -L 12276:127.0.0.1:12276 -L 12277:127.0.0.1:12277 GLOVE_MACHINE &
python -m kaiku.app.run --gloves 127.0.0.1 --record demos
```

The Operator panel on the page shows both gloves and their rate (about 100 Hz) once they stream.

## 3. Calibrate

1. Sit so that your hands rest comfortably above the Coil Pro, where you want the robot's hands to be at home.
2. Hold both hands flat, palms down, fingers together and pointing away from you.
3. Press **Calibrate** in the Operator panel.
4. Close each hand once, thumb to fingers. Kaiku learns how small your gloves read a closed hand.

From then on your hands move the lead robot's hands about that home pose, and closing thumb and fingers
around a part grips it.

Kaiku refuses to calibrate when the hands are not flat. It also finds out by itself which way up the Coil
Pro's frame is, from the backs of your flat hands.

## What comes from where

- Wrist position and rotation: the driver's stream (hub and arm sensors against the Coil Pro), solved by Kaiku.
- Fingers: Rokoko's hand solver whenever it runs; Kaiku's own solver otherwise (`--hand-solver off` forces
  that).
- How far the robot's fingers close: the distance between thumb tip and fingertips, scaled to your hands
  from the flat pose and your tightest pinch.

## Keep the session

```bash
python -m kaiku.rgmp.record --host 127.0.0.1 --out session.rgmp     # the raw glove stream, until Ctrl-C
python -m kaiku.rgmp.replay session.rgmp                            # serves it again, byte for byte
python -m kaiku.app.run --gloves 127.0.0.1:12276 --hand-solver off  # and Kaiku runs on the replay
```

## Checked without hardware

Rokoko's driver can emulate gloves. This is how the glove path was tested against the real driver
(Device SDK 2026.10.20, driver 1.9.7) before any hardware was at hand:

```bash
cat > rig.json <<'EOF'
[{"type": "smartglove", "hub_addr": 11, "hw_ver": {"major": 3, "minor": 0}, "fw_ver": {"major": 3, "minor": 4, "patch": 0}, "is_left": true},
 {"type": "smartglove", "hub_addr": 12, "hw_ver": {"major": 3, "minor": 0}, "fw_ver": {"major": 3, "minor": 4, "patch": 0}, "is_left": false}]
EOF
rkk-hand-solver --emit world --driver-arg=--pipeline-source --driver-arg=default-emulated \
                --driver-arg=--emulated-devices --driver-arg=rig.json
python -m kaiku.app.run --n 8 --gloves 127.0.0.1
```

Both gloves connect at about 100 Hz and the fingers come from Rokoko's solver. The emulated gloves do not
move and have no Coil Pro, so this checks the wire format, the decoding and the finger path, not a full
teleoperation session.
