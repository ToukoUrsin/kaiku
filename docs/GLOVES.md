# Smartgloves II: five minutes to the robot

You need two Smartgloves II, one Coil Pro, and the Rokoko Device SDK on the Mac. Keep the gloves and Coil
Pro within the SDK receiver's range.

## Plug in and start

1. Plug the receiver and Coil Pro into the Mac. Put on both gloves and turn them on. Wait about 20 seconds.
2. In Kaiku's checkout, run one command:

   ```bash
   python -m kaiku.app.gloves
   ```

   It starts Rokoko's driver and hand solver when they are not already running, then prints a checklist. It
   never opens a window. The default page is `http://127.0.0.1:8651/`.
3. Wait for each line to tick. A glove line shows its rate (normally about 100 Hz). The Coil Pro line means
   both wrists have a position; without it the hands can turn but cannot move the robot.

The command saves a session in `~/kaiku-sessions/YYYYMMDD-HHMM/`. The raw stream is
`kaiku.rgmp.record` in that folder, beside the demonstrations. Use `--session-dir DIR` to choose another
folder, `--n 8` on a small machine, or `--check` to run only the checklist. Ctrl-C stops only programs this
command started.

## Calibrate without a mouse

When both gloves and the Coil Pro are ready, hold both hands flat and still above the Coil Pro, palms down,
for about 1.5 seconds. The page shows a dot for each hand and a ring for the hold. Then close the left hand
once and the right hand once. It says **Calibrated. The robot follows your hands.** The button in the Operator
panel remains available. Press `C` on the page, or the button, to calibrate again; holding still while already
calibrated does not start a new capture.

## Save and replay

Every clean lead scene is saved with its table, command stream, and drawn hand points. The 47 echoes are saved
as demonstrations too. To replay the recorded human scenes on a new set of follower tables:

```bash
python -m kaiku.app.run --n 8 --session ~/kaiku-sessions/20261114-1040
```

The page labels the source **Recorded session** and draws the recorded hands. A lead scene that did not finish
cleanly is not saved and is skipped.

## If it does not work

| What you see | Do this |
|---|---|
| SDK is not found or licensed | Install the Device SDK and sign in once. |
| Driver or solver is not answering | Run `rkk-hand-solver --emit world` and run the command again. |
| One glove is missing | Wake it, check its receiver, and check the USB cable. |
| No position from the Coil Pro | Plug it in, keep it near the gloves, and wait for its line to tick. |
| Hand rate is low | Move closer to the receiver and remove other 2.4 GHz interference. |
| Hardware still fails | Stop the command and run `kaiku.app.run --session DIR` from the last clean session. |

## Check without hardware

Rokoko's driver can emit two still, emulated gloves. With the SDK installed, run:

```bash
rkk-hand-solver --emit world --driver-arg=--pipeline-source --driver-arg=default-emulated \
  --driver-arg=--emulated-devices --driver-arg=docs/emulated-rig.json
python -m kaiku.app.gloves --check
```

The emulated gloves check the SDK, wire format, solver, and finger path. They have no Coil Pro and do not move,
so the position line correctly stays unticked.
