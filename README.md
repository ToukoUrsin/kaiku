# Kaiku

One hand drives 48 robots at once.

Kaiku is Finnish for echo. One person's hand sorts parts with one Unitree G1, and 47 more G1s do the same job at the same moment, each in a scene of its own: the parts lie elsewhere, are turned differently, sometimes have another shape. Physics and the challenge's safety filter check every robot, and the ones that sorted their scene are saved as demonstrations in the challenge's dataset format. A policy trained on them then sorts scenes it has never seen, with nobody driving.

**Live demo, nothing to install:** https://kaiku-echo.fly.dev

Built at Junction 2026 for the Robot Teleoperation Challenge by Rokoko.

## What the echoes are worth

The same operator, the same model, the same training, the same test. The only difference is whether the policy also gets the echoes of each demonstration.

| Demonstrations by the operator | Alone | With echoes |
|---:|---:|---:|
| 4 | 0% | 23% |
| 8 | 0% | 36% |
| 16 | 0.35% (0%, 1.04%, 0%) | 81% |
| 32 | 58% | 95% |
| 64 | 83% | 98% |
| 128 | 92% | 99% |

For the original fixed 16-demo set, the three plain runs averaged 0.35% (0%, 1.04%, 0%) and the echo runs averaged 81%. Other 16-demo sets varied much more: the live page recorded 31%, 0%, 23% and 0% alone. The old single-set “0%” headline was therefore too strong. 32 demonstrations with echoes beat 128 without, so the operator drives a quarter of the time.

Each number is the mean of 3 training runs, each tested on 96 scenes neither policy had seen. A scene counts only when every part rests in its own bin and the safety filter clipped no command. The numbers are in [`study.json`](kaiku/policy/pretrained/study.json) and all 48 runs, with every count behind them, in [`study_runs.jsonl`](kaiku/policy/pretrained/study_runs.jsonl). The Train button in the demo repeats the comparison on the last 16 demonstrations of the session while you watch.

## Against MimicGen-style generation and plain copying

Three ways to turn 16 demonstrations into more were given the same 16 scenes and the same 47 new tables for each (752 attempts): Kaiku's echo, a MimicGen-style replay we wrote ourselves, and copying the lead's joints. A demonstration counts when every part rests in its own bin and the safety filter clipped nothing.

| method | clean / attempts | kept (95% interval) | 3 parts | 4 parts | 5 parts |
|---|---:|---:|---:|---:|---:|
| Kaiku's echo | 214 / 752 | 28.5% (25.4–31.8%) | 47.5% | 31.4% | 12.3% |
| MimicGen-style replay | 5 / 752 | 0.7% (0.3–1.5%) | 2.1% | 0.5% | 0% |
| copying the lead's joints | 0 / 752 | 0% (0–0.5%) | 0% | 0% | 0% |

Then one policy was trained on each result, three times, and tested on 96 tables none of them had seen:

| the policy was trained on | training pairs | tables sorted, of 288 (95% interval) | lowest and highest run |
|---|---:|---:|---:|
| the 16 demonstrations alone | 16,401 | 9.0% (6.2–12.9%) | 6–14% |
| + what copying kept (nothing) | 16,401 | 9.0% (6.2–12.9%) | 6–14% |
| + the 5 the MimicGen-style replay kept | 17,180 | 8.0% (5.4–11.7%) | 0–16% |
| + 5 of Kaiku's echoes, the same number | 21,296 | 19.4% (15.3–24.4%) | 12–27% |
| + all 214 of Kaiku's echoes | 226,789 | 72.9% (67.5–77.7%) | 71–76% |

Same model, same 30,000 training steps, same seeds, same 96 tables for every row. The runs are in [`baselines.json`](kaiku/policy/pretrained/baselines.json) and [`baseline_runs.jsonl`](kaiku/policy/pretrained/baseline_runs.jsonl), the 96 tables in [`baseline_heldout96.jsonl`](kaiku/policy/pretrained/baseline_heldout96.jsonl).

What this does and does not say:

- The MimicGen-style replay is ours: [`baseline.py`](kaiku/policy/baseline.py). It finds each reach-and-grasp and carry-and-drop from the measured lift and rest times, moves the hand path into the new part's frame, and replays it open loop through the same IK, safety filter and physics. It is not MimicGen's or DexMimicGen's code, and it has none of their learned or replanned recovery. A better implementation would keep more.
- These tables are harder than the ones the live demo draws: 637 of the 752 moved the parts 10 cm or more on average, and none was checked for reach. That is why the echo kept 28.5% here and keeps nearly every table live, where each follower's table is drawn within the robot's reach.
- Where Kaiku does not win: the MimicGen-style batch took less time in total (1,199 s against 1,503 s), and on the 8 tables closest to the source it kept 2 while the echo kept 4. Per kept demonstration the echo took 7 s and the replay 240 s.
- The 16 demonstrations came from the scripted stand-in operator, not from a person, and the policy reads part positions from the simulator.
- An earlier pass with 10,000 steps on CPUs sorted no held-out table in any row. It is kept in `baselines.json` and is not the result above.

To repeat it: `python tools/run_baseline_measure.py manifest ...` once, then `run` with `--method echo`, `mimicgen` and `copy` on that manifest, `tools/assemble_policy_dataset.py` for each, and `modal run tools/modal_matrix.py` for the policies (51 minutes of T4 time in all). The generation took 3,450 seconds of simulator time in sandboxes.

## Who drives

Every source goes down one path: a hand solver, retargeting to the Dex3's three fingers, the wrist through arm inverse kinematics. The page always says which one is driving.

- **Your own hand, on the hosted page.** Press Drive it yourself, then Use my hand. Your camera's 21 hand landmarks go down that path and the robot's fingers follow yours, each finger its own: curl your index finger and the robot's index finger closes, touch thumb to middle finger and its middle finger closes, make a fist and the hand closes. The page shows how far each is closed. The picture stays in your browser. The grasp is assisted until you press Grasp it myself; then your fingers do it and the robot only turns its wrist and lifts. Without a camera, a pointer works.
- **One Smartglove.** `--one-glove HOST:12276`. The fingers and the pinch come from the glove and the position from a pointer or a camera. With a Coil Pro the position is the glove's own: hold the hand flat and still for two seconds and the robot's hand goes where yours goes.
- **A pair of Smartgloves on a Coil Pro.** `--gloves HOST:12276`, both arms.
- **Nobody.** A scripted hand in emulated Smartgloves II, with tremor and sensor noise, sent through the same socket, solver and retargeting.

The camera path is checked on film of real hands that are not ours: a fist opening one finger at a time, filmed from above (its landmarks are in `tests/fixtures/real_hand`, two tests run on them), and a hand held up to a camera touching thumb to each fingertip, played into the page as its camera. Nobody has worn real Smartgloves with Kaiku yet. The glove paths are tested with Rokoko's own driver and hand solver and the driver's emulated Smartgloves II.

## What is new

Redoing a demonstration around the parts of another scene is known from [MimicGen](https://mimicgen.github.io) and [DexMimicGen](https://dexmimicgen.github.io). There it runs afterwards, on recorded demonstrations. Kaiku does it during teleoperation: the 47 echoes move while the operator moves, from the operator's hand as it moves, so the operator sees at once on which tables the motion works. Every echo passes the challenge's safety filter and physics before it counts, and what is saved is already the challenge's dataset.

## How it works

1. **Hand to robot.** Smartgloves II stream over RGMP (Rokoko's TCP protocol); a camera gives 21 hand landmarks. Kaiku solves the hand from either and retargets it to the G1's arms and Dex3 hands: fingers to the three-finger hand, wrists through arm inverse kinematics.
2. **Echo.** Copying joint angles only works when every scene is the same. Kaiku copies the motion relative to the things on the table instead. A follower's hand goes where the operator's hand is relative to its own part, using where the parts are right now in both scenes. A follower with further to reach falls a little behind instead of closing its fingers too early.
3. **Check.** Every command of every robot goes through the challenge's safety filter (arm 1 rad/s, hand 3 rad/s, workspace box, counter keep-out). MuJoCo decides whether the parts really ended in their bins. An echo that drops a part is not saved. Kaiku's filter runs all 48 robots at once, so it is checked against the challenge's own code (unchanged, in `third_party/aaltonomous_safety`): 500 saved demonstrations, 504,817 ticks, went through it with no command changed (`tools/check_safety.py`).
4. **Dataset.** The recorder writes the challenge's interface: 28 joint targets and 28 joint states at 30 Hz in the pelvis frame. `kaiku.data.export` turns a session into LeRobot v3.0 and GR00T datasets, with where the parts are in every frame and who drove each episode, and the twin's cameras rendered on request ([docs/DATASET.md](docs/DATASET.md)).
5. **Policy.** A small network (4 residual layers, 512 wide) reads the robot and the parts and plans the next half second. It trains in two minutes on a rented T4 and drives the robots through the same safety filter. The Train button on the hosted demo trains its two policies on Modal, four to six minutes from the click to the result; if Modal cannot be reached it trains on the demo's own machine, which takes about 25.

## Run it

```bash
git clone https://github.com/ToukoUrsin/kaiku && cd kaiku
python -m venv .venv && .venv/bin/pip install -e ".[policy,export,dev]"
.venv/bin/python -m kaiku.app.run          # then open http://localhost:8651
```

The first run fetches the G1 model from MuJoCo Menagerie (39 MB). 48 robots at 30 Hz run in real time on an 18-core Apple M5 Pro; `--n 8` runs on a small laptop.

With Smartgloves, point Kaiku at the machine where the Rokoko driver (Device SDK) runs. It streams RGMP on port 12276:

```bash
.venv/bin/python -m kaiku.app.run --gloves 192.168.1.20:12276 --record demos
```

Rokoko's own hand solver gives the fingers when it runs next to the driver, and calibration is one flat-hands pose: [docs/GLOVES.md](docs/GLOVES.md).

Without gloves a stand-in operator drives: a scripted person wearing emulated Smartgloves II, with tremor and sensor noise, sent through the same RGMP socket, hand solver and retargeting as real gloves.

```bash
python -m kaiku.rgmp.record --host HOST --out session.rgmp    # keep a real glove session
python -m kaiku.rgmp.replay session.rgmp                      # and serve it again, byte for byte
python -m kaiku.policy.fit demos --out policies               # train with and without echoes, test both
python -m kaiku.policy.rollout policies/kaiku.pt              # 48 robots, 48 new scenes, no operator
python -m kaiku.data.export demos out --limit 20              # LeRobot v3.0 and GR00T (the page's Dataset tab downloads the last scene)
pytest                                                        # the tests, about three minutes
```

## What is real and what is not

- Everything runs in a MuJoCo twin of the Aaltonomous sorting station. No real G1 has moved yet. The way there is built: `--g1 8000` serves the lead robot to the humanoid challenge's own runner as a policy server ([docs/G1.md](docs/G1.md)). With the runner's mock robot it followed the operator for 60 seconds through the official safety filter: 1,800 ticks, 29.6 commands a second, none clipped.
- The default simulator comparison reads where the parts are from the simulator. The camera-fed comparison is reported in the next section. Both are simulated in the MuJoCo twin.
- The operator in the study is the stand-in, recorded live at real time on the hosted demo. No real gloves have been worn yet. The glove path runs against Rokoko's real driver and hand solver (Device SDK 2026.10.20) with the driver's emulated Smartgloves II, and against Kaiku's own emulator and recordings.
- The parts and bins are the challenge's (45 mm cube, cylinder and triangle; red, yellow and blue bins). 3 to 5 parts per scene.

## From the camera

| what tells the policy where the parts are | scenes sorted | parts sorted |
|---|---:|---:|
| the simulator | 96 of 96 | 370 of 370 |
| the head camera | 18 of 96 | 189 of 370 |
| the head camera with 3 mm depth noise and 1 px edge noise | 20 of 96 | 190 of 370 |

The camera uses `camera_noisy_shape.pt`; the simulator row uses `kaiku.pt`. In three zero-progress scenes, the first large error was a 31.8 degree cube yaw error at 0.4 seconds, a 43.2 degree triangle yaw error in the first frame, and a 10.9 mm position shift at 2.8 seconds while a closed hand was over the part. The policy now decides each tabletop pose, yaw, and shape from clear frames, keeps it while a hand is near, follows the hand while carrying, and looks again after a failed grasp. It sees the robot joints, known bin places, and head-camera colour and depth images. The camera is rendered in the MuJoCo twin, so the robot, parts, bins, and physics remain simulated.

## Layout

```
kaiku/rgmp      RGMP client, glove decoder, emulator, record and replay
kaiku/hand      hand solver, calibration, filters
kaiku/teleop    retargeting to the G1, the stand-in operator
kaiku/fanout    the echo
kaiku/sim       MuJoCo twin, batch physics, arm IK, safety filter
kaiku/data      recorder, LeRobot and GR00T export
kaiku/policy    features, training, rollout, the shipped policy and study
kaiku/app       the 30 Hz runner behind the demo, the lead robot as a policy server for a real G1
kaiku/web       the page
tools/          the study and training on Modal, the camera export, the safety check, film capture
deploy/fly      the hosted demo
```

MIT license. The G1 model is from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3-Clause).
