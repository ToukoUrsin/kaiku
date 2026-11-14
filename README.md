# Kaiku

One pair of Rokoko gloves drives 48 robots at once.

Kaiku is Finnish for echo. You sort parts with one Unitree G1 through Smartgloves, and 47 more G1s do the same job at the same moment, each in a scene of its own: the parts lie elsewhere, are turned differently, sometimes have another shape. Physics and the challenge's safety filter check every robot, and the ones that sorted their scene are saved as demonstrations in the challenge's dataset format. A policy trained on them then sorts scenes it has never seen, with nobody driving.

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

For the original fixed 16-demo set, the three plain runs averaged 0.35% (0%, 1.04%, 0%) and the echo runs averaged 81%. Other 16-demo sets varied much more: the live page recorded 31%, 0%, 23% and 0% alone. The old single-set “0%” headline was therefore too strong. 32 demonstrations with echoes beat 128 without, so the operator wears the gloves a quarter of the time.

Each number is the mean of 3 training runs, each tested on 96 scenes neither policy had seen. A scene counts only when every part rests in its own bin and the safety filter clipped no command. The numbers are in [`study.json`](kaiku/policy/pretrained/study.json) and all 48 runs, with every count behind them, in [`study_runs.jsonl`](kaiku/policy/pretrained/study_runs.jsonl). The Train button in the demo repeats the comparison on the last 16 demonstrations of the session while you watch.

## Against MimicGen, and how sure the numbers are

MimicGen's published recipe splits a demonstration into object-centred pieces, moves each piece into the new object's pose, joins pieces, executes the result, and keeps only clean runs. We implemented that rule in [`baseline.py`](kaiku/policy/baseline.py), using the same inverse kinematics, shaping, safety filter and MuJoCo verdict as Kaiku. The carry and drop pieces use the bin frame. DexMimicGen's extra coordination subtasks do not arise here because the two hands never hold one part together.

The requested full 32 lead scenes × 47 tables and five 16-demo policy sets are not complete in this checkout. The host was overloaded and Modal was over its spend limit, so we ran the fast geometry tests and one lead × three-table simulator smoke test instead. All three smoke attempts were rejected as clean demonstrations (0/3, 0%); each took 33.4 seconds. That is a generator smoke measurement, not a claim about the full baseline. The complete audit record, including the older policy runs that must not be mixed into this comparison, is [`baselines.json`](kaiku/policy/pretrained/baselines.json).

| source | clean demonstrations | attempts | yield | when it can run |
|---|---:|---:|---:|---|
| Kaiku echo recording | 13,092 saved | denominator not recorded | — | live, while the operator drives |
| MimicGen re-implementation smoke | 0 | 3 | 0% | after the lead episode |
| copy | not run | — | — | after the lead episode |

Kaiku can follow a part that slides and show a failing table while the operator is still driving because the anchors are live. The offline baseline cannot provide that timing. We do not claim a policy advantage from the smoke result, and no matched MimicGen policy number is reported until the requested Fly run is completed. These are our simulator results and our re-implementation, not the MimicGen or DexMimicGen code, datasets or benchmarks.

## What is new

Redoing a demonstration around the parts of another scene is known from [MimicGen](https://mimicgen.github.io) and [DexMimicGen](https://dexmimicgen.github.io). There it runs afterwards, on recorded demonstrations. Kaiku does it during teleoperation: the 47 echoes move while the operator moves, from the glove stream itself, so the operator sees at once on which tables the motion works. Every echo passes the challenge's safety filter and physics before it counts, and what is saved is already the challenge's dataset.

## How it works

1. **Gloves to robot.** Smartgloves II stream over RGMP (Rokoko's TCP protocol). Kaiku reads the stream, solves both hands, and retargets them to the G1's arms and Dex3 hands: fingers to the three-finger hand, wrists through arm inverse kinematics.
2. **Echo.** Copying joint angles only works when every scene is the same. Kaiku copies the motion relative to the things on the table instead. A follower's hand goes where the operator's hand is relative to its own part, using where the parts are right now in both scenes. A follower with further to reach falls a little behind instead of closing its fingers too early.
3. **Check.** Every command of every robot goes through the challenge's safety filter (arm 1 rad/s, hand 3 rad/s, workspace box, counter keep-out). MuJoCo decides whether the parts really ended in their bins. An echo that drops a part is not saved. Kaiku's filter runs all 48 robots at once, so it is checked against the challenge's own code (unchanged, in `third_party/aaltonomous_safety`): 500 saved demonstrations, 504,817 ticks, went through it with no command changed (`tools/check_safety.py`).
4. **Dataset.** The recorder writes the challenge's interface: 28 joint targets and 28 joint states at 30 Hz in the pelvis frame. `kaiku.data.export` turns a session into LeRobot v3.0 and GR00T datasets, with the twin's cameras rendered on request.
5. **Policy.** A small network (4 residual layers, 512 wide) reads the robot and the parts and plans the next half second. It trains in two and a half minutes on an A100 and drives the robots through the same safety filter.

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
pytest                                                        # 108 tests
```

## What is real and what is not

- Everything runs in a MuJoCo twin of the Aaltonomous sorting station. No real G1 has moved yet. The way there is built: `--g1 8000` serves the lead robot to the humanoid challenge's own runner as a policy server ([docs/G1.md](docs/G1.md)). With the runner's mock robot it followed the operator for 60 seconds through the official safety filter: 1,800 ticks, 29.6 commands a second, none clipped.
- The policy reads where the parts are from the simulator, not from cameras. On the real station that has to come from the head camera.
- The operator in the study is the stand-in, recorded live at real time on the hosted demo. No real gloves have been worn yet. The glove path runs against Rokoko's real driver and hand solver (Device SDK 2026.10.20) with the driver's emulated Smartgloves II, and against Kaiku's own emulator and recordings.
- The parts and bins are the challenge's (45 mm cube, cylinder and triangle; red, yellow and blue bins). 3 to 5 parts per scene.

## From the camera, not the simulator

The simulator-state policy sorted all 96 evaluation scenes (370/370 parts) with no clipped
command.  The same pretrained policy, same seed and normal 80-second cap sorted 0/96 scenes
from the head camera, even with clean images.  The camera is therefore the limiting part of this
path today.

| policy input | scenes sorted | parts sorted | unsafe/clipped commands | 95% scene interval |
|---|---:|---:|---:|---:|
| simulator object state | 96/96 | 370/370 | 0 | 96.23–100% |
| camera, clean | 0/96 | 19/370 | 0 | 0–3.77% |
| camera, 3 mm depth + 1 px edge noise | 0/96 | 21/370 | 0 | 0–3.77% |
| camera, same noise after shape fix | 0/96 | 16/370 | 0 | 0–3.77% |

The noisy policy result has one seed (`n=1`); a second noisy policy seed was not run because
rendering the 96 scenes took about 30 minutes even after sharding the CPU work.  The JSONL scene
records and summaries are in [`perception_runs.jsonl`](kaiku/policy/pretrained/perception_runs.jsonl)
and the four arm files beside it.  The intervals are exact two-sided 95% binomial intervals.
To reproduce a summary from any arm file:

```bash
python -c 'import json,sys; r=[json.loads(x) for x in open(sys.argv[1])]; print(len(r),sum(x["success"] for x in r),sum(x["sorted"] for x in r),sum(x["parts"] for x in r),sum(x["clipped"] for x in r))' kaiku/policy/pretrained/perception_camera_clean_100.jsonl
```

The camera estimate uses only the head camera's colour and depth images.  On a separate
development set of 200 rendered parts (seed 777), the clean estimate was 3.80 mm mean position
error (14.77 mm at p95), 2.93 degrees mean turn error after symmetry, 90.5% shape accuracy and
100% colour accuracy.  With 3 mm depth noise and one-pixel edge noise it was 5.39 mm, 14.95 mm,
3.79 degrees, 68.5% and 100%.  The earlier noisy shape measurement was 43%; the change uses
projected top-face area and compactness to make the edge estimate less brittle.  On the held-out
policy seed this did not improve sorting (21 parts before, 16 after), so that result is kept.

In camera mode the policy still receives the robot's own arm and hand joint state (a real G1
measures that itself), the fixed head-camera calibration, and the known bin geometry.  The
simulator remains the referee for success and clipping.  No simulator object position,
orientation, or identity is passed to the policy.  This is a simulated camera in a MuJoCo twin,
not a real camera on a real G1.

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
