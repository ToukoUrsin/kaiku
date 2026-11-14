# Kaiku

One pair of Rokoko gloves drives 48 robots at once.

Kaiku is Finnish for echo. You sort parts with one Unitree G1 through Smartgloves, and 47 more G1s do the same job at the same moment, each in a scene of its own: the parts lie elsewhere, are turned differently, sometimes have another shape. Physics and the challenge's safety filter check every robot, and the ones that sorted their scene are saved as demonstrations in the challenge's dataset format. A policy trained on them then sorts scenes it has never seen, with nobody driving.

**Live demo, nothing to install:** https://touko-ursin--kaiku.us-east.modal.run

Built at Junction 2026 for the Robot Teleoperation Challenge by Rokoko.

## What the echoes are worth

The same operator, the same model, the same training, the same test. The only difference is whether the policy also gets the echoes of each demonstration.

| Demonstrations by the operator | Alone | With echoes |
|---:|---:|---:|
| 4 | 0% | 23% |
| 8 | 0% | 36% |
| 16 | 0% | 81% |
| 32 | 58% | 95% |
| 64 | 83% | 98% |
| 128 | 92% | 99% |

From 16 demonstrations a policy sorts nothing on its own and 81% of new scenes with echoes. 32 demonstrations with echoes beat 128 without, so the operator wears the gloves a quarter of the time.

Each number is the mean of 3 training runs, each tested on 96 scenes neither policy had seen. A scene counts only when every part rests in its own bin and the safety filter clipped no command. The runs are in [`kaiku/policy/pretrained/study.json`](kaiku/policy/pretrained/study.json), and the Train button in the demo repeats the comparison on the last 16 demonstrations of the session while you watch.

## How it works

1. **Gloves to robot.** Smartgloves II stream over RGMP (Rokoko's TCP protocol). Kaiku reads the stream, solves both hands, and retargets them to the G1's arms and Dex3 hands: fingers to the three-finger hand, wrists through arm inverse kinematics.
2. **Echo.** Copying joint angles only works when every scene is the same. Kaiku copies the motion relative to the things on the table instead. A follower's hand goes where the operator's hand is relative to its own part, using where the parts are right now in both scenes. A follower with further to reach falls a little behind instead of closing its fingers too early.
3. **Check.** Every command of every robot goes through the challenge's safety filter (arm 1 rad/s, hand 3 rad/s, workspace box, counter keep-out). MuJoCo decides whether the parts really ended in their bins. An echo that drops a part is not saved.
4. **Dataset.** The recorder writes the challenge's interface: 28 joint targets and 28 joint states at 30 Hz in the pelvis frame. `kaiku.data.export` turns a session into LeRobot v3.0 and GR00T datasets, with the twin's cameras rendered on request.
5. **Policy.** A small network (4 residual layers, 512 wide) reads the robot and the parts and plans the next half second. It is trained in two minutes on a T4 and drives the robots through the same safety filter.

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
python -m kaiku.data.export demos out --limit 20              # LeRobot v3.0 and GR00T
pytest                                                        # 100 tests
```

## What is real and what is not

- Everything runs in a MuJoCo twin of the Aaltonomous sorting station. No real G1 has moved yet. The commands are the challenge's own (joint targets, 30 Hz, safety limits), so the lead robot's stream can go to the real one.
- The operator in the study is the stand-in, recorded live at real time on the hosted demo. No real gloves have been worn yet. The glove path runs against Rokoko's real driver and hand solver (Device SDK 2026.10.20) with the driver's emulated Smartgloves II, and against Kaiku's own emulator and recordings.
- The parts and bins are the challenge's (45 mm cube, cylinder and triangle; red, yellow and blue bins). 3 to 5 parts per scene.

## Layout

```
kaiku/rgmp      RGMP client, glove decoder, emulator, record and replay
kaiku/hand      hand solver, calibration, filters
kaiku/teleop    retargeting to the G1, the stand-in operator
kaiku/fanout    the echo
kaiku/sim       MuJoCo twin, batch physics, arm IK, safety filter
kaiku/data      recorder, LeRobot and GR00T export
kaiku/policy    features, training, rollout, the shipped policy and study
kaiku/app       the 30 Hz runner behind the demo
kaiku/web       the page
tools/          Modal jobs: the hosted demo, the study, the camera export
```

MIT license. The G1 model is from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3-Clause).
