# Kaiku, after a real glove session

Use this instead of [SUBMISSION.md](SUBMISSION.md) once a person has driven Kaiku in real Smartgloves, and not before. `python -m kaiku.app.sitting` says so in `SITTING.md` when that session exists and is valid.

**Punchline:** One pair of Rokoko gloves, 48 robots.

**Challenge:** The Robot Teleoperation Challenge by Rokoko

**Demo:** https://kaiku-echo.fly.dev

## Description

Kaiku is Finnish for echo. One person in Rokoko Smartgloves sorts parts with a Unitree G1, and 47 more G1s do the same job at the same moment. Each one has its own scene: the parts are somewhere else, turned differently, sometimes a different shape. So the motion is not copied. It is redone on every table, relative to the parts that are actually there, while the operator is still moving.

Robot learning runs out of demonstrations long before it runs out of anything else. From one set of 16 demonstrations a policy sorted under 1% of new scenes; other sets of 16 gave 0% to 31%. From the same 16 plus their echoes it sorted 81%. 32 demonstrations with echoes beat 128 without. Then nobody drives, and the 48 robots sort scenes they have never seen.

The session in the film and in the hosted replay is a person wearing real Smartgloves at Junction, read by Rokoko's driver and hand solver. Open the demo to watch it replayed with its echoes, or press Drive it yourself: without gloves a camera sees your hand and the robot's fingers follow yours, down the same path the gloves use. Press See it without Kaiku to watch 47 robots copy the same motion and miss, then the same scene again with Kaiku on.

Every robot goes through the humanoid challenge's safety filter, and physics decides whether the parts really landed in the right bins. Only the clean runs are saved, in the humanoid challenge's LeRobot and GR00T format, and the page offers them as a download.

What is real: real Smartgloves on a real person, Rokoko's driver and hand solver, and the humanoid challenge's own runner taking the lead robot's commands, 1,800 in a row through its safety filter with none clipped. What is not: it runs in a MuJoCo twin of the sorting station, no real G1 has moved, and while nobody drives the hosted demo it replays the recorded session. The page says what is driving.

Redoing demonstrations in new scenes is known from MimicGen and DexMimicGen, where it runs afterwards on recordings. Kaiku does it live, from the hand stream, with physics and the safety filter deciding what counts. We measured the difference with our own MimicGen-style replay, on the same 16 demonstrations and the same 752 new tables. The replay kept 5 clean demonstrations and the echo kept 214. A policy trained with the replay's five sorted 8% of tables it had not seen. With the echoes, 73%.

## How we used Modal

The policy study and the matched comparison were trained and tested on Modal. Each run got a container of its own with a T4 and 16 cores: the policy trains on the GPU and is then rolled out on the cores, over scenes it had not seen. That is 48 runs for the study (8 demonstration counts, with and without echoes, 3 seeds each) plus the comparison against our MimicGen-style replay. The camera export of the dataset runs on a Modal GPU, and the demonstrations sit on a Modal Volume.

## Deck line

One operator in Rokoko gloves drives one robot. 47 more redo the job on their own tables, and a policy learns from all 48.

## Booth pitch

One person wore Rokoko gloves and sorted parts with one robot. Behind it 47 more robots did the same job, each on a table where the parts lie somewhere else. A policy trained on 16 demonstrations sorted almost no new scenes. With the echoes, 81%. That is the recording playing now. Hold your hand up to the camera and you drive it yourself.

## If it was one glove

Rokoko's booth had one glove. If that is the session, change three lines and nothing else:

- Punchline: One Rokoko glove, 48 robots.
- First sentence of the description: One person wearing one Rokoko Smartglove sorts parts with a Unitree G1, and 47 more G1s do the same job at the same moment.
- What is real: one real Smartglove on a real person. Its fingers drove the robot's fingers; the hand's position came from the Coil Pro, or from the laptop's pointer when there was none.

## Technologies

Rokoko Smartgloves II, Rokoko Device SDK (RGMP, hand solver), Unitree G1 with Dex3 hands, MuJoCo, MediaPipe hand landmarks, Python, PyTorch, Modal, LeRobot v3.0, GR00T, three.js, Fly.io
