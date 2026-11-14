# Kaiku

This is the version in force: no real gloves have been worn. The version for after a real glove session is [SUBMISSION.gloves.md](SUBMISSION.gloves.md).

**Punchline:** One hand, 48 robots.

**Challenge:** The Robot Teleoperation Challenge by Rokoko

**Demo:** https://kaiku-echo.fly.dev

## Description

Kaiku is Finnish for echo. One person sorts parts with a Unitree G1, and 47 more G1s do the same job at the same moment. Each one has its own scene: the parts are somewhere else, turned differently, sometimes a different shape. So the motion is not copied. It is redone on every table, relative to the parts that are actually there, while the person is still moving.

Robot learning runs out of demonstrations long before it runs out of anything else. From one set of 16 demonstrations a policy sorted under 1% of new scenes; other sets of 16 gave 0% to 31%. From the same 16 plus their echoes it sorted 81%. 32 demonstrations with echoes beat 128 without. Then nobody drives, and the 48 robots sort scenes they have never seen.

Open the demo and press Drive it yourself. Hold your hand up to the camera and the robot's fingers follow yours: the hand's 21 landmarks go down the same path Rokoko's gloves use. The grasp is assisted until you press Grasp it myself. Press See it without Kaiku to watch 47 robots copy the same motion and miss, then the same scene again with Kaiku on.

Every robot goes through the humanoid challenge's safety filter, and physics decides whether the parts really landed in the right bins. Only the clean runs are saved, in the humanoid challenge's LeRobot and GR00T format, and the page offers them as a download.

What is real: a person's hand, seen by a camera, drives the robot's fingers through the glove pipeline. Rokoko's driver and hand solver feed the same pipeline, tested with the driver's emulated Smartgloves II, as a pair and as a single glove. The humanoid challenge's own runner takes the lead robot's commands, 1,800 in a row through its safety filter with none clipped. What is not: nobody has worn real Smartgloves yet, it runs in a MuJoCo twin of the sorting station, and while no visitor drives the hosted demo a scripted hand in emulated gloves does. The page says which one is driving.

Redoing demonstrations in new scenes is known from MimicGen and DexMimicGen, where it runs afterwards on recordings. Kaiku does it live, from the hand stream, with physics and the safety filter deciding what counts. We measured the difference with our own MimicGen-style replay, on the same 16 demonstrations and the same 752 new tables. The replay kept 5 clean demonstrations and the echo kept 214. A policy trained with the replay's five sorted 8% of tables it had not seen. With the echoes, 73%.

## How we used Modal

The policy study and the matched comparison were trained and tested on Modal. Each run got a container of its own with a T4 and 16 cores: the policy trains on the GPU and is then rolled out on the cores, over scenes it had not seen. That is 48 runs for the study (8 demonstration counts, with and without echoes, 3 seeds each) plus the comparison against our MimicGen-style replay. The Train button on the live page trains its two policies on Modal while you watch, about four minutes from the click to the result. The camera export of the dataset runs on a Modal GPU, and the demonstrations sit on a Modal Volume.

## Deck line

A person's hand drives one robot. 47 more redo the job on their own tables, and a policy learns from all 48.

## Booth pitch

Hold your hand up to the laptop camera. The robot's fingers follow yours and you sort a part. Behind it 47 more robots do the same job, each on a table where the parts lie somewhere else. A policy trained on 16 demonstrations sorted almost no new scenes. With the echoes, 81%. Rokoko gloves plug into the same stream. We tested that with Rokoko's driver and its emulated gloves, because we had no real ones.

## Technologies

Rokoko Smartgloves II, Rokoko Device SDK (RGMP, hand solver), Unitree G1 with Dex3 hands, MuJoCo, MediaPipe hand landmarks, Python, PyTorch, Modal, LeRobot v3.0, GR00T, three.js, Fly.io
