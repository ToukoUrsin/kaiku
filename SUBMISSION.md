# Kaiku

**Punchline:** One pair of Rokoko gloves, 48 robots.

**Challenge:** The Robot Teleoperation Challenge by Rokoko

**Demo:** https://touko-ursin--kaiku.us-east.modal.run

## Description

Kaiku is Finnish for echo. One person sorts parts with a Unitree G1 through Rokoko Smartgloves, and 47 more G1s do the same job at the same moment. Each one has its own scene: the parts are somewhere else, turned differently, sometimes a different shape. So the motion is not copied. It is redone on every table, relative to the parts that are actually there, while the operator is still moving.

Every robot goes through the humanoid challenge's safety filter, and physics decides whether the parts really landed in the right bins. Only the clean runs are saved, in the humanoid challenge's LeRobot and GR00T format.

Robot learning runs out of demonstrations long before it runs out of anything else. From 16 demonstrations by the operator a policy sorted 0% of new scenes. From the same 16 plus their echoes it sorted 81%. 32 demonstrations with echoes beat 128 without. Then the gloves come off, and all 48 robots sort scenes they have never seen.

Open the demo and press Train to rerun that comparison live, or Policy to watch the robots work with nobody driving.

It runs in a MuJoCo twin of the sorting station. The commands are the real robot's: 28 joint targets at 30 Hz inside its safety limits. In the hosted demo the gloves are emulated Smartgloves II on a scripted hand, streamed over RGMP through the same pipeline as real ones.

## Technologies

Rokoko Smartgloves II, Rokoko Device SDK (RGMP), Unitree G1 with Dex3 hands, MuJoCo, Python, PyTorch, LeRobot v3.0, GR00T, three.js, Modal
