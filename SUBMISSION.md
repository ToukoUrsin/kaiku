# Kaiku

**Punchline:** One pair of Rokoko gloves, 48 robots.

**Challenge:** The Robot Teleoperation Challenge by Rokoko

**Demo:** https://touko-ursin--kaiku.us-east.modal.run

## Description

Kaiku is Finnish for echo. One person sorts parts with a Unitree G1 through Rokoko Smartgloves, and 47 more G1s do the same job at the same moment. Each one has its own scene: the parts are somewhere else, turned differently, sometimes a different shape. So the motion is not copied. It is redone on every table, relative to the parts that are actually there, while the operator is still moving.

Robot learning runs out of demonstrations long before it runs out of anything else. From 16 demonstrations by the operator a policy sorted 0% of new scenes. From the same 16 plus their echoes it sorted 81%. 32 demonstrations with echoes beat 128 without. Then the gloves come off, and all 48 robots sort scenes they have never seen.

Every robot goes through the humanoid challenge's safety filter, and physics decides whether the parts really landed in the right bins. Only the clean runs are saved, in the humanoid challenge's LeRobot and GR00T format.

Open the demo and press Train to rerun that comparison live, or Policy to watch the robots work with nobody driving.

What is real: Rokoko's driver and hand solver read the gloves (tested with the driver's emulated Smartgloves II), and the humanoid challenge's own runner takes the lead robot's commands, 1,800 in a row through its safety filter with none clipped. What is not yet: it runs in a MuJoCo twin of the sorting station, the operator in the hosted demo is a scripted hand in emulated gloves, and the policy reads where the parts are from the simulator instead of cameras.

Redoing demonstrations in new scenes is known from MimicGen and DexMimicGen, where it runs afterwards on recordings. Kaiku does it live, from the glove stream, with physics and the safety filter deciding what counts.

## Technologies

Rokoko Smartgloves II, Rokoko Device SDK (RGMP), Unitree G1 with Dex3 hands, MuJoCo, Python, PyTorch, LeRobot v3.0, GR00T, three.js, Modal
