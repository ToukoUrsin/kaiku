# The glove sitting

One sitting with real Smartgloves gives everything else: the recording, the replay the hosted demo
plays, and the study on a person's own demonstrations. One glove is enough.

## What to bring

- Smartgloves II, one or two, and their USB cables. (Smartgloves I stream over Wi-Fi only: join them
  to the laptop's network in Rokoko Studio first.)
- A Coil Pro if there is one. Without it the glove gives the fingers and the trackpad gives the place.
- The laptop with the Rokoko Device SDK installed and signed in.

## The steps

1. Plug in the glove, and the Coil Pro if you have one. Switch the glove on and put it on.
2. In a terminal: `cd ~/kaiku && .venv/bin/python -m kaiku.app.gloves`
3. Watch the checklist tick itself. It starts Rokoko's driver and hand solver, waits up to 5 minutes
   for a glove, then prints what drives what and the page's address.
4. Open the address (http://127.0.0.1:8651/).
   - Two gloves on a Coil Pro: hold both hands flat and still until the page says it is calibrated.
     Your two hands are the robot's two hands.
   - One glove on a Coil Pro: hold the hand flat and still for 2 seconds. The robot's hand goes where
     yours goes and closes when yours closes.
   - One glove, no Coil Pro: press **Drive it yourself**. Point at a part with the trackpad, pinch
     thumb and index finger in the glove to pick it up, move over the bin of its colour, open the hand.
     **Use my hand** takes the place from the laptop camera instead of the trackpad.
5. Sort scenes until the page says **30 of 30**. A scene is three parts and takes 20 to 40 seconds, so
   the sitting is under 25 minutes. Fewer scenes still count; stop when you like.
6. Press Ctrl-C in the terminal. The session folder gets a `DONE` file.

Everything the driver sends is recorded from the moment the checklist is ticked, and every scene you
sort is saved with the robots that followed it.

## What follows by itself

`tools/sitting_watch.sh` on the box sees the `DONE` file within a minute, pulls the session and runs
`python -m kaiku.app.sitting` on it:

- It keeps only the scenes a gloved hand sorted. The stand-in's, a pointer's and a camera hand's
  scenes are not glove demonstrations and are left out.
- It makes the replay the hosted demo plays when nobody drives (`kaiku/sessions/human`).
- It runs the study in sandboxes, three seeds: a policy trained on the person's scenes alone against
  one trained on the same scenes with their echoes, each tested on 96 scenes neither has seen.
- It writes `SITTING.md` into the session: what was recorded, what the study measured, the deploy
  command, and which words and deck to use. A run that did not finish is named, never estimated.

## What the checklist can say

| Message | What to do |
|---|---|
| Install Rokoko Device SDK on this Mac | Install it, then run the command again. |
| Sign in to Rokoko on this Mac | Sign in, then run the command again. |
| Activate the Rokoko Device SDK | Activate it, then run the command again. |
| Renew the Rokoko licence or free a seat | Do that, then run the command again. |
| Start the Rokoko driver and leave it running | The command starts it itself; if this stays, another driver is holding the device: close Rokoko Studio. |
| Plug the glove into this Mac by USB, switch it on and put it on | It waits 5 minutes for the first frame. |
| The driver is open but no glove frames arrived for 3 seconds | Check the cable and that the glove is on. |
| The left glove's rate is too low | Move closer to its receiver, or use the cable. |
| Port 8651 is already in use | Nothing: it takes the next free port and prints it. |
| Scene dropped: the glove stream stopped | Nothing: that scene is thrown away and a new one is drawn. Keep the glove awake. |
| You have 30 of 30 clean scenes | Press Ctrl-C. |

## What has been tested

On the laptop with Rokoko's driver, Rokoko's hand solver and its emulated Smartglove II: the checklist,
the page in the person's hands from the first scene, the glove's pinch picking a part up, the
recording, stopping, the pull to the box, and the study in sandboxes. Rokoko's emulated glove holds
one pose, which Rokoko's hand solver reads as a pinch, so whole scenes sorted by a glove's pinch are
tested with Kaiku's own moving emulator. A session from an emulated glove is labelled emulated and
changes nothing.

Nobody has worn real Smartgloves with Kaiku yet.
