# A real person's hands, as a camera sees them

`counting_left.jsonl` and `counting_right.jsonl` are the 21 hand landmarks of the two hands in
"Finger-counting in Dutch" by S. Perquin (Wikimedia Commons, CC BY-SA 4.0,
https://commons.wikimedia.org/wiki/File:Finger-counting_in_Dutch.webm): a fist that opens one finger
at a time, thumb first. One line per frame at 15 frames a second: `t` seconds, `label` the hand
MediaPipe called it, `world` the landmarks in metres.

Made with the hand model the page ships (`kaiku/web/static/vendor/mediapipe/hand_landmarker.task`),
each half of the picture cropped to one hand. The hands lie on a table and are seen from above,
which is a harder view for the model than a palm held up to a laptop.

These landmarks are shared under the same licence, CC BY-SA 4.0.
