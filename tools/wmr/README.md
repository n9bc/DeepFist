# WebMorseRunner labeled-data generator

Generates realistic labeled contest CW clips from the WebMorseRunner engine
(real callsigns + validated keying) for training DeepFist.

## Requires
A WebMorseRunner clone at `c:\dev\WebMorseRunner` (public domain / Unlicense).
These `.mjs` are copies for provenance; run them **from the WebMorseRunner clone**
(they import its `station.js`, `keyer.js`, `recording.js`, and read `calls.txt`).

## Use
```
cd c:\dev\WebMorseRunner
node gen_dataset.mjs --n 8000 --out C:/dev/DeepFist/runs/wmr_data
```
Produces `clip_N.wav` (6 s @ 11025 Hz) + `labels.jsonl` ({file,text,meta}).
Contest utterances (CQ/answer/exchange/TU/repeat), WPX serial / WW zone /
ARRL state+power exchanges, cut numbers, plus QRM/QSB/QRN/AWGN.

## Train with it
```
scripts/train.py --wmr runs/wmr_data --wmr-prob 0.5 --out runs/exp5 ...
```
Blends WMR clips (resampled 11025->3200) with the on-the-fly numpy generator.
