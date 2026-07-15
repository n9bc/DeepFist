#!/usr/bin/env bash
# Watch Lyra Recordings for NEW, finished captures. Emits one line per recording
# once its audio_001.wav has stopped growing (i.e. Brent hit stop). Baseline =
# whatever exists at start, so only new sessions are announced.
REC="C:/Users/bcrie/Documents/Lyra/Recordings"
declare -A announced sizes
# Baseline: mark everything already present as announced (don't re-harvest old).
for d in "$REC"/*/; do announced["$(basename "$d")"]=1; done
echo "watching $REC  (baseline: ${#announced[@]} existing recordings ignored)"
while true; do
  for d in "$REC"/*/; do
    name="$(basename "$d")"
    wav="$d/audio_001.wav"
    [ -f "$wav" ] || continue
    [ -n "${announced[$name]}" ] && continue
    sz=$(stat -c%s "$wav" 2>/dev/null || echo 0)
    if [ "${sizes[$name]}" = "$sz" ] && [ "$sz" -gt 100000 ]; then
      announced["$name"]=1
      echo "READY $name ${sz} bytes"
    else
      sizes["$name"]=$sz
    fi
  done
  sleep 12
done
