# Incident Report Log (IR-CGC-07)
*Synthetic incident history — for assignment use only.*

## IR-2024-014  CGC-100A  Anti-surge excursion during feed upset
During a rapid feed-rate reduction the operating point crossed the surge control line before the
anti-surge valve fully opened, producing four short surge cycles with axial and radial vibration
spikes. No mechanical damage was found. Root cause: recycle valve stroke time slower than the
rate-of-change of flow during the upset. Corrective action: retune anti-surge controller for
faster opening and add a rate-of-change feed-forward. Lesson: fouled machines at reduced flow
have the least surge margin and are most exposed to fast upsets.

## IR-2024-031  CGC-100A  Stage-3 discharge temperature calibration drift
Performance monitoring flagged a rising stage-3 discharge temperature not matched by an efficiency
change. Cross-check against the predicted polytropic discharge temperature showed the transmitter
had drifted about 9 degC over several weeks. Root cause: transmitter calibration drift. Lesson:
never trust a single performance instrument; always cross-check against a physics-based estimate.

## IR-2024-052  CGC-200B  High-vibration bearing trip
Drive-end 1x vibration climbed over roughly six weeks from baseline to the trip level; the train
tripped on high vibration and was down six days for bearing replacement. The rise was gradual and
was visible in trend data well before the trip. Root cause: drive-end journal bearing wear.
Lesson: a slow 1x growth is an actionable early warning; a condition-based alert could have
converted this unplanned trip into a planned intervention.

## IR-2024-048  CGC-100A  Interstage cooler fouling
Cooling-water-side fouling of an interstage cooler raised interstage gas temperatures, increasing
downstream compression duty and specific energy by about 3% for several weeks until the cooler was
cleaned. Lesson: energy losses are not always in the compressor itself; interstage cooling matters.
