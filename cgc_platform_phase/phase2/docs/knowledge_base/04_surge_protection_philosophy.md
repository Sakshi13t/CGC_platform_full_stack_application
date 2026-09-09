# Surge Protection Philosophy (SP-CGC-04)
*Synthetic internal document — for assignment use only.*

## 1. What surge is
Surge is an aerodynamic flow reversal that occurs when flow through a centrifugal compressor
falls below the surge limit for the current head. It causes rapid, damaging axial and radial
vibration and thrust reversals and must be prevented, not merely detected.

## 2. Control lines
- Surge limit line (SLL): the measured onset of surge from the OEM map.
- Surge control line (SCL): set at 112% of the surge-limit flow. The anti-surge (recycle)
  valve begins to open as the operating point approaches the SCL.
- Minimum operating surge margin: 8% distance from the SLL.

## 3. Interaction with fouling
Fouling shifts the effective operating point toward surge because it narrows the flow path
and raises the required head. As efficiency degrades over a run, available surge margin
falls. A fouled machine at reduced flow is the highest-risk condition.

## 4. Automatic actions
If surge margin falls below 8%, the anti-surge valve opens to recycle gas and restore margin.
Recycling consumes energy, so persistent recycle at steady load is a symptom that either the
machine is fouled or throughput is too low for the current configuration.

## 5. Optimiser constraint
Any energy-optimisation or operating-point recommendation MUST hold surge margin at or above
the 8% minimum. A recommendation that reduces recycle or lowers speed must be rejected if it
would breach the surge margin constraint. This is a hard safety constraint and overrides any
energy-saving objective.
