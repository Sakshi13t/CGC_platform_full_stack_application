# CGC Reliability & Condition-Monitoring Standard (RS-CGC-01)
*Synthetic internal standard — for assignment use only. Not a reproduction of any published standard.*

## 1. Scope
This standard defines condition-monitoring limits and reliability practices for the
cracked gas centrifugal compressor (CGC) trains CGC-100A and CGC-200B. It applies to
online monitoring, alarm management, and shutdown protection.

## 2. Vibration limits (shaft relative, peak-to-peak, microns)
| State | Level | Action |
|---|---|---|
| Normal | < 38 um | Routine monitoring |
| Alert | 38 – 50 um | Investigate within 24 h; increase sampling |
| Alarm | 50 – 62 um | Notify reliability engineer; plan intervention |
| Trip | >= 62 um | Automatic shutdown (2-out-of-3 voting) |

Axial (thrust) displacement alarm at 45 um, trip at 60 um. A sustained 1x amplitude
rise of more than 60% above the running baseline is treated as an incipient bearing
fault regardless of absolute level.

## 3. Bearing metal temperature
Alarm at 85 degC, trip at 95 degC. A rate-of-rise exceeding 3 degC/hour must be alarmed
even below the absolute threshold.

## 4. Lube-oil system
Supply pressure low alarm at 2.0 bar, trip at 1.8 bar. Supply temperature high alarm at
52 degC. Seal-gas differential pressure must remain above 0.40 bar at all times.

## 5. Performance monitoring
Polytropic efficiency and polytropic head shall be computed online from suction and
discharge pressure/temperature, gas molecular weight and flow. A drop of 4 percentage
points in polytropic efficiency below the post-wash clean baseline is the primary
performance-degradation trigger (see wash program RS-CGC-03).

## 6. Data quality
Sensors frozen for more than 30 minutes, or reading outside physical range, shall be
flagged BAD and excluded from performance calculations. Performance KPIs must never be
computed from a single instrument without a cross-check (e.g., discharge temperature
vs. predicted polytropic discharge temperature).
