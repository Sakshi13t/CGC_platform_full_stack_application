# Maintenance Strategy & RCM Summary (MS-CGC-06)
*Synthetic internal document — for assignment use only.*

## 1. Strategy
Maintenance is condition-based, driven by online monitoring (RS-CGC-01) and performance
monitoring (RS-CGC-03), supplemented by time-based tasks during turnarounds.

## 2. Run-length objective
The reliability objective is to maximise run length between interventions while never breaching
a protective limit. "Run length" is measured in operating hours since the last online wash (for
performance) and hours since the last mechanical intervention (for reliability).

## 3. Intervention types
| Type | Trigger | Downtime |
|---|---|---|
| Online wash | Performance triggers in RS-CGC-03 | Hours (rate cut, not full shutdown) |
| Offline cleaning | Online wash recovers < 60% | 2 - 4 days |
| Bearing replacement | Vibration alarm/trip, 1x growth | 4 - 7 days (unplanned if reactive) |
| Major overhaul | Turnaround schedule | Weeks |

## 4. Remaining useful life (RUL)
RUL is estimated for two failure modes: (a) performance RUL — hours until a wash trigger is
forecast to be met; (b) mechanical RUL — hours until a vibration or bearing-temperature limit is
forecast to be reached. The shorter of the two governs planning.

## 5. Planning rule
Interventions are planned to convert unplanned shutdowns into planned ones. An unplanned trip
(for example a high-vibration bearing trip) costs several times more than a planned intervention
of the same scope, because of lost production and collateral damage.
