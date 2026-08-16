# Tuning history: FACE_DAMPING_FACTOR

This isn't a guessed constant — it's been adjusted twice against real
observed data from live testing (toggling "Portrait dampening" on/off
against the same Google Images search results), not just theory. Recording
the actual numbers here so future changes have something concrete to check
against instead of re-guessing from scratch.

## Test case: Elon Musk portrait search results

Four studio-style headshots the base classifier confidently (and wrongly)
called AI-generated, with dampening off: **98%, 98%, 98%, 66%**.

| Factor | 98% cases become | 66% case becomes | Verdict shift |
|---|---|---|---|
| 0% (off) | 98% | 66% | all "Likely AI" |
| 0.25 (first attempt) | 75% | 66%→ ~50%\* | still "Likely AI" — confirmed the math was being applied correctly (proportional to the formula), but not a strong enough correction |
| 0.4 | 59% | 40% | all four land in "Uncertain" |
| **0.6 (current)** | **39%** | **26%** | the three 98% cases stay "Uncertain"; the 66% case flips to "Likely Real" |
| 0.65 | 34% | 23% | all four flip to "Likely Real" — rejected: this would fully clear even a 98%-confidence AI call, with no floor of caution left |

\* the 66% case wasn't independently re-measured at 0.25 in the live test: the reported values were the three 98%→75% cases plus one that read 66% *before* dampening was re-applied at that setting. Treat the 0.25 row as approximate for that specific number.

## Known cost of increasing this further

This dampener triggers on "a person fills the frame" — it cannot tell a
real photo of a person from a genuinely AI-generated one. Every increase
here trades reduced false positives on real portraits against reduced
sensitivity to true positives on synthetic ones (deepfakes, fake profile
photos). At 0.6, a 90%+ confidence AI call still can't be fully cleared to
"Likely Real" (it floors out at "Uncertain"), which was a deliberate
design choice to keep some margin here. Going further (e.g. to 0.65+)
removes that margin entirely.

If accuracy against genuinely AI-generated portraits ever needs to be
re-verified, the fastest check is: find or generate a handful of
realistic AI portraits (e.g. from a face-generation showcase site) and
confirm they still land as "Likely AI" or at minimum "Uncertain" with
dampening on, not "Likely Real".
