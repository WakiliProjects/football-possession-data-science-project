# Dataculer: Technical Document

**More than a Hack 2027, Phase 1: Ball Possession**
**Team:** Dataculer · **Submission file:** `Dataculer.csv` · **Code:** https://github.com/WakiliProjects/football-possession-data-science

---

## 1. The approach

### What we built

A three-state hidden Markov model (Home, Away and Dead) decoded over each half of the match,
followed by five rules that apply the labelling conventions of `CHALLENGE.md`. The whole
pipeline is `solve.py`, needs only numpy and pandas, runs in about 8 seconds and uses no
random numbers.

```
tracking (ATD) -> per-frame signals -> anchors -> HMM tables -> Viterbi -> conventions -> Dataculer.csv
                  ball-to-player dist   frames     alive/dead    H / A / D   single touch,
                  ball speed            whose       from anchors; every       restart timing,
                  median player speed   state is    Home vs Away  0.2 s       kick-offs
                  goalkeeper fix        certain     by EM
```

### Why this design

The ranking metric counts whole possessions, so the output has to be a segmentation rather than
a per-frame vote. Three facts about the data shaped the model.

**1. The ball is not missing at random.** Inside the graded frames the ball track has only 150
gaps, and 133 of them are longer than 4 seconds. The tracker bridges short losses itself, so
what remains are long holes, and most are stoppages: the ball is in a player's hands, off the
pitch or outside the camera's interest. A gap is therefore evidence about dead or alive, not
about which team.

**2. Half of the dead time has a visible ball.** In the public window, two of the four stoppages
have the ball tracked throughout as it rolls away, is carried back and is placed. A "no ball
means dead" rule, like the starter baseline, cannot see these. They
are recognisable from the combination of signals: nobody near a slow ball, players walking and
a ball lying still.

**3. Goalkeeper tracks carry the wrong team 11% of the time.** The two track slots that live next
to a goal are the goalkeepers, but the tracker regularly parks the away-keeper slot on the home
keeper and the reverse. Each time, a back-pass to the keeper looks like a turnover. A keeper does
not leave his own half, so we assign those two slots by pitch side rather than by slot. This
affects 11.3% of graded frames.

A hidden Markov model fits this problem. The three states are exactly the three labels, the
transition matrix expresses that possessions last seconds rather than frames, and the emission
tables weigh several weak signals at once.

### No ground truth in the fit

We did not want a model fitted to 17 possessions. The emission tables are estimated from the
match itself, from frames whose state follows from physics. We call these the anchors.

| Anchor | Definition | State |
|---|---|---|
| Ball out | Ball at least 1 m outside the lines for 1.5 s, or until it is lost | Dead |
| Stoppage gap | Ball untracked for 4 s or more while the median player walks (under 2 m/s) | Dead |
| Ball at rest | Ball inside the pitch moving under 0.5 m/s for 3 s | Dead |
| Pass chain | Three or more consecutive touches by one team, linked by passes (ball above 7 m/s, no opponent touch between) | Alive |

Dead anchors cover 41.4% of the graded frames, and alive anchors cover 11.7%. The alive and dead
tables are the feature histograms over these two sets. Dead anchors are also hard constraints
during decoding. The first two rows of the table show why the signals separate the states: a ball
more than 6 m from every player takes 8.5% of alive frames and 45.7% of dead ones, and a median
player speed under 1.2 m/s takes 15% of alive frames and 65% of dead ones.

The Home and Away split cannot come from anchors, so it is fitted by EM: 15 iterations of
forward-backward over the full match, under one constraint, that swapping the teams swaps the
table. EM settles on a team in possession being in clear contact with the ball 33% of the time
and in contact under pressure 10.5% of the time, while its opponent touches it 2.3% of the time.

The transition matrix is not fitted. It is set from three round priors chosen in advance from
general football knowledge, not from this match: a possession of about 10 s, a stoppage of about
20 s and most possessions (about 70%) ending in a turnover rather than a stoppage. Section 5 shows
the output does not change across a wide range of all three.

### Features

Each 0.2 s step emits two discrete features.

- **Contact by ball speed** (30 values). Contact is one of: nobody within 1.5 m, nobody within
  6 m, home clear, home under pressure (an opponent within 3 m), away under pressure or away
  clear. Ball speed is binned at 1, 3, 7 and 12 m/s. The pairing matters, because a ball far
  from everyone is a long pass when fast and a stoppage when slow.
- **Median player speed** (6 values, binned at 0.8, 1.2, 1.6, 2.2 and 3.0 m/s).

When the ball is not tracked the first feature contributes nothing, so the state is carried by
player speed and by the transition matrix. Viterbi runs on integer scores, and the tables are
rounded to four decimals first, so the output does not depend on floating-point details.

---

## 2. Assets used and how

| Asset | Use |
|---|---|
| `BAR-ATH_tactical_atd.txt` | The input. Ball position, player positions and speeds for all 46 track slots. |
| `BAR-ATH_tactical_atd.xml` | Maps each track slot to a team. |
| `BAR-ATH_tactical_smart_tagging.xml` | Kick-offs only (6 tags), each accepted only if the ball is tracked within 8 m of the centre spot in the tag window. Five are confirmed. The sixth, tagged at frame 62012 with no ball near the centre, is ignored. A `GOALS` tag ending at a confirmed kick-off marks the time between goal and kick-off as dead. |
| `eval/gt/BAR-ATH_possession_gt_public.csv` | Verification only. We used it to align frames, to read the labelling convention so we understood what counts as possession and to run the format and score checks in section 5. No table, constant or rule was fitted, tuned or searched against it. `Dataculer.csv` is produced from the tracking and tagging alone, and the ground-truth file is read only by the optional `--score` check. |

**Tried and dropped**

| What | Why it was dropped |
|---|---|
| Smart Tagging team tags (Progression, Build Up, Transitions and so on) as evidence for the team in possession | In the public window these tags name the team in possession on 76% of the live frames they cover (73% for Progression). Contact evidence alone is right on 92% of live frames there, so the tags would add noise rather than information. |
| Smart Tagging set-piece windows as dead-ball evidence | Every window is exactly 35 s, the restart falls anywhere inside it, and only 59 set pieces are tagged against the 114 stoppages we find. We use them as an independent check instead (section 5). |
| Unconstrained EM on all tables | It converges to the wrong states. Dead becomes "ball not tracked" in one variant and "players slow" in another, with 65% of frames dead. The anchors prevent this. |
| Homography | The tracking is already in pitch coordinates. Not used. |
| Broadcast tracking | On its own timeline with cuts and replays, and no mapping to the tactical frames. Not used. |
| Video | Not used. We never checked the output against the video. |
| The ball's third column in the ATD file | Present on 10% of rows, mostly before kick-off, and uncorrelated with ball speed. Ignored. |

No external data, pre-trained model or LLM is part of the pipeline. No frame was labelled by
hand.

---

## 3. Possession rules implemented

**Control.** A player can play the ball when it is within 1.5 m of him. Contact is clear when no
opponent is within 3 m of the ball, and pressed otherwise. Pressed contact is weaker evidence,
because under pressure the tracked distances cannot tell the carrier from the challenger.

**Loose balls and passes in flight.** With nobody within 1.5 m, the feature is identical for Home
and Away, so the model has no reason to switch, and the team that last had the ball keeps it. This
is the convention in `CHALLENGE.md`.

**Challenges, deflections and single touches.** A change of possession needs sustained evidence.
With the fitted tables, roughly two 0.2 s steps of clear opponent contact outweigh the cost of a
switch, and more when the contact is pressed. In addition, a possession shorter than 1 s that sits
inside the other team's possession is treated as a single touch and removed.

**When a turnover happens.** The boundary is placed on the first frame the new team is in contact
with the ball after the old team's last contact, which is the reception rather than the pass.

**Dead ball.** A stoppage is what the HMM decodes as Dead: mostly the anchors (ball out, a long
gap with walking players or a ball at rest) extended to the frames around them where the ball is
slow, the ball is far from players and the players are walking. Dead-ball ownership is never
labelled. Two rules clean the result:

- a stoppage shorter than 4 s that no anchor supports is discarded, since a high ball that
  projects far from every player can look like one;
- play between two stoppages must last 2 s and contain a ball faster than 7 m/s, because a player
  walking the ball to the corner flag is not a possession.

**Restarts.** Play resumes when the ball leaves its resting spot. Within 3 s before the decoded
restart we find the last frame with the ball under 1.5 m/s and move the restart to the first
frame after it above 3 m/s. If the ball only reappears at the restart, as with most throw-ins,
the decoded time stands. The team that restarts is whichever team the model sees in contact
first, and it is not taken from the tags.

**Kick-offs.** A confirmed kick-off tag fixes the restart on the frame where the ball is nearest
the centre spot. Everything from the start of the second half to its kick-off is Dead (the
decoded kick-off is frame 113743). The first half is treated as live from its first graded frame.

**Ungraded frames** (before the match, at half-time and after the final whistle) are written as
`D`. We never write `X`, because the model always has a best guess, and an abstention is worth no
more than repeating the previous label.

---

## 4. Limitations

**Short possessions.** On the public window the ground truth has 17 possessions, and we report 12.
Of the six we fail to match, four are under 3.6 s (19, 33, 71 and 90 frames), and the tracking
does not separate them from touches the ground truth ignores: a 7-frame home touch counts as a
possession at frame 15358, while an 18-frame one at 15272 does not. We chose not to chase them.
Across the match this costs recall on brief possessions and merges their neighbours.

**High balls.** The ball is projected onto the pitch plane, so a ball in the air appears metres
away from the player who heads or receives it. Aerial duels and long clearances are the main
source of late or missed turnovers. The first turnover of the match, at frame 11351, is found 115
frames late for this reason.

**Team identity of outfield tracks.** We repair the goalkeepers only. Outfield slots also swap
identity when players cross, and a swap between opponents flips the contact evidence until the
track ends. We did not find a reliable way to detect this without the video.

**Stoppages with no anchor.** A foul followed by a quick free kick, with the ball tracked and
moving the whole time, leaves little to see. In the other direction, a goalkeeper holding the ball
while his track is lost looks like a ball lying far from everyone. The start of a stoppage is
located less precisely than its end: in the public window the four starts are off by 2, 11, 21 and 75 frames, and the four restarts by 2, 9, 17 and 34.

**Slow restarts.** After a short goal kick or a back-pass restart, the players are still walking
and the ball is slow, so the model can stay in Dead one pass too long. This is the 34-frame error
above.

**One match.** Every constant is a physical quantity set in advance, and section 5 shows the
output barely moves when they change, but none of it has been tested on a second match.

**With more time** we would use the video to estimate ball height, or at least detect flight from
the shape of the projected trajectory; detect outfield identity swaps from formation consistency;
align the broadcast tracking to recover the ball where the tactical tracker loses it; and replace
the fixed-rate transitions with explicit duration distributions.

---

## 5. Verification

All numbers below come from the official `run_evaluation.py` and from `solve.py`.

**Public 5-minute window** (7,500 frames, 17 possessions):

| | spellF1 | macroF1 | F1 H / A / D | alive/dead F1 | transition F1 (within 1 s) |
|---|---|---|---|---|---|
| This pipeline | 0.5517 | 0.9177 | 0.913 / 0.900 / 0.941 | 0.9859 | 0.457 |

The scorer's bootstrap interval for our spellF1 on this window is 0.36 to 0.73, and one
possession moves it by about 0.06, so a single short possession swings it. The frame-level columns
are far less noisy. This window is only 7,500 frames and we used it as a check, so the full-match
score may well be lower.

**An independent check over the full match.** Smart Tagging lists 59 set pieces. The model does
not use them, apart from kick-offs, so they can test it, since a tagged set piece should contain a
restart. Counting a Dead-to-alive transition between 10 s before the tag window and its end, our
output has one for 53 of the 59 set pieces.

**Full-match output.** 241 possessions (mean 13.0 s) and 114 stoppages. Graded frames are 33%
Home, 17% Away and 51% Dead. That is about 52 minutes of ball in play out of 106 graded, and a
66 to 34 split of live time.

**Sensitivity.** Each constant was changed on its own and the pipeline re-run. Agreement is the
share of graded frames with the same label as the submitted file. Self-spellF1 is the official
spellF1 of the variant scored against the submitted file, over the full match.

| Change | Possessions | Stoppages | Agreement | Self-spellF1 |
|---|---|---|---|---|
| submitted | 241 | 114 | 1.000 | 1.000 |
| decision step 4 / 6 / 8 frames (5) | 263 / 231 / 208 | 114 / 111 / 110 | 0.979 / 0.980 / 0.977 | 0.89 / 0.90 / 0.87 |
| contact radius 1.2 / 1.8 m (1.5) | 226 / 277 | 114 / 112 | 0.980 / 0.980 | 0.92 / 0.87 |
| pressure radius 2.5 / 4 m (3) | 243 / 248 | 113 / 111 | 0.996 / 0.990 | 0.97 / 0.94 |
| loose-ball distance 5 / 8 m (6) | 242 / 240 | 114 / 114 | 0.996 / 0.995 | 0.99 / 0.98 |
| walking speed 1.7 / 2.3 m/s (2.0) | 239 / 245 | 112 / 116 | 0.995 / 0.992 | 0.99 / 0.97 |
| stoppage gap 3 / 6 s (4) | 241 / 238 | 114 / 110 | 1.000 / 0.994 | 1.00 / 0.99 |
| ball-out margin 0.5 / 2 m (1) | 241 / 240 | 115 / 109 | 0.997 / 0.993 | 0.99 / 0.97 |
| mean possession 8 / 16 s (10) | 251 / 235 | 114 / 114 | 0.995 / 0.999 | 0.96 / 0.98 |
| mean stoppage 15 / 30 s (20) | 241 / 241 | 114 / 114 | 1.000 / 1.000 | 1.00 / 1.00 |
| turnover share 0.6 / 0.8 (0.7) | 237 / 249 | 114 / 114 | 0.999 / 0.996 | 0.99 / 0.97 |
| shortest stoppage 3 / 5 s (4) | 245 / 238 | 122 / 110 | 0.995 / 0.997 | 0.98 / 0.98 |
| single-touch rule off / 1.5 s (1) | 263 / 229 | 114 / 114 | 0.997 / 0.998 | 0.95 / 0.97 |
| table pseudo-count 1 / 20 (5) | 255 / 228 | 113 / 114 | 0.992 / 0.993 | 0.95 / 0.96 |

The output depends mostly on two choices, the decision step and the contact radius, and even
there at least 97% of frames keep their label.

---

## 6. Reproduction

```bash
pip install -r requirements.txt          # numpy 2.2.6, pandas 2.3.3; Python 3.10+
# put the tactical bundle in data/tactical/ (or pass --data-dir, or set MTAH_DATA_DIR)
python solve.py                          # writes Dataculer.csv, prints its SHA-256
python solve.py --score                  # same, plus the public-window check
cd eval && python validate_submission.py ../Dataculer.csv
```

Expected:

```
sha256 4c62badea9c381ccb0df958dc8aa691ac59eddbace3dc4161ae7ff752a9d0d37
OK    Dataculer.csv   199319 rows, frame 0..199318
```

We obtained this hash on two machines: Linux with Python 3.13, numpy 2.5.3 and pandas 3.0.5, and
Linux with Python 3.10, numpy 2.2.6 and pandas 2.3.3.

Files read: `BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_smart_tagging.xml`. Nothing else from the bundle is needed.
