# Dataculer — MTAH 2027, Phase 1: Ball Possession

Submission of team **Dataculer** to [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(Metrica Sports / FC Barcelona / Mobile World Capital Barcelona).

`solve.py` labels every frame of BAR-ATH as `H` (home in possession), `A` (away in possession)
or `D` (dead ball) from the provided tactical tracking, and writes `Dataculer.csv`.

| File | What it is |
|---|---|
| `solve.py` | The whole pipeline. One command regenerates the submission. |
| `Dataculer.csv` | The submitted possession file (199,319 rows). |
| `Dataculer.md` | Technical document: approach, assets, possession rules, limitations. |
| `requirements.txt` | Pinned dependencies. |
| `eval/` | Copy of the official scorer and validator, for convenience. |

## Method in short

A three-state hidden Markov model (Home / Away / Dead) decoded over each half. No ground truth
is used to fit it: its tables are estimated from frames of the match whose state is physically
certain (ball off the pitch, ball lost while everybody walks, ball lying still; or one team
stringing passes together). A few rules then apply the labelling conventions of the challenge
(a single touch is not a possession, play restarts when the ball leaves its spot, kick-offs).
Details are in [`Dataculer.md`](Dataculer.md).

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Needs Python 3.10 or newer. Only `numpy` and `pandas` are needed.

## Data

The match data is licensed separately and is **not** in this repository. After enrolling, unpack
the bundle so that the tactical files sit in `data/tactical/`:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

`solve.py` also finds them in `data/`, next to `solve.py`, under `$MTAH_DATA_DIR` or wherever
`--data-dir` points. The video, the homography and the broadcast files are not used.

## Regenerate the submission

```bash
python solve.py
```

This writes `Dataculer.csv` and prints its SHA-256. The pipeline has no random component and
decodes with integer arithmetic, so the file is the same on every run:

```
sha256 4c62badea9c381ccb0df958dc8aa691ac59eddbace3dc4161ae7ff752a9d0d37
```

Runtime is about 10 seconds; peak memory is about 1.4 GB.

## Validate and score

```bash
cd eval
python validate_submission.py ../Dataculer.csv        # format check
cp ../Dataculer.csv results/
python run_evaluation.py --gt public                  # public 5-minute window
```

`python solve.py --score` prints the same public-window numbers directly.

Public 5-minute window (17 possessions; used to verify the pipeline, not to tune it):

| | spellF1 | macroF1 | alive/dead F1 |
|---|---|---|---|
| This pipeline | 0.5517 | 0.9177 | 0.9859 |
| Our first version (nearest player + smoothing) | 0.5176 | 0.6655 | 0.8959 |

One possession is worth about 0.06 spellF1 on this window, so the spellF1 column cannot
separate the two; the frame-level columns can.

## Notes

The `eval/` folder is a copy of the organisers' evaluation code from the competition repository.
The match data is not included and must not be committed.
