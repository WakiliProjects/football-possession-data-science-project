#!/usr/bin/env python3
"""
MTAH 2027 - Phase 1: Ball Possession            Team Dataculer
=================================================================
Labels every frame of BAR-ATH as H / A / D from the provided tactical tracking.

Method in one paragraph
-----------------------
A three-state hidden Markov model (Home, Away, Dead) is decoded over the two
halves. Nothing is learned from ground truth. The emission tables are estimated
from the match itself, using frames whose state is physically certain
("anchors"): the ball is off the pitch, the ball is lost for seconds while every
player walks, the ball lies still for seconds (dead); or one team strings
passes together (alive). The Home/Away part of the table is fitted by EM under
a symmetry constraint (swapping the teams swaps the table). Decoding is followed
by a handful of rule-based corrections that encode the labelling conventions in
CHALLENGE.md (a single touch is not a change of possession, play restarts when
the ball leaves its resting spot, kick-offs).

Usage
-----
    python solve.py                       # writes Dataculer.csv next to this file
    python solve.py --data-dir /path/to/bundle --out my.csv
    python solve.py --score               # also print the public 5-minute check

Data is looked up in, in order: --data-dir, $MTAH_DATA_DIR, ./data, this folder
(each also with a tactical/ sub-folder, the layout of the official bundle).

Dependencies: numpy, pandas (see requirements.txt). Deterministic: no random
numbers, integer Viterbi, tables rounded before decoding.
"""

import argparse
import hashlib
import io
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Match constants (DATA.md / games.py)
# ---------------------------------------------------------------------------
GAME = "BAR-ATH"
N_FRAMES = 199_319
FPS = 25
PITCH_L, PITCH_W = 105.3, 68.0                 # metres
HALVES = [(11_185, 90_652), (110_808, 190_198)]  # graded video frames, half-open
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# ---------------------------------------------------------------------------
# Model constants. None of these was searched on the public ground truth: each
# is a physical quantity or a football convention, and section 5 of Dataculer.md
# shows how little the output moves when they are varied.
# ---------------------------------------------------------------------------
CFG = dict(
    # --- contact geometry -------------------------------------------------
    R=1.5,            # m   a player this close to the ball can play it
    R2=3.0,           # m   an opponent this close is "pressing" (contact is contested)
    far=6.0,          # m   ball further than this from everyone is loose / in flight
    bs_bins=[1.0, 3.0, 7.0, 12.0],   # m/s  ball speed: still / carried / rolled / passed / struck
    msp_bins=[0.8, 1.2, 1.6, 2.2, 3.0],  # m/s  median player speed: standing ... running
    # --- anchors (physically certain frames) --------------------------------
    oob=1.0,          # m   ball this far outside the lines ...
    oob_min=38,       # fr  ... for 1.5 s (or until it is lost) is out of play, not a high ball
    gap_min=100,      # fr  ball lost for at least 4 s ...
    walk=2.0,         # m/s ... while the median player walks (standard walking zone) = stoppage
    rest_speed=0.5,   # m/s ball slower than this ...
    rest_min=75,      # fr  ... for 3 s is placed for a restart
    touch_min=3,      # fr  shortest contact counted as a touch in a pass chain
    chain_min=3,      # touches of one team in a row, linked by passes = certainly alive
    pass_max=100,     # fr  longest pass (4 s)
    pass_speed=7.0,   # m/s a ball that has been kicked or thrown
    prior=5.0,        # Dirichlet pseudo-count for every table cell
    # --- HMM ----------------------------------------------------------------
    blk=5,            # fr  decision step (0.2 s)
    spell_s=10.0,     # s   round a-priori prior for how long a team keeps the ball
    #                       (general football knowledge, NOT this match); section 5: output
    #                       is identical for any value from 8 to 16 s
    switch_frac=0.7,  # a-priori prior: most possessions end in a turnover, not a stoppage
    dead_s=20.0,      # s   round a-priori prior for stoppage length (not from this match)
    em_iters=15,
    round_dec=4,      # tables are rounded before decoding (reproducibility)
    # --- kick-offs (Smart Tagging, confirmed by the ball) -------------------
    ko_lo=60, ko_hi=125,  # fr  the kick-off happens in this part of the tag window
    ko_radius=8.0,    # m   ... with the ball this close to the centre spot
    ko_pre=125,       # fr  the 5 s before a kick-off are dead
    ko_goal_link=100, # fr  a GOALS tag ending this close to a kick-off tag belongs to it
    goal_lag=50,      # fr  dead from 2 s after the GOALS tag starts
    ko_free=250,      # fr  after a kick-off, a tracking gap is not a stoppage anchor
    # --- conventions applied after decoding --------------------------------
    min_dead=100,     # fr  no restart takes less than 4 s (unless anchored)
    min_alive=50,     # fr  play between two stoppages lasts at least 2 s ...
    #                       ... and contains a kicked ball (pass_speed)
    min_spell=25,     # fr  under 1 s is a single touch, not a possession (CHALLENGE.md)
    snap=75,          # fr  search window when snapping a turnover to the first contact
    rs_back=75,       # fr  search window for the ball's resting spot before a restart
    rs_fwd=50,        # fr  search window for the ball leaving it
    rs_rest=1.5,      # m/s resting
    rs_go=3.0,        # m/s leaving
)


# ===========================================================================
# 1. Loading
# ===========================================================================

def find_data_dir(cli_dir=None):
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [cli_dir, os.environ.get("MTAH_DATA_DIR"), os.path.join(here, "data"), here]
    name = GAME + "_tactical_atd.txt"
    for root in roots:
        if not root:
            continue
        for folder in (os.path.join(root, "tactical"), root):
            if os.path.exists(os.path.join(folder, name)):
                return folder
    raise FileNotFoundError(
        "Cannot find %s. Put the tactical bundle in ./data/tactical, or pass --data-dir, "
        "or set MTAH_DATA_DIR." % name)


def load_team_slots(xml_path):
    """Track slot -> 0 (home, first team in the file) or 1 (away)."""
    root = ET.parse(xml_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in %s, found %d" % (xml_path, len(team_ids)))
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")])


def load_tracking(txt_path, n_slots):
    """ATD text -> player array (N, slots, 3) and ball array (N, 2); normalised units, NaN = missing.

    One line per frame:  frame:x,y,speed;x,y,speed;...:ball_x,ball_y,ball_z
    """
    with open(txt_path) as fh:
        raw = fh.read().translate(str.maketrans(":;", ",,"))
    table = pd.read_csv(io.StringIO(raw), header=None, dtype=np.float64).values
    del raw
    if table.shape[1] != 1 + 3 * n_slots + 3:
        raise ValueError("unexpected tracking layout: %d columns" % table.shape[1])
    rows = table[:, 0].astype(np.int64) - 1          # frame_id is 1-based
    keep = (rows >= 0) & (rows < N_FRAMES)
    players = np.full((N_FRAMES, n_slots, 3), np.nan)
    ball = np.full((N_FRAMES, 2), np.nan)
    players[rows[keep]] = table[keep, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball[rows[keep]] = table[keep, 1 + 3 * n_slots:3 + 3 * n_slots]
    return players, ball


def load_tags(xml_path):
    """Smart Tagging -> DataFrame(code, Team, Type, fs, fe) with frame = round(seconds * 25)."""
    rows = []
    for inst in ET.parse(xml_path).getroot().iter("instance"):
        row = {"code": (inst.findtext("code") or "").upper(), "Team": "", "Type": ""}
        for label in inst.findall("label"):
            group = (label.findtext("group") or "").strip().title()
            if group in ("Team", "Type"):
                row[group] = (label.findtext("text") or "").strip()
        row["fs"] = int(round(float(inst.findtext("start")) * FPS))
        row["fe"] = int(round(float(inst.findtext("end")) * FPS))
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# 2. Per-frame signals
# ===========================================================================

def runs(mask):
    """Start and stop (half-open) of every True run."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def goalkeeper_fix(px, team):
    """Per-frame home/away flag for every slot, with goalkeeper slots re-assigned by pitch side.

    The two slots that live next to a goal are the goalkeepers. The tracker regularly parks a
    goalkeeper slot on the *other* goalkeeper, which turns every pass by that keeper into a
    turnover. A keeper never leaves his own half, so the half he stands in decides his team.
    """
    is_home = np.broadcast_to(team == 0, px.shape).copy()
    for a, b in HALVES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.array([np.nanmedian(px[a:b, k]) if np.isfinite(px[a:b, k]).mean() > 0.5
                            else np.nan for k in range(px.shape[1])])
        keepers = np.flatnonzero(np.abs(med - PITCH_L / 2) > 30.0)
        home_keepers = [k for k in keepers if team[k] == 0]
        if not home_keepers:
            continue
        home_left = med[home_keepers[0]] < PITCH_L / 2
        for k in keepers:
            is_home[a:b, k] = (px[a:b, k] < PITCH_L / 2) == home_left
    return is_home


def signals(players, ball, team):
    px, py = players[:, :, 0] * PITCH_L, players[:, :, 1] * PITCH_W
    bx, by = ball[:, 0] * PITCH_L, ball[:, 1] * PITCH_W
    vis = np.isfinite(bx)
    is_home = goalkeeper_fix(px, team)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)
        dh = np.nanmin(np.where(is_home, dist, np.inf), axis=1)
        da = np.nanmin(np.where(~is_home, dist, np.inf), axis=1)
        msp = np.nanmedian(players[:, :, 2], axis=1)     # median player speed, m/s
    dh[~vis] = np.nan
    da[~vis] = np.nan
    bs = np.sqrt(np.gradient(bx) ** 2 + np.gradient(by) ** 2) * FPS   # ball speed, m/s
    return dict(bx=bx, by=by, vis=vis, dh=dh, da=da, bs=bs, msp=msp)


SWAP6 = [0, 1, 5, 4, 3, 2]       # contact category under a Home <-> Away swap


def features(S, cfg):
    """Discrete features, the anchors, and the per-frame contact category.

    contact: 0 nobody within R (ball near), 1 nobody within `far`,
             2 home clear, 3 home pressed, 4 away pressed, 5 away clear      (-1 = ball not tracked)
    """
    bx, by, vis, dh, da, bs, msp = (S[k] for k in ("bx", "by", "vis", "dh", "da", "bs", "msp"))
    n = N_FRAMES

    # ball outside the lines (a high ball projects outside for a moment; that is not "out")
    exc = np.maximum.reduce([-bx, bx - PITCH_L, -by, by - PITCH_W])
    exc[~vis] = -1.0
    out = np.zeros(n, bool)
    for a, b in zip(*runs(exc > 0)):
        if exc[a:b].max() >= cfg["oob"] and (b - a >= cfg["oob_min"] or (b < n and not vis[b])):
            out[a:b] = True

    # tracking gaps: length and how fast the players move meanwhile
    gap_len = np.zeros(n, np.int64)
    gap_speed = np.zeros(n)
    for a, b in zip(*runs(~vis)):
        gap_len[a:b] = b - a
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gap_speed[a:b] = np.nanmean(msp[a:b])

    R, R2 = cfg["R"], cfg["R2"]
    d = np.fmin(dh, da)
    c = np.full(n, -1)
    c[vis & (d >= R) & (d < cfg["far"])] = 0
    c[vis & (d >= cfg["far"])] = 1
    home_near = vis & (dh < R) & (dh <= da)
    away_near = vis & (da < R) & (da < dh)
    c[home_near & (da >= R2)] = 2
    c[home_near & (da < R2)] = 3
    c[away_near & (dh < R2)] = 4
    c[away_near & (dh >= R2)] = 5

    nb = len(cfg["bs_bins"]) + 1
    speed_bin = np.digitize(np.nan_to_num(bs, nan=0.0), cfg["bs_bins"])
    contact_speed = np.where(c >= 0, c * nb + speed_bin, -1)
    swap = [SWAP6[k // nb] * nb + k % nb for k in range(6 * nb)]
    msp_bin = np.digitize(np.nan_to_num(msp, nan=1.5), cfg["msp_bins"])

    # dead anchors
    rest = np.zeros(n, bool)
    for a, b in zip(*runs(vis & (exc <= 0) & (bs < cfg["rest_speed"]))):
        if b - a >= cfg["rest_min"]:
            rest[a:b] = True
    dead_anchor = out | ((gap_len >= cfg["gap_min"]) & (gap_speed < cfg["walk"])) | rest

    # alive anchors: pass chains of one team
    alive_anchor = np.zeros(n, bool)
    for own, opp in ((2, 5), (5, 2)):
        s, e = runs(c == own)
        keep = (e - s) >= cfg["touch_min"]
        s, e = s[keep], e[keep]
        other = (c == opp) | (c == 3) | (c == 4)
        chain = [0] if len(s) else []
        for i in range(1, len(s) + 1):
            linked = False
            if i < len(s):
                a, b = e[i - 1], s[i]
                linked = (b > a and (b - a) <= cfg["pass_max"] and vis[a:b].all()
                          and not out[a:b].any() and not other[a:b].any()
                          and np.nanmax(bs[a:b]) >= cfg["pass_speed"])
            if linked:
                chain.append(i)
            else:
                if len(chain) >= cfg["chain_min"]:
                    alive_anchor[s[chain[0]]:e[chain[-1]]] = True
                chain = [i]
    alive_anchor &= ~dead_anchor

    feats = {"contact": (contact_speed, 6 * nb, swap), "msp": (msp_bin, len(cfg["msp_bins"]) + 1, None)}
    return feats, dead_anchor, alive_anchor, c


def kickoffs(tags, S, dead_anchor, cfg):
    """Forced-dead and forced-alive frames around kick-offs that the ball confirms."""
    bx, by, vis = S["bx"], S["by"], S["vis"]
    force_dead = np.zeros(N_FRAMES, bool)
    force_alive = np.zeros(N_FRAMES, bool)
    force_alive[HALVES[0][0]:HALVES[0][0] + cfg["blk"]] = True    # first half is graded from kick-off
    goals = tags[tags.code == "GOALS"]
    for r in tags[(tags.code == "SET PIECES") & tags.Type.str.lower().str.startswith("kick off")].itertuples():
        lo, hi = r.fs + cfg["ko_lo"], r.fs + cfg["ko_hi"]
        if lo < 0 or hi > N_FRAMES:
            continue
        dist = np.hypot(bx[lo:hi] - PITCH_L / 2, by[lo:hi] - PITCH_W / 2)
        ok = vis[lo:hi] & (dist < cfg["ko_radius"])
        if not ok.any():
            continue                         # the tag is an inference; without the ball we ignore it
        ko = lo + int(np.argmin(np.where(ok, dist, np.inf)))
        half = [h for h in HALVES if h[0] <= ko < h[1]]
        if not half:
            continue
        start = ko - cfg["ko_pre"]
        if "start" in r.Type.lower():
            start = half[0][0]               # nothing is played before the half starts
        for g in goals.itertuples():
            if abs(g.fe - r.fs) <= cfg["ko_goal_link"]:
                start = min(start, g.fs + cfg["goal_lag"])
        start = max(start, half[0][0])
        force_dead[start:ko] = True
        force_alive[ko:ko + cfg["blk"]] = True
        dead_anchor[ko:ko + cfg["ko_free"]] = False
    return force_dead, force_alive


# ===========================================================================
# 3. Hidden Markov model
# ===========================================================================

def forward_backward(log_b, trans, pi):
    """Scaled forward-backward; returns state posteriors (T, K)."""
    T, K = log_b.shape
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    alpha = np.zeros((T, K))
    scale = np.zeros(T)
    a = pi * b[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ trans) * b[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t]
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = (trans @ (b[t + 1] * beta[t + 1])) / scale[t + 1]
    gamma = alpha * beta
    return gamma / gamma.sum(axis=1, keepdims=True)


def viterbi_int(score_b, score_a, score_pi):
    """Viterbi on integer scores: exact arithmetic, so ties break the same way on every machine."""
    T, K = score_b.shape
    delta = score_pi + score_b[0]
    back = np.zeros((T, K), np.int8)
    for t in range(1, T):
        cand = delta[:, None] + score_a
        back[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + score_b[t]
    path = np.zeros(T, np.int8)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


class PossessionHMM:
    NEG = -1e9

    def __init__(self, feats, dead_anchor, alive_anchor, force_dead, force_alive, cfg):
        self.cfg = cfg
        self.names = list(feats)
        self.meta = {k: (feats[k][1], feats[k][2]) for k in self.names}
        blk = cfg["blk"]
        graded = np.zeros(N_FRAMES, bool)
        self.seqs, self.dead, self.alive = [], [], []
        for a, b in HALVES:
            graded[a:b] = True
            idx = np.minimum(np.arange(a, b, blk) + blk // 2, b - 1)     # centre frame of each step
            self.seqs.append({k: feats[k][0][idx] for k in self.names})
            self.dead.append((dead_anchor | force_dead)[idx])
            self.alive.append(force_alive[idx])

        # transitions from the duration priors
        dt = blk / float(FPS)
        p_end = dt / cfg["spell_s"]
        p_switch = p_end * cfg["switch_frac"]
        p_dead = p_end - p_switch
        p_restart = dt / cfg["dead_s"]
        self.trans = np.array([[1 - p_end, p_switch, p_dead],
                               [p_switch, 1 - p_end, p_dead],
                               [p_restart / 2, p_restart / 2, 1 - p_restart]])
        self.pi = np.array([0.25, 0.25, 0.5])

        # alive-vs-dead tables from the anchors
        self.E = {}
        for k in self.names:
            n, swap = self.meta[k]
            x = feats[k][0]
            alive = np.bincount(x[alive_anchor & graded & ~force_dead & (x >= 0)], minlength=n) + cfg["prior"]
            dead = np.bincount(x[dead_anchor & graded & (x >= 0)], minlength=n) + cfg["prior"]
            if swap is not None:              # neither table may prefer a team
                alive = (alive + alive[swap]) / 2.0
                dead = (dead + dead[swap]) / 2.0
            alive, dead = alive / alive.sum(), dead / dead.sum()
            self.E[k] = np.array([alive, alive, dead])
        # starting point for EM: the team in contact probably has the ball
        nb = len(cfg["bs_bins"]) + 1
        lean = np.repeat(np.array([1.0, 1.0, 1.9, 1.4, 0.6, 0.1]), nb)
        home = self.E["contact"][0] * lean
        home /= home.sum()
        self.E["contact"][0] = home
        self.E["contact"][1] = home[self.meta["contact"][1]]

    def log_b(self, i, tables):
        seq, dead, alive = self.seqs[i], self.dead[i], self.alive[i]
        out = np.zeros((len(dead), 3))
        for k in self.names:
            x = seq[k]
            ok = x >= 0                        # an untracked ball says nothing
            out[ok] += np.log(tables[k])[:, x[ok]].T
        out[dead, H] = out[dead, A] = self.NEG
        out[alive & ~dead, D] = self.NEG
        return out

    def fit_team_table(self):
        """EM on the Home/Away rows of the contact table, tied by the team-swap symmetry."""
        n, swap = self.meta["contact"]
        for _ in range(self.cfg["em_iters"]):
            counts = np.full((2, n), self.cfg["prior"])
            for i, seq in enumerate(self.seqs):
                gamma = forward_backward(self.log_b(i, self.E), self.trans, self.pi)
                x = seq["contact"]
                ok = x >= 0
                for state in (H, A):
                    counts[state] += np.bincount(x[ok], weights=gamma[ok, state], minlength=n)
            home = counts[H] + counts[A][swap]
            home /= home.sum()
            self.E["contact"][H] = home
            self.E["contact"][A] = home[swap]
        return self

    def decode(self):
        dec = self.cfg["round_dec"]
        tables = {k: np.round(v, dec) + 10.0 ** -dec for k, v in self.E.items()}
        to_int = lambda x: np.round(np.maximum(x, -1e7) * 1000.0).astype(np.int64)
        score_a, score_pi = to_int(np.log(self.trans)), to_int(np.log(self.pi))
        labels = np.full(N_FRAMES, "D", dtype="<U1")       # outside the halves nothing is in play
        for i, (a, b) in enumerate(HALVES):
            path = viterbi_int(to_int(self.log_b(i, tables)), score_a, score_pi)
            labels[a:b] = np.repeat(LABELS[path], self.cfg["blk"])[:b - a]
        return labels


# ===========================================================================
# 4. Conventions applied after decoding
# ===========================================================================

def segments(labels, a, b):
    w = labels[a:b]
    cuts = np.flatnonzero(w[1:] != w[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(w)]))
    return [(a + int(bounds[i]), a + int(bounds[i + 1]), str(w[bounds[i]])) for i in range(len(bounds) - 1)]


def apply_conventions(labels, S, contact, anchored, locked, cfg):
    labels = labels.copy()
    bs, vis = S["bs"], S["vis"]
    home_touch = (contact == 2) | (contact == 3)
    away_touch = (contact == 4) | (contact == 5)

    for a, b in HALVES:
        for _ in range(4):
            changed = False
            # (i) play between two stoppages must last 2 s and contain a kicked ball
            segs = segments(labels, a, b)
            i = 0
            while i < len(segs):
                if segs[i][2] == "D":
                    i += 1
                    continue
                j = i
                while j < len(segs) and segs[j][2] != "D":
                    j += 1
                s0, e0 = segs[i][0], segs[j - 1][1]
                if i > 0 and j < len(segs) and not locked[s0:e0].any():
                    top = float(np.max(np.where(vis[s0:e0], np.nan_to_num(bs[s0:e0]), 0.0)))
                    if top < cfg["pass_speed"] or (e0 - s0) < cfg["min_alive"]:
                        labels[s0:e0] = "D"
                        changed = True
                i = j
            # (ii) a stoppage shorter than 4 s that no anchor supports is a tracking artefact
            segs = segments(labels, a, b)
            for k in range(1, len(segs) - 1):
                s0, e0, lab = segs[k]
                if lab == "D" and e0 - s0 < cfg["min_dead"] and anchored[s0:e0].mean() < 0.5:
                    labels[s0:e0] = segs[k - 1][2]
                    changed = True
            # (iii) a possession under 1 s inside the other team's possession is a single touch
            segs = segments(labels, a, b)
            for k in range(1, len(segs) - 1):
                s0, e0, lab = segs[k]
                if (lab != "D" and e0 - s0 < cfg["min_spell"] and segs[k - 1][2] == segs[k + 1][2]
                        and segs[k - 1][2] != "D" and not locked[s0:e0].any()):
                    labels[s0:e0] = segs[k - 1][2]
                    changed = True
            if not changed:
                break

        # (iv) a turnover happens when the new team first touches the ball
        segs = segments(labels, a, b)
        for k in range(len(segs) - 1):
            s0, e0, lab = segs[k]
            s1, e1, nxt = segs[k + 1]
            if lab == "D" or nxt == "D":
                continue
            old, new = (home_touch, away_touch) if lab == "H" else (away_touch, home_touch)
            lo, hi = max(s0 + 1, e0 - cfg["snap"]), min(e1 - 1, e0 + cfg["snap"])
            if hi <= lo:
                continue
            last_old = np.flatnonzero(old[lo:e0])
            start = lo + int(last_old[-1]) + 1 if len(last_old) else lo
            first_new = np.flatnonzero(new[start:hi])
            if not len(first_new):
                continue
            t = start + int(first_new[0])
            if locked[min(t, e0):max(t, e0) + 1].any():
                continue
            if t < e0:
                labels[t:e0] = nxt
            else:
                labels[e0:t] = lab

        # (v) play restarts when the ball leaves its resting spot
        segs = segments(labels, a, b)
        for k in range(len(segs) - 1):
            s0, e0, lab = segs[k]
            if lab != "D":
                continue
            lo, hi = max(s0 + 1, e0 - cfg["rs_back"]), min(segs[k + 1][1] - 1, e0 + cfg["rs_fwd"])
            if e0 <= lo or hi <= lo:
                continue
            resting = np.flatnonzero(vis[lo:e0] & (bs[lo:e0] < cfg["rs_rest"]))
            if not len(resting):
                continue
            r = lo + int(resting[-1])
            leaving = np.flatnonzero(vis[r:hi] & (bs[r:hi] >= cfg["rs_go"]))
            if not len(leaving):
                continue
            t = r + int(leaving[0])
            if locked[min(t, e0):max(t, e0) + 1].any():
                continue
            if t < e0:
                labels[t:e0] = segs[k + 1][2]
            else:
                labels[e0:t] = "D"
    return labels


# ===========================================================================
# 5. Pipeline
# ===========================================================================

def solve(data_dir, cfg=CFG, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    stem = os.path.join(data_dir, GAME + "_tactical")
    team = load_team_slots(stem + "_atd.xml")
    say("Reading tracking ...")
    players, ball = load_tracking(stem + "_atd.txt", len(team))
    tags = load_tags(stem + "_smart_tagging.xml")

    S = signals(players, ball, team)
    feats, dead_anchor, alive_anchor, contact = features(S, cfg)
    force_dead, force_alive = kickoffs(tags, S, dead_anchor, cfg)
    graded = np.zeros(N_FRAMES, bool)
    for a, b in HALVES:
        graded[a:b] = True
    say("  ball tracked on %.1f%% of graded frames" % (100 * S["vis"][graded].mean()))
    say("  anchors: dead %.1f%%, alive %.1f%% of graded frames"
        % (100 * (dead_anchor & graded).mean() / graded.mean(), 100 * (alive_anchor & graded).mean() / graded.mean()))

    say("Fitting and decoding the HMM ...")
    model = PossessionHMM(feats, dead_anchor, alive_anchor, force_dead, force_alive, cfg).fit_team_table()
    labels = model.decode()
    labels = apply_conventions(labels, S, contact, dead_anchor | force_dead, force_dead | force_alive, cfg)

    n_spells = sum(1 for a, b in HALVES for s in segments(labels, a, b) if s[2] != "D")
    n_dead = sum(1 for a, b in HALVES for s in segments(labels, a, b) if s[2] == "D")
    share = {lab: float((labels[graded] == lab).mean()) for lab in "HAD"}
    say("  %d possessions, %d stoppages; graded share H %.2f / A %.2f / D %.2f"
        % (n_spells, n_dead, share["H"], share["A"], share["D"]))
    return labels


def write_csv(path, labels):
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, lab) for i, lab in enumerate(labels))
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print("Wrote %d rows -> %s" % (len(labels), path))
    print("sha256 %s" % digest)
    return digest


def public_check(labels):
    """Score against the public 5 minutes with the official library. Verification only."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "eval"))
    import possession_metrics as pm
    gt_path = os.path.join(here, "eval", "gt", GAME + "_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    pred = pm.label_codes(pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}))
    res = pm.evaluate(gt, pred)
    print("Public 5-minute check (verification only, 17 possessions): "
          "spellF1 %.4f  macroF1 %.4f  alive/dead F1 %.4f"
          % (res["spell_f1_mean"], res["macro_f1"], res["alive_dead_f1"]))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Dataculer possession pipeline (MTAH 2027, phase 1)")
    ap.add_argument("--data-dir", default=None, help="folder holding the tactical bundle")
    ap.add_argument("--out", default=os.path.join(here, "Dataculer.csv"))
    ap.add_argument("--score", action="store_true", help="print the public 5-minute check")
    args = ap.parse_args()

    labels = solve(find_data_dir(args.data_dir))
    write_csv(args.out, labels)
    if args.score:
        public_check(labels)


if __name__ == "__main__":
    main()
