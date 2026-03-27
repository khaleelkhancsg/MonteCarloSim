#!/usr/bin/env python3
"""
Prop Firm Monte Carlo Simulator  ·  Enhanced Edition v2
═══════════════════════════════════════════════════════════
Tabs:
  1. Monte Carlo     — core simulation with inline info panels
  2. Walk Forward    — IS/OOS robustness testing & regime stress test
  3. Sensitivity     — tornado chart, parameter curves, 2-D heatmap

Requirements:  pip install numpy matplotlib scipy
Run:           python prop_sim.py
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.ticker
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math

# ══════════════════════════════════════════════════════════════════
# CHALLENGE CONSTANTS
# ══════════════════════════════════════════════════════════════════
DD_LIMIT         = 2_000.0
PROFIT_TARGET    = 3_000.0
DAILY_LOSS_LIM   = 400.0
QUAL_MIN         = 150.0
QUAL_DAYS_NEED   = 5
MAX_TRADES_DAY   = 15
MAX_DAYS         = 30
MAX_CONTRACTS    = 20
BASE_POINT_VALUE = 5.0

# ══════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════
C = {
    "bg":      "#FAFAF8", "panel":   "#F1EFE8", "border":  "#D3D1C7",
    "text":    "#2C2C2A", "muted":   "#73726c", "win":     "#3B6D11",
    "bust":    "#A32D2D", "timeout": "#5F5E5A", "blue":    "#185FA5",
    "blue_l":  "#B5D4F4", "amber":   "#854F0B", "purple":  "#5B2D8E",
    "teal":    "#1A7070", "gold":    "#9A7B10",
}

# ══════════════════════════════════════════════════════════════════
# INFO TEXTS — shown via [ℹ] buttons throughout the UI
# ══════════════════════════════════════════════════════════════════
INFO: Dict[str, str] = {

"monte_carlo_tab": """MONTE CARLO SIMULATION — WHAT & WHY
══════════════════════════════════════

Monte Carlo simulation runs thousands of random trials using the same
underlying rules, then aggregates outcomes into probability distributions.

In this prop-firm context:
  Each run = one 30-day challenge attempt
  Parameters = your strategy's statistical edge
  Outcomes   = pass / bust / timeout

Why it matters for prop firms:
  Even a genuine positive-edge strategy can fail a challenge due to bad
  luck alone. Monte Carlo quantifies that probability so you can choose
  sizing and risk geometry that shifts the odds in your favour.

Core challenge rules simulated:
  • $3,000 profit target
  • $2,000 trailing drawdown limit (tracks from peak equity)
  • $400 daily loss limit (session stops when hit)
  • 5 qualifying days required (daily P&L ≥ $150 each)
  • 30-day hard timeout
  • Maximum 20 contracts per trade

Recommended workflow:
  1. Pick a strategy preset matching your approach
  2. Calibrate win rate / avg win / avg loss to honest estimates
  3. Set realistic commission + slippage
  4. Run 1,000–2,000 simulations
  5. Use the Contracts sweep chart to find optimal sizing
  6. Validate robustness in Walk Forward tab
  7. Identify fragile parameters in Sensitivity tab""",

"equity_paths": """SAMPLE EQUITY PATH CHART
══════════════════════════════════════

Each line is one simulated challenge attempt: cumulative P&L plotted
day-by-day from Day 1 through to the outcome.

Colour key:
  Green  → Pass    — hit $3,000 profit target with ≥5 qualifying days
  Red    → Bust    — trailing drawdown exceeded the $2,000 limit
  Grey   → Timeout — 30 days elapsed without pass or bust

Reading patterns:
  Steep green paths, early target → strong, well-sized edge
  Red paths clustering days 1–5   → over-leveraged (too many contracts)
  Flat grey mass                  → EV near zero, or qualifying-day
                                    constraint is the bottleneck
  Wide spread of all outcomes     → high-variance strategy; add more
                                    simulations for stable statistics

Reference lines:
  Dashed green at +$3,000 — profit target
  Dashed red at   –$2,000 — trailing drawdown bust threshold
                            (measured from peak equity, not start)""",

"outcome_dist": """OUTCOME DISTRIBUTION CHART
══════════════════════════════════════

Shows what fraction of all Monte Carlo runs ended in each outcome.

Pass %    — probability of completing the challenge within 30 days
Bust %    — probability of blowing the $2,000 trailing drawdown limit
Timeout % — ran all 30 days without a decisive result

Benchmarks to target:
  Pass   > 40%  → solid, repeatable edge for this challenge structure
  Bust   < 25%  → well-controlled drawdown risk
  Timeout < 20% → strategy generates enough activity to resolve within 30 days

Common failure modes:
  Bust > 50%     : Over-leveraged; reduce contracts or tighten sizing rule
  Timeout > 60%  : Too conservative; increase trades/day or contract count
  Pass   < 20%   : Edge is weak or qualifying-day constraint is binding

Key insight:
  Raising aggression simultaneously increases pass AND bust rates.
  The contracts sweep chart below shows the optimal balance point.""",

"risk_geom_chart": """RISK GEOMETRY — SIZE vs BUFFER CHART
══════════════════════════════════════

Shows how each sizing method translates the remaining drawdown buffer
into a position-size multiplier.

X-axis: remaining drawdown buffer as % of $2,000 (100% = intact, 0% = cliff)
Y-axis: position-size multiplier applied to your chosen contract count

All five methods are plotted. Your active method is highlighted in bold.

Shaded red zone (0–25% buffer): the "danger zone" — under $500 remaining,
any normal-sized losing trade can end the attempt. Size must be near zero.

Curve shapes:
  Flat at 1.0 (Fixed)       — ignores the cliff entirely; most dangerous
  Straight line (Linear)    — proportional; transparent and simple
  Concave (Half Kelly)       — stays near full size longer, fades gracefully
  Power-law (Risk Geometry) — shape tunable via k exponent (your pick)
  Rising into loss (Martingale) — maximum bust risk; for reference only

The geometry of this curve is one of the most important levers available.
Use the main simulation to tune it: lower k if bust rate is too high.""",

"days_chart": """DAYS TO OUTCOME CHART
══════════════════════════════════════

Histogram of how many days elapsed before each simulation resolved.

Green bars — distribution of days-to-pass
Red bars   — distribution of days-to-bust

What to aim for:
  Pass median around days 12–18   → comfortable margin before 30-day cutoff
  Bust median > day 8             → losses develop slowly (not catastrophic)
  Clear separation between pass/bust distributions → decisive edge

Warning signs:
  Pass and bust heavily overlapping  → mostly random outcome (no real edge)
  Passes concentrated days 25–30    → timeout-limited; qualifying-day
                                       constraint is likely the bottleneck
  Busts before day 5                → severe over-leverage; cut contracts

The vertical dashed line = average days to pass.""",

"daily_pnl": """DAILY P&L DISTRIBUTION CHART
══════════════════════════════════════

Histogram of every individual day's P&L across all simulations combined.

Statistics shown on chart:
  mean — average daily return
  std  — day-to-day variability

Why each statistic matters:

1. The $150 qualifying-day hurdle:
   You need at least 5 days with P&L ≥ $150. If mean daily P&L ≈ $150,
   you will barely accumulate these — you want comfortable margin above it.

2. The $400 daily loss limit:
   Acts as a hard truncation of the left tail. Days that would have been
   larger losses are clipped here. Visible as a wall in the histogram.

3. Distribution shape:
   Tight right-skewed distribution (most mass $0–$300, thin right tail)
   is ideal. Wide symmetrical distributions mean single days can swing
   the entire attempt outcome.

Low std + mean well above $150 = sustainable, challenge-friendly profile.""",

"sweep_chart": """CONTRACTS SENSITIVITY CHART (1–20)
══════════════════════════════════════

How pass rate, bust rate, and mean final P&L change as contract count
scales from 1 to 20, with all other parameters held fixed.

Three lines:
  Green  — pass rate (%) — left axis
  Red    — bust rate (%) — left axis
  Amber  — mean final P&L ($) — right axis

Key insight:
  There is a contract count where (pass rate − bust rate) is maximised.
  Beyond this point, variance grows faster than expected value, and bust
  rate overtakes pass rate.

Typical patterns:
  Green peaks at 1–4 contracts: very narrow edge; scale slowly
  Green peaks at 5–10 contracts: healthy edge with reasonable scale
  Both lines rise steeply together: variance is the dominant driver

Action: identify the peak of the green line. Use that as your sizing
ceiling. Cross-reference with Walk Forward tab to confirm OOS robustness.

Note: uses fast mini-simulation (~200 runs/point) for speed.""",

"futures_mode": """MICRO FUTURES — INSTRUMENT CONTEXT
══════════════════════════════════════

Micro futures give index-futures exposure at 1/10th of the full contract
notional — ideal for prop-firm challenge sizing.

MES — Micro E-mini S&P 500
  Dollar value : $5.00 per full index point
  Tick size    : 0.25 index points = $1.25 per tick
  Character    : Tracks S&P 500. Relatively smooth intraday moves.
  Best suited  : Range trading, mean reversion, pairs strategies

MNQ — Micro E-mini Nasdaq-100
  Dollar value : $2.00 per full index point
  Tick size    : 0.25 index points = $0.50 per tick
  Character    : Higher beta, faster/wider intraday swings.
  Best suited  : Momentum, breakout, trend-following strategies

Calibration note:
  The Avg win / Avg loss sliders are in TRUE DOLLARS per contract.
  Enter the actual dollar amount you make or lose on a typical winning
  or losing trade, as recorded in your trade log — for whichever
  instrument you have selected.

  MES examples (point value $5.00):
    10-point winner = $50  →  set avg win to 50
    8-point loser   = $40  →  set avg loss to 40

  MNQ examples (point value $2.00):
    15-point winner = $30  →  set avg win to 30
    10-point loser  = $20  →  set avg loss to 20

  Important: switching instrument does NOT auto-scale the sliders.
  Always re-enter aw/al in the dollar terms of your chosen instrument.

Maximum cap: 20 contracts per trade (prop-firm hard limit in simulator).""",

"trade_params": """TRADE PARAMETERS — CALIBRATION GUIDE
══════════════════════════════════════

Win Rate (%):
  Hit rate — the fraction of trades that close profitably.
  25–45% : Breakout / momentum (requires high avg win / avg loss ratio)
  45–60% : Trend-following / swing (balanced systems)
  60–75% : Mean-reversion scalping (requires tight losses)
  75%+   : High-frequency micro-scalp (very narrow edge per trade)

  Reality check: live win rates typically run 3–8% below backtested
  rates due to adverse selection, execution slippage, and curve-fitting.

Avg Win per Contract ($):
  Mean dollar profit on winning trades at 1 contract.
  Sets your payoff asymmetry alongside Avg Loss.

Avg Loss per Contract ($):
  Mean dollar loss on losing trades at 1 contract.
  Risk/Reward ratio = Avg Win / Avg Loss.
  A win rate of 40% needs R/R ≥ 1.5 to stay positive EV.

Trades per Day:
  Expected trade entries per session.
  Too few  → timeout risk (not enough time for 5 qualifying days)
  Too many → $400 daily limit hit, cutting sessions short

Live EV formula (shown in real time):
  EV = win_rate × avg_win − (1−win_rate) × avg_loss − friction
  Negative EV = no sizing rule can create a long-run winning system.""",

"exec_friction": """EXECUTION FRICTION
══════════════════════════════════════

Models the real cost of entering and exiting trades.

Commission (round-turn, $):
  All-in brokerage cost per contract per trade — opening + closing combined.
  Typical ranges for micro futures:
    $0.50–$0.80 : competitive prop-firm / discount broker rate
    $0.80–$1.50 : standard retail broker
    $1.50–$2.50 : legacy platforms or high-minimum accounts

Slippage (ticks):
  Average ticks between intended fill price and actual fill received.
  0.0 ticks : limit-order only, liquid session (optimistic)
  0.25 ticks : realistic for market orders in liquid MES/MNQ
  0.5–1.0   : faster strategies or volatile session opens
  2.0+      : aggressive entries in thin books or news events

Combined friction example (MES, 1 contract):
  Commission: $0.74
  Slippage:   0.25 ticks × $1.25/tick = $0.31
  Total:      $1.05 per trade

At 8 trades/day × 20 days = $168 total friction per attempt.
That is 5.6% of the $3,000 target — material at this account size.

Rule of thumb: always use a friction estimate that is slightly too high.
Underestimating friction is the most common error in strategy modelling.""",

"position_sizing": """POSITION SIZING — RISK GEOMETRY SYSTEM
══════════════════════════════════════

Controls how exposure scales with your remaining $2,000 drawdown buffer.
Buffer = $2,000 − (peak equity − current equity).

Why this matters more than most traders realise:
  With $100 buffer left, a single normal-sized losing trade ends the
  attempt. Smart sizing prevents this cliff-edge failure mode.

Methods:
  Fixed size
    Constant full size throughout. Simple, but blind to the cliff.
    Only recommended when bust rate is naturally very low.

  Linear scale-down
    size = buffer / $2,000   (50% buffer → 50% size)
    Proportional, transparent, and easy to reason about.

  Half Kelly
    size = 0.5 + 0.5 × (buffer / $2,000)
    Blends half fixed + half proportional. Smoother fade.

  Risk Geometry  ← recommended starting point
    size = (buffer / $2,000) ^ (1 / k)
    k < 1 : aggressive early reduction (curve bends down quickly)
    k = 1 : same as linear
    k > 1 : gradual until near cliff, then sharper cut

  Martingale
    Increases size as losses mount. Maximum bust risk.
    Included for stress-testing purposes only — do not use live.

Starting recommendation: Risk Geometry with k = 1.5
Tune by watching bust rate in the main simulation.""",

"wf_main": """WALK FORWARD TESTING — OVERVIEW
══════════════════════════════════════

Walk Forward Testing (WFT) is the gold standard method for assessing
whether a strategy's edge is genuine or an artefact of regime-specific
over-fitting.

How this simulator implements WFT:
  Using a parameter-based model (no historical price data), WFT is
  implemented as a parameter-stability test across N folds:

  In-Sample (IS) per fold:
    Simulate with current parameters ± small random fold noise.
    Represents: your "optimised" parameters for that sub-period.

  Out-of-Sample (OOS) per fold:
    Same fold parameters + a systematic degradation factor, modelling
    the well-documented IS→OOS performance decay seen in live trading.

Walk Forward Efficiency (WFE):
  WFE = mean OOS pass rate / mean IS pass rate
  
  WFE ≥ 0.80  : Excellent — edge transfers well to unseen conditions
  0.65–0.80   : Good — acceptable degradation
  0.50–0.65   : Marginal — parameters likely over-tuned to one regime
  < 0.50      : Poor — highly regime-sensitive; simplify the strategy

Stability score:
  Measures consistency of OOS results across folds.
  Low stability = path-dependent, regime-sensitive strategy.
  High stability = edge is repeatable regardless of sub-period.""",

"regime_stress": """REGIME STRESS TEST
══════════════════════════════════════

Tests your current strategy parameters against three stylised market
regime shifts, plus your baseline for direct comparison.

Regime definitions:

  Trending (bull/bear) day
    Win rate +5pp, avg win +20%, trades per day −1
    Models: persistent directional moves, open-drive days, trend days

  Choppy / Mean-reverting day
    Win rate +3pp, avg win −15%, avg loss +10%, trades per day +2
    Models: tight-range sessions, fading overextension, scalp conditions

  Volatile / Whipsaw day
    Win rate −8pp, avg win +30%, avg loss +25%
    Models: news events, gap-and-go sessions, high-volatility days

  Baseline  (your current settings — reference bar)

Interpretation:
  A robust strategy maintains pass rate > 30% across all three regimes.
  If performance collapses in one regime, you need a live regime-
  detection filter to disable trading in adverse conditions.

  Trending-regime collapse     → strategy depends on directional flow
  Choppy-regime collapse       → strategy struggles without clean trends
  Volatile-regime collapse     → strategy is hurt by unpredictable swings""",

"sensitivity_main": """PARAMETER SENSITIVITY — OVERVIEW
══════════════════════════════════════

Quantifies how much each input parameter affects your pass rate.
Answers the critical question: "which parameters do I need to nail?"

Why sensitivity analysis matters for live deployment:
  1. Identifies load-bearing parameters — small estimation errors here
     translate to large changes in real-world pass rate.
  2. Reveals the "robust region" — the parameter combinations where
     pass rate remains high despite some uncertainty.
  3. Guides calibration priorities — tells you where to invest effort
     gathering better real-market data from your strategy.

Three tools available:

  Tornado chart
    Ranks ALL parameters by their impact on pass rate.
    One horizontal bar per parameter, sorted widest to narrowest.
    Widest bar = most critical parameter to get right.

  Sensitivity curve (bottom left)
    Full pass-rate curve for a single selected parameter.
    Shows optima, cliffs, and flat safe regions.

  2-D Heatmap (bottom right)
    Pass rate surface for two parameters varying simultaneously.
    Reveals interactions and the width of the "green island"
    (the viable parameter space).""",

"tornado": """TORNADO CHART — PARAMETER IMPACT RANKING
══════════════════════════════════════

Each parameter is varied from LOW to HIGH while all others are
held at their current baseline value.

Bar = total change in pass rate (pass rate at HIGH minus at LOW).
Sort order = largest absolute impact at top (tornado funnel shape).

Colour:
  Green portion — pass rate at the HIGH parameter value
  Red portion   — pass rate at the LOW parameter value
  Centre (0)    — baseline pass rate (current settings)

Wide bar → highly leveraged parameter; be conservative in estimation.
Narrow bar → modelling freedom; pass rate is stable here.

Typical findings across strategy types:
  Win rate is almost always the widest bar (most impactful)
  Avg win and avg loss are typically next
  Commission + slippage are often surprisingly impactful at micro size
  Trades/day affects qualifying-day accumulation most strongly

Action: focus live calibration and data gathering on the top 2–3 bars.""",

"sens_curve": """SINGLE-PARAMETER SENSITIVITY CURVE
══════════════════════════════════════

Full pass-rate curve as the selected parameter sweeps from its minimum
to maximum value. All other parameters held fixed at baseline.

Reading the chart:
  Steep slope   → fragile — a 5% estimation error is costly here
  Flat region   → robust  — you have slack in this parameter
  Local maximum → optimal point exists; tuning beyond it hurts
  Monotone rise → "more is always better" up to the simulator limit

Reference lines:
  Vertical dashed   = your current baseline setting
  Horizontal dashed = 40% pass rate benchmark

Practical example:
  If the win rate curve drops steeply from 55% → 45% win rate,
  then deploying with a live win rate 5% below your backtest estimate
  cuts pass rate by 15–20 percentage points. This justifies conservative
  win rate assumptions in your planning model.

  Conversely, if the commission curve is nearly flat, you have flexibility
  on broker selection without materially affecting outcomes.""",

"regime_select": """MARKET REGIME FILTER
══════════════════════════════════════

Applies a regime overlay to the Monte Carlo simulation, adjusting your
base trade parameters to reflect a specific type of market session.

WHY THIS MATTERS:
  Your win rate, avg win, and avg loss are not constant — they shift
  with market character. A strategy calibrated on mixed conditions
  may perform very differently on a trending day vs. a choppy day.
  The regime filter lets you stress-test or target a specific session type.

REGIME OPTIONS:

  No filter (mixed)
    No adjustment. Your slider values are used exactly as set.
    Use this for a neutral, regime-agnostic estimate.

  Trending (bull/bear drive day)
    Win rate +5pp  |  Avg win ×1.2  |  Trades/day −1
    Models: open-drive sessions, news-continuation, trend days.
    Directional strategies typically shine here.

  Choppy (mean-reverting / range day)
    Win rate +3pp  |  Avg win ×0.85  |  Avg loss ×1.1  |  Trades/day +2
    Models: tight-range chop, fade-the-move sessions, grinding days.
    More trades but smaller winners; stop-runs inflate losses slightly.

  Volatile (whipsaw / event day)
    Win rate −8pp  |  Avg win ×1.3  |  Avg loss ×1.25
    Models: FOMC, CPI, earnings gaps. Both wins and losses are larger,
    but lower hit rate due to unpredictable noise.

  Custom blend
    Define a probability-weighted mix of the four session types.
    Each day in the simulation independently samples a regime from
    your blend weights — more realistic than a single fixed regime.
    Example: 30% Trending, 40% Choppy, 20% Volatile, 10% Baseline
    reflects a realistic distribution of session types.

EFFECTIVE PARAMETERS:
  When a fixed regime is selected, the "Effective params" strip shows
  the actual values being passed to the simulator after regime adjustment.
  The original slider values are preserved — switching back to
  "No filter" restores them instantly.

INTERACTION WITH WALK FORWARD:
  The regime filter applies only to the MC tab simulation.
  The Walk Forward Regime Stress Test (WF tab) independently cycles
  through all three regimes and compares them — it does not use this
  filter setting.""",

"heatmap": """2-D SENSITIVITY HEATMAP
══════════════════════════════════════

A grid showing pass rate for every combination of two parameters.
Each cell's colour = pass rate at that (x, y) combination.

Colour scale:
  Dark green  → high pass rate (target region)
  Amber / tan → moderate pass rate
  Dark red    → low pass rate (avoid)
  White star  (★) → your current baseline parameter settings

Why joint analysis matters:
  Parameters interact in non-obvious ways:
  • Higher win rate can compensate for lower avg win
  • More contracts may require higher win rate to avoid excess busting
  • Single-variable sweeps miss these trade-offs entirely

How to use the heatmap:
  1. Pick the two parameters you are most uncertain about
  2. Run the heatmap (10×10 grid, 5–10 seconds compute)
  3. Find the "green island" — the viable region of parameter space
  4. Locate the white star within the island
  5. Measure island width:
       Wide green region  → robust strategy; uncertainty is acceptable
       Narrow green region → fragile strategy; calibration must be precise
  6. If the star is near the edge of the green island, you are near
     a performance cliff — recalibrate toward the island centre.""",

# ─── Walk Forward chart-level info ───────────────────────────────

"wf_folds": """IS vs OOS PASS RATE BY FOLD
══════════════════════════════════════

Grouped bars showing In-Sample (IS) and Out-of-Sample (OOS) pass
rates for each walk-forward fold.

Blue bars  = IS pass rate — your strategy's performance on the training
             window (parameters tuned to this period ± fold noise).
Amber bars = OOS pass rate — performance on the unseen test window,
             with a systematic degradation penalty applied.

How to read this chart:
  • OOS bar close to IS bar → good generalisation for that fold
  • OOS bar much lower      → that fold's IS parameters don't carry over
  • Consistent height across folds → stable, regime-independent edge
  • Erratic OOS variation   → path-dependent or lucky IS optimisation

Target: OOS bars consistently above 30–40% across all folds.
The 40% dashed reference line marks the benchmark pass rate.""",

"wf_wfe": """WALK FORWARD EFFICIENCY (WFE) GAUGE
══════════════════════════════════════

The WFE gauge summarises the entire walk-forward test in one number.

WFE = mean OOS pass rate ÷ mean IS pass rate

The pointer shows your WFE value. Colour zones:
  🟢 Excellent  (≥0.80) — OOS retains ≥80% of IS performance
  🟡 Good       (0.65–0.80) — acceptable decay for most strategies
  🟠 Marginal   (0.50–0.65) — significant IS→OOS decay; re-examine
  🔴 Poor       (<0.50)     — edge dissolves out-of-sample

Stability score (shown below gauge):
  Measures the coefficient of variation (CV) of OOS fold results.
  High stability → OOS results are consistent across folds
  Low stability  → highly variable; may depend on regime luck

Statistics shown:
  Mean IS  — average in-sample pass rate across all folds
  Mean OOS — average out-of-sample pass rate across all folds
  Stability — 1 − CV(OOS rates), where 1.0 = perfectly consistent""",

"wf_equity": """IS vs OOS EQUITY CURVES
══════════════════════════════════════

Sample equity paths drawn from the walk-forward simulation, showing
how in-sample and out-of-sample attempts unfold day by day.

Colour key:
  Solid green     — IS run that passed the challenge
  Solid red       — IS run that busted
  Dashed amber    — OOS run that passed
  Dashed pink/red — OOS run that busted

Reference lines:
  Dashed green at +$3,000 — profit target
  Dashed red at  –$2,000  — trailing drawdown limit

What to look for:
  OOS passes reaching the target before day 30 → edge is real
  OOS paths clustering tighter than IS         → reduced variance OOS
    (usually because OOS degradation reduces effective size)
  OOS paths barely reaching target             → marginal edge OOS
  OOS dominated by busts                       → strategy is IS-tuned""",

"wf_degradation": """IS→OOS DEGRADATION BY FOLD
══════════════════════════════════════

Bar height = IS pass rate minus OOS pass rate for each fold.
Positive values mean OOS underperformed IS (expected and normal).

Colour zones:
  🟢 Green  (≤15pp gap)  — acceptable degradation; edge transfers well
  🟠 Amber  (15–30pp gap) — notable decay; monitor live performance closely
  🔴 Red    (>30pp gap)   — high IS→OOS decay; strategy may be over-fitted
    to specific conditions that won't repeat

The 15pp warning line is drawn as a dashed amber reference.

If all bars are red: your strategy may be highly dependent on having
  "perfect" parameter values. Use the Sensitivity tab to find a more
  robust parameter region where performance is stable across a wider
  range of inputs.

Negative bars (rare): OOS outperformed IS — possible if degradation
  parameter was set to zero or if positive regime coincided with OOS.""",

"wf_stability": """OOS PASS RATE — FOLD-BY-FOLD STABILITY
══════════════════════════════════════

Line chart of OOS pass rates across walk-forward folds, with a shaded
±1 standard deviation band around the OOS line.

Lines:
  Amber solid   — OOS pass rate per fold (the critical line)
  Blue dashed   — IS pass rate per fold (reference)
  Shaded band   — ±1 std dev of OOS rates (consistency envelope)
  Dotted lines  — mean IS and mean OOS (horizontal averages)

What stability means:
  Tight shaded band + flat line → consistent edge across all periods
  Wide band + volatile line     → performance depends on which period
                                   you happen to trade in

The OOS std deviation is shown in the bottom-right corner.
A std dev below 10pp is generally considered stable.

Ideal pattern: amber line flat and well above 30%, narrow band.""",

# ─── Sensitivity tab chart-level info ────────────────────────────

"sens_summary": """SENSITIVITY SUMMARY — IMPACT BAR CHART
══════════════════════════════════════

A ranked bar chart showing the absolute pass rate swing for each
parameter (from its low value to its high value).

Bar length = |pass rate at HIGH − pass rate at LOW|, in percentage
             points. Longer bar = greater leverage on your outcome.

Colour coding by relative impact:
  🔴 Red    — high impact (top tier; >70% of max bar length)
              Critical inputs: small estimation errors are costly here.
  🟠 Amber  — medium impact (35–70% of max)
              Important but not fatal if slightly miscalibrated.
  🟢 Green  — low impact (<35% of max)
              Robust inputs: strategy is insensitive here.

This chart is automatically populated when you run the Tornado Chart.
It shows the same data as the tornado, but framed as absolute impact
rather than a centred deviation from baseline — making it easier to
assess which parameters deserve the most calibration effort.""",


"alg_kelly": """KELLY SIZING FRONTIER
══════════════════════════════════════
Shows the analytically estimated pass rate vs contract count.
Blue line = Gaussian CLT approximation (corrected for DD constraint).
Amber dots = actual Monte Carlo validation at checked contract counts.
Green dashed = Kelly-recommended contract count (applied Kelly fraction × full Kelly × DD buffer / avg loss).
Use this chart to see the marginal return of adding contracts beyond the Kelly recommendation.""",

"alg_beven": """BREAK-EVEN MAP  (win rate vs R/R)
══════════════════════════════════════
2-D contour of normalised EV = wr * R/R − (1−wr).
Dark green = strongly positive EV. Dark red = negative EV.
The black contour line is the break-even boundary where EV = 0.
★ = your current win rate and R/R.
Use this map to visualise how far you are from negative EV territory,
and what win rate / R/R combination would keep you viable if either degrades.""",

"alg_qual": """QUALIFYING DAY PROBABILITY  (binomial)
══════════════════════════════════════
P(accumulating ≥5 qualifying days within k available days).
Based on the binomial distribution: P(≥5 | k, p_qual)
where p_qual = P(daily P&L ≥ $150) estimated via CLT normal approximation.
The vertical dashed line = days-to-target at expected value rate.
If the curve is still below 80% at the days-to-target point, qualifying days
are a binding constraint — focus on improving EV/day above $150.""",

"alg_ror": """RISK OF RUIN  (continuous approximation)
══════════════════════════════════════
R(f) = exp(−2 × EV × T / σ²)  where f = fraction of DD buffer risked per trade.
Derived from the diffusion approximation to Brownian motion with drift.
The dashed line = Kelly-recommended fraction.
The amber dotted line = 5% RoR threshold (commonly used maximum acceptable level).
Points to the right of where RoR > 5% are over-leveraged for this strategy's edge quality.""",

"alg_wfe_map": """WFE SENSITIVITY MAP  (noise vs degradation)
══════════════════════════════════════
Contour plot of estimated Walk Forward Efficiency vs two key inputs:
  X axis = fold parameter noise % (how much parameters vary across IS folds)
  Y axis = IS→OOS degradation % (the systematic OOS performance decay)
Analytical formula: WFE ≈ (1 − deg×0.8) / (1 + noise×0.5)
White contour lines mark WFE = 0.80 (Excellent), 0.65 (Good), 0.50 (Marginal).
★ = recommended operating point (10% noise, 15% degradation).
Use this to understand which combination of noise and degradation your strategy can tolerate.""",

"alg_frontier": """RISK-RETURN FRONTIER  (by contract count)
══════════════════════════════════════
Each point = a contract count (1–15c) plotted as (σ/day, EV/day).
Moving right and up = more risk AND more expected return.
The green highlighted point = Kelly-recommended contracts.
The efficiency frontier shows where the Sharpe proxy (EV/σ) is maximised.
Points past the Kelly-recommended count have diminishing Sharpe ratios — you
are taking on disproportionately more variance than incremental EV gain.
Optimal sizing lives near the bend in the frontier curve.""",
"sens_dist": """HEATMAP PASS RATE DISTRIBUTION
══════════════════════════════════════

Histogram of all cell values in the most recently run 2-D heatmap.
Each cell in the heatmap grid contributes one bar here.

What this shows:
  • The distribution of pass rates across all parameter combinations
  • How concentrated outcomes are vs. how spread they are

Reference lines:
  Amber dashed — mean pass rate across all cells
  Green dotted — 40% pass rate benchmark

Key metric (shown top-right):
  "X% of cells ≥ 40% pass rate" — the fraction of the parameter
  space where your strategy crosses the benchmark threshold.

Interpretation:
  High % (>60%) → large viable region; strategy is robust to parameter
                  uncertainty — you don't need perfect calibration.
  Low % (<30%)  → narrow viable region; a small mis-estimation could
                  push you into the red zone on the heatmap.
  Wide histogram → performance varies a lot across the parameter space.
  Narrow histogram → relatively uniform pass rate (good or bad globally).""",

"calibration_confidence": """CALIBRATION CONFIDENCE  —  Why this matters
══════════════════════════════════════

The simulation answers two very different questions depending on this setting:

  Confidence OFF (None):
    P(pass | win rate is EXACTLY 68%, avg win is EXACTLY $44, ...)
    This gives 90–100% pass rates for any positive-EV strategy because
    it assumes you know your parameters with perfect precision. No real
    trader does. This is the textbook Monte Carlo answer.

  Confidence ON:
    P(pass | win rate is APPROXIMATELY 68% ± uncertainty, ...)
    Each individual simulation run samples its own win rate, avg win,
    and avg loss from a distribution around your slider values.
    This is the MARGINAL probability, integrating over parameter
    uncertainty — the statistically correct question to ask.

Why does parameter uncertainty reduce pass rates so dramatically?
  A strategy with wr=68% that works EXACTLY produces a daily Sharpe
  so high that ruin is almost impossible. But if there is a 15%
  chance your live win rate is actually 55–60% (due to regime change,
  overfitting, or execution slippage), those bad runs contribute
  significant bust and timeout mass to the distribution.

Confidence levels (σ = one standard deviation):
  None        No uncertainty — exact parameters (textbook)
  Tight       ±2pp win rate, ±10% aw/al  (500+ live trades calibrating)
  Typical     ±5pp win rate, ±20% aw/al  (100–300 live trades)
  Wide        ±10pp win rate, ±35% aw/al  (early-stage, limited data)
  Conservative ±15pp win rate, ±50% aw/al  (backtested only, no live data)

Rule of thumb: if you have not traded this strategy live for at least
100 trades, use Wide or Conservative. Tight is only appropriate when
you have extensive live data that matches your parameter estimates.""",
}

# ══════════════════════════════════════════════════════════════════
# CONTRACT SPECS
# ══════════════════════════════════════════════════════════════════
CONTRACT_SPECS: Dict[str, dict] = {
    "MES": {
        "name": "Micro E-mini S&P 500", "symbol": "MES",
        "point_value": 5.0, "tick_size": 0.25, "tick_value": 1.25,
        "contract_note": "1/10 of ES notional exposure",
        "context": "Smoother equity curves; good for mean-reversion and range strategies.",
    },
    "MNQ": {
        "name": "Micro E-mini Nasdaq-100", "symbol": "MNQ",
        "point_value": 2.0, "tick_size": 0.25, "tick_value": 0.50,
        "contract_note": "1/10 of NQ notional exposure",
        "context": "Higher beta, faster swings; good for momentum and breakout strategies.",
    },
}

# ══════════════════════════════════════════════════════════════════
# STRATEGY / SIZING LIBRARIES
# ══════════════════════════════════════════════════════════════════
STRATEGY_LIBRARY: Dict[str, dict] = {
    "Kalman pairs  (MES/MNQ)": {
        "type": "statistical arbitrage / mean reversion",
        "what_it_entails": "Trades a spread between correlated instruments using a Kalman-filtered relationship.",
        "win_rate_band": "about 50–60% in many examples",
        "risk_profile": "Moderate hit-rate, R/R ~1.8, spread risk, execution-sensitive",
        "best_regime": "Stable correlation, mean-reverting intraday conditions",
        "notes": "Preset: wr=60%, avg win $100, avg loss $55 (R/R 1.82), 4 trades/day. Pairs systems can achieve good R/R by holding winners into full mean reversion.",
    },
    "EMA crossover  (5-min)": {
        "type": "trend-following / momentum",
        "what_it_entails": "Fast EMA crossing a slower EMA to define long/short direction.",
        "win_rate_band": "typically 40–50%; context filters improve this",
        "risk_profile": "Whipsaw-prone in chop; high R/R (≥2:1) compensates the lower hit rate",
        "best_regime": "Directional sessions with follow-through",
        "notes": "Preset calibrated: wr=44%, avg win $145, avg loss $68 (R/R 2.1:1), 8 trades/day. False signals rise sharply in sideways markets.",
    },
    "Mean-reversion scalp": {
        "type": "high-frequency mean reversion",
        "what_it_entails": "Fades stretched moves back toward a short-term average.",
        "win_rate_band": "often 60–75% when regime selection is good",
        "risk_profile": "High hit-rate, small and tight trades; loses edge fast under slippage",
        "best_regime": "Range compression, overextension, liquidity pockets",
        "notes": "Preset calibrated: wr=68%, avg win $44, avg loss $36 (R/R 1.2:1), 12 trades/day. Friction cost is ~6% of gross EV — use tight slippage estimate.",
    },
    "Momentum breakout": {
        "type": "breakout / trend continuation",
        "what_it_entails": "Triggers when price escapes a consolidation or compression range.",
        "win_rate_band": "roughly 25–40%",
        "risk_profile": "Low hit-rate, high asymmetry, large tail winners",
        "best_regime": "Volatility expansion, opening range breaks, strong news/trend days",
        "notes": "Needs enough average winner size to overcome the low hit-rate.",
    },
    "Range trading": {
        "type": "support / resistance mean reversion",
        "what_it_entails": "Buys near support and sells near resistance inside a defined range.",
        "win_rate_band": "often 55–70% when the range is genuine",
        "risk_profile": "Moderate hit-rate, tight stops, higher trade frequency",
        "best_regime": "Sideways, mean-reverting conditions with clean boundaries",
        "notes": "Preset: wr=65%, avg win $55, avg loss $42 (R/R 1.31), 9 trades/day. Higher frequency compensates the tighter edge. Needs regime filter when market trends.",
    },
    "Custom": {
        "type": "user-defined",
        "what_it_entails": "Placeholder for your own model. Use it to stress-test custom assumptions.",
        "win_rate_band": "user-defined",
        "risk_profile": "depends on inputs",
        "best_regime": "depends on inputs",
        "notes": "Treat default values as a neutral starting point only.",
    },
}

SIZING_LIBRARY: Dict[str, dict] = {
    "Fixed size": {
        "descriptor": "Constant size regardless of drawdown buffer.",
        "geometry": "Flat profile — ignores the shrinking survival margin entirely.",
        "risk_note": "Only safe when edge and variance are very tightly controlled.",
    },
    "Linear scale-down": {
        "descriptor": "Reduces exposure linearly as the buffer shrinks.",
        "geometry": "Straight-line from full size (100% buffer) to zero (0% buffer).",
        "risk_note": "Simple and transparent. The most common defensive choice.",
    },
    "Half Kelly": {
        "descriptor": "Conservative Kelly-like taper.",
        "geometry": "Starts near full size, gradually de-risks. Minimum 50% floor.",
        "risk_note": "Fractional Kelly is safer than full Kelly under parameter uncertainty.",
    },
    "Risk geometry": {
        "descriptor": "Power-law curve controlled by exponent k.",
        "geometry": "Curve bends based on k. Low k = fast de-risk; high k = gradual cut.",
        "risk_note": "Best option for fine-tuned control of the buffer-to-size relationship.",
    },
    "Martingale": {
        "descriptor": "Increases size as drawdown deepens (loss-chasing).",
        "geometry": "Convex path that grows exposure into danger zones.",
        "risk_note": "Most fragile choice under trailing-drawdown constraints. Stress-test only.",
    },
}

STRATEGY_PRESETS: Dict[str, dict] = {
    # All presets verified: EV/day > $3000/30 = $100/day at 1 MES contract, default friction
    # Kalman: wr=60%, aw=$100 (R/R=1.82) → EV/day=$148, dtt=20d
    "Kalman pairs  (MES/MNQ)": dict(wr=60, aw=100, al=55,  td=4),
    # EMA: wr=44%, aw=$145 (R/R=2.13) → EV/day=$197, dtt=15d
    "EMA crossover  (5-min)":   dict(wr=44, aw=145, al=68,  td=8),
    # MR scalp: wr=68%, aw=$44 (R/R=1.22) → EV/day=$208, dtt=14d
    "Mean-reversion scalp":     dict(wr=68, aw=44,  al=36,  td=12),
    # Breakout: wr=38%, aw=$185 (R/R=2.43) → EV/day=$111, dtt=27d
    "Momentum breakout":        dict(wr=38, aw=185, al=76,  td=5),
    # Range: wr=65%, aw=$55 (R/R=1.31) → EV/day=$180, dtt=17d
    "Range trading":            dict(wr=65, aw=55,  al=42,  td=9),
    "Custom":                   dict(wr=55, aw=80,  al=70,  td=6),
}
SIZING_OPTS = list(SIZING_LIBRARY.keys())

# ══════════════════════════════════════════════════════════════════
# REGIME DEFINITIONS  (shared by MC tab selector + WF stress test)
# ══════════════════════════════════════════════════════════════════
REGIME_DEFS: Dict[str, dict] = {
    "No filter (mixed)": dict(wr_d=0.0,  aw_m=1.00, al_m=1.00, td_d=0,  color=C["muted"],
                               desc="No adjustment — your raw parameter estimates are used as-is."),
    "Trending":          dict(wr_d=+5.0, aw_m=1.20, al_m=1.00, td_d=-1, color=C["blue"],
                               desc="Persistent directional days: win rate +5pp, avg win ×1.2, trades/day −1."),
    "Choppy":            dict(wr_d=+3.0, aw_m=0.85, al_m=1.10, td_d=+2, color=C["teal"],
                               desc="Range-bound, mean-reverting: win rate +3pp, avg win ×0.85, avg loss ×1.1, trades/day +2."),
    "Volatile":          dict(wr_d=-8.0, aw_m=1.30, al_m=1.25, td_d=0,  color=C["amber"],
                               desc="News/whipsaw days: win rate −8pp, avg win ×1.3, avg loss ×1.25."),
    "Custom blend":      dict(wr_d=0.0,  aw_m=1.00, al_m=1.00, td_d=0,  color=C["purple"],
                               desc="Define a probability-weighted mix of regimes. Each session day samples from the blend."),
}

# Sensitivity sweep parameter definitions
SENS_PARAMS = {
    "Win rate (%)":         dict(attr="wr",           lo=25,  hi=80,  fmt=".0f"),
    "Avg win / ctr ($)":    dict(attr="aw",           lo=10,  hi=300, fmt=".0f"),
    "Avg loss / ctr ($)":   dict(attr="al",           lo=10,  hi=300, fmt=".0f"),
    "Trades / day":         dict(attr="td",           lo=1,   hi=15,  fmt=".0f"),
    "Contracts":            dict(attr="contracts",    lo=1,   hi=20,  fmt=".0f"),
    "Commission RT ($)":    dict(attr="commission_rt",lo=0.0, hi=5.0, fmt=".2f"),
    "Slippage (ticks)":     dict(attr="slippage_ticks",lo=0.0,hi=4.0, fmt=".2f"),
}

# ══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════
@dataclass
class Run:
    outcome:    str
    days:       int
    final_pnl:  float
    peak_pnl:   float
    max_dd:     float
    qual_days:  int
    equity:     List[float]
    daily_pnls: List[float]

# ══════════════════════════════════════════════════════════════════
# CORE SIMULATION HELPERS
# ══════════════════════════════════════════════════════════════════
def _size_mult(pnl: float, peak: float, method: str, k: float) -> float:
    used    = max(0.0, peak - pnl)
    buf_pct = max(0.0, (DD_LIMIT - used) / DD_LIMIT)
    if method == "Fixed size":        return 1.0
    elif method == "Linear scale-down": return max(0.05, buf_pct)
    elif method == "Half Kelly":        return max(0.05, 0.5 + 0.5 * buf_pct)
    elif method == "Risk geometry":
        exp = 1.0 / max(k, 0.05)
        return max(0.05, buf_pct ** exp)
    elif method == "Martingale":        return min(3.0, max(0.05, 2.0 - buf_pct))
    return 1.0

def _instrument_factor(instrument: str) -> float:
    # NOTE: this function is retained for legacy compatibility but is no longer
    # called in the simulation engine (simulate_one treats aw/al as true $/contract).
    return CONTRACT_SPECS[instrument]["point_value"] / BASE_POINT_VALUE

def _trade_cost(instrument: str, commission_rt: float, slippage_ticks: float) -> float:
    return max(0.0, commission_rt) + max(0.0, slippage_ticks) * CONTRACT_SPECS[instrument]["tick_value"]

def simulate_one(
    wr: float, aw: float, al: float, td: int,
    sizing: str, k: float, instrument: str,
    contracts: int, commission_rt: float, slippage_ticks: float,
    rng: np.random.Generator,
) -> Run:
    pnl, peak, qual = 0.0, 0.0, 0
    eq: List[float] = [0.0]
    daily_pnls: List[float] = []
    cost = contracts * _trade_cost(instrument, commission_rt, slippage_ticks)

    for day in range(MAX_DAYS):
        day_pnl  = 0.0
        n_trades = int(np.clip(td + rng.integers(-1, 2), 1, MAX_TRADES_DAY))
        for _ in range(n_trades):
            if day_pnl <= -DAILY_LOSS_LIM:
                break
            mult  = _size_mult(pnl, peak, sizing, k)
            won   = rng.random() < (wr / 100.0)
            noise = rng.uniform(0.55, 1.45)
            # aw/al are true $/contract; multiply by contracts and noise only
            gross = (aw * mult * noise) if won else (-al * mult * noise)
            trade = gross * contracts - cost
            day_pnl += trade
            pnl     += trade
            peak     = max(peak, pnl)
            if peak - pnl >= DD_LIMIT:
                eq.append(pnl); daily_pnls.append(day_pnl)
                return Run("bust", day+1, pnl, peak, peak-pnl, qual, eq, daily_pnls)
            # FIX: count today as a qualifying day if day_pnl already >= QUAL_MIN
            if pnl >= PROFIT_TARGET and (qual + (1 if day_pnl >= QUAL_MIN else 0)) >= QUAL_DAYS_NEED:
                eq.append(pnl); daily_pnls.append(day_pnl)
                return Run("pass", day+1, pnl, peak, peak-pnl, qual, eq, daily_pnls)
        eq.append(pnl); daily_pnls.append(day_pnl)
        if day_pnl >= QUAL_MIN:
            qual += 1
    return Run("timeout", MAX_DAYS, pnl, peak, peak-pnl, qual, eq, daily_pnls)


def simulate_one_blended(
    wr: float, aw: float, al: float, td: int,
    sizing: str, k: float, instrument: str,
    contracts: int, commission_rt: float, slippage_ticks: float,
    rng: np.random.Generator,
    blend_names: List[str],
    blend_weights: List[float],
) -> Run:
    """Like simulate_one but each day samples a regime from the blend distribution."""
    pnl, peak, qual = 0.0, 0.0, 0
    eq: List[float] = [0.0]
    daily_pnls: List[float] = []
    cost    = contracts * _trade_cost(instrument, commission_rt, slippage_ticks)
    weights = np.asarray(blend_weights, dtype=float)
    weights = weights / weights.sum()  # normalise

    for day in range(MAX_DAYS):
        # Sample today's regime
        regime_name = rng.choice(blend_names, p=weights)
        r = REGIME_DEFS[regime_name]
        d_wr = float(np.clip(wr + r["wr_d"], 10, 95))
        d_aw = aw * r["aw_m"]
        d_al = al * r["al_m"]
        d_td = int(np.clip(td + r["td_d"], 1, MAX_TRADES_DAY))

        day_pnl  = 0.0
        n_trades = int(np.clip(d_td + rng.integers(-1, 2), 1, MAX_TRADES_DAY))
        for _ in range(n_trades):
            if day_pnl <= -DAILY_LOSS_LIM:
                break
            mult  = _size_mult(pnl, peak, sizing, k)
            won   = rng.random() < (d_wr / 100.0)
            noise = rng.uniform(0.55, 1.45)
            gross = (d_aw * mult * noise) if won else (-d_al * mult * noise)
            trade = gross * contracts - cost
            day_pnl += trade
            pnl     += trade
            peak     = max(peak, pnl)
            if peak - pnl >= DD_LIMIT:
                eq.append(pnl); daily_pnls.append(day_pnl)
                return Run("bust", day+1, pnl, peak, peak-pnl, qual, eq, daily_pnls)
            # FIX: count today as a qualifying day if day_pnl already >= QUAL_MIN
            if pnl >= PROFIT_TARGET and (qual + (1 if day_pnl >= QUAL_MIN else 0)) >= QUAL_DAYS_NEED:
                eq.append(pnl); daily_pnls.append(day_pnl)
                return Run("pass", day+1, pnl, peak, peak-pnl, qual, eq, daily_pnls)
        eq.append(pnl); daily_pnls.append(day_pnl)
        if day_pnl >= QUAL_MIN:
            qual += 1
    return Run("timeout", MAX_DAYS, pnl, peak, peak-pnl, qual, eq, daily_pnls)


def _sample_params(wr, aw, al, sigma_wr, sigma_pay, rng):
    """Sample perturbed parameters from a distribution around the given values.

    This models calibration uncertainty — the fact that your estimated win rate
    and payoff sizes are measurements with error, not known quantities.

    Each simulation run draws its own parameters, so the resulting pass rate is
    the MARGINAL probability integrating over parameter uncertainty:
        P(pass) = ∫ P(pass|θ) · P(θ|data) dθ
    rather than the conditional P(pass|θ=θ_point_estimate).

    Args:
        sigma_wr  : std deviation of win rate in percentage points
        sigma_pay : fractional std deviation of avg win / avg loss
    """
    if sigma_wr == 0.0:
        return wr, aw, al
    s_wr = float(np.clip(wr + rng.normal(0, sigma_wr), 10.0, 95.0))
    s_aw = float(max(1.0, aw * (1.0 + rng.normal(0, sigma_pay))))
    s_al = float(max(1.0, al * (1.0 + rng.normal(0, sigma_pay))))
    return s_wr, s_aw, s_al


def run_mc(wr, aw, al, td, sizing, k, instrument, contracts, commission_rt, slippage_ticks, n_sims,
           regime: str = "No filter (mixed)", blend_weights: Optional[Dict[str, float]] = None,
           sigma_wr: float = 0.0, sigma_pay: float = 0.0):
    """Run Monte Carlo simulation with optional regime filter, custom blend,
    and per-run parameter uncertainty (Calibration Confidence).

    When sigma_wr > 0, each run draws its own win rate / payoffs from a
    distribution around the provided values. This produces the marginal
    pass probability integrating over parameter estimation uncertainty.
    """
    rng = np.random.default_rng()

    def _one_run():
        s_wr, s_aw, s_al = _sample_params(wr, aw, al, sigma_wr, sigma_pay, rng)
        if regime == "Custom blend" and blend_weights:
            names   = list(blend_weights.keys())
            weights = [blend_weights[n] for n in names]
            return simulate_one_blended(s_wr, s_aw, s_al, td, sizing, k, instrument,
                                        contracts, commission_rt, slippage_ticks, rng,
                                        names, weights)
        elif regime != "No filter (mixed)" and regime in REGIME_DEFS:
            r      = REGIME_DEFS[regime]
            eff_wr = float(np.clip(s_wr + r["wr_d"], 10, 95))
            eff_aw = s_aw * r["aw_m"]
            eff_al = s_al * r["al_m"]
            eff_td = int(np.clip(td + r["td_d"], 1, MAX_TRADES_DAY))
            return simulate_one(eff_wr, eff_aw, eff_al, eff_td, sizing, k, instrument,
                                 contracts, commission_rt, slippage_ticks, rng)
        else:
            return simulate_one(s_wr, s_aw, s_al, td, sizing, k, instrument,
                                 contracts, commission_rt, slippage_ticks, rng)

    results = [_one_run() for _ in range(n_sims)]

    passes  = [r for r in results if r.outcome=="pass"]
    busts   = [r for r in results if r.outcome=="bust"]
    tos     = [r for r in results if r.outcome=="timeout"]
    sample: List[Run] = []
    for pool, n in [(passes,14),(busts,14),(tos,10)]:
        if pool:
            idx = rng.choice(len(pool), size=min(n,len(pool)), replace=False)
            sample.extend([pool[i] for i in idx])
    return results, sample

def sweep_contracts(wr,aw,al,td,sizing,k,instrument,commission_rt,slippage_ticks,n_sims):
    rng = np.random.default_rng()
    x = list(range(1, MAX_CONTRACTS+1))
    pass_rates, bust_rates, evs, avg_days = [], [], [], []
    for c in x:
        res = [simulate_one(wr,aw,al,td,sizing,k,instrument,c,commission_rt,slippage_ticks,rng) for _ in range(n_sims)]
        pass_rates.append(sum(r.outcome=="pass" for r in res)/n_sims*100)
        bust_rates.append(sum(r.outcome=="bust" for r in res)/n_sims*100)
        evs.append(float(np.mean([r.final_pnl for r in res])))
        passed=[r.days for r in res if r.outcome=="pass"]
        avg_days.append(float(np.mean(passed)) if passed else 0.0)
    return {"contracts":x,"pass_rates":pass_rates,"bust_rates":bust_rates,"evs":evs,"avg_days":avg_days}

def compute_stats(results: List[Run], n: int) -> Dict:
    passes = [r for r in results if r.outcome=="pass"]
    busts  = [r for r in results if r.outcome=="bust"]
    tos    = [r for r in results if r.outcome=="timeout"]
    all_daily = [x for r in results for x in r.daily_pnls]
    return {
        "pass_rate": len(passes)/n*100, "bust_rate": len(busts)/n*100, "to_rate": len(tos)/n*100,
        "n_pass": len(passes), "n_bust": len(busts), "n_timeout": len(tos),
        "avg_days":    float(np.mean([r.days for r in passes])) if passes else 0.0,
        "median_days": float(np.median([r.days for r in passes])) if passes else 0.0,
        "avg_qual":    float(np.mean([r.qual_days for r in results])) if results else 0.0,
        "pass_days":   [r.days for r in passes],
        "bust_days":   [r.days for r in busts],
        "all_daily":   all_daily,
        "mean_final":  float(np.mean([r.final_pnl for r in results])) if results else 0.0,
        "max_dd_avg":  float(np.mean([r.max_dd for r in results])) if results else 0.0,
        "pnl_std":     float(np.std([r.final_pnl for r in results],ddof=1)) if len(results)>1 else 0.0,
    }

# ══════════════════════════════════════════════════════════════════
# WALK FORWARD FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def run_walk_forward(base: dict, n_folds: int, noise_pct: float, oos_deg: float, oos_frac: float, n_sims: int) -> dict:
    """
    Walk-forward test across n_folds.
    IS  = current params ± fold noise (regime variation)
    OOS = IS params - systematic degradation (IS→OOS decay)
    """
    rng = np.random.default_rng(42)
    is_rates, oos_rates = [], []
    is_sample, oos_sample = [], []

    for fold in range(n_folds):
        noise = noise_pct / 100.0
        # IS fold params (simulate regime variation across sub-periods)
        f_wr = float(np.clip(base["wr"] + rng.normal(0, noise * base["wr"]), 20, 85))
        f_aw = float(max(5, base["aw"] * (1 + rng.normal(0, noise))))
        f_al = float(max(5, base["al"] * (1 + rng.normal(0, noise))))
        f_td = int(np.clip(base["td"] + rng.integers(-1, 2), 1, MAX_TRADES_DAY))

        # OOS: apply systematic degradation + additional OOS noise
        # Win rate degradation is additive (it is a percentage point, not a multiplier)
        # Avg win/loss degradation is multiplicative (they are dollar amounts)
        d = oos_deg / 100.0
        oos_noise_scale = noise * 0.5   # OOS has half the parameter noise of IS
        o_wr = float(np.clip(f_wr - d * 0.8 * base["wr"] + rng.normal(0, oos_noise_scale * base["wr"]), 15, 90))
        o_aw = float(max(5, f_aw * (1 - d * 0.6) * (1 + rng.normal(0, oos_noise_scale))))
        o_al = float(max(5, f_al * (1 + d * 0.3) * (1 + rng.normal(0, oos_noise_scale))))

        n_is  = max(30, int(n_sims * (1 - oos_frac)))
        n_oos = max(30, n_sims - n_is)

        is_res  = [simulate_one(f_wr,f_aw,f_al,f_td,base["sizing"],base["k"],base["instrument"],base["contracts"],base["commission_rt"],base["slippage_ticks"],rng) for _ in range(n_is)]
        oos_res = [simulate_one(o_wr,o_aw,o_al,f_td,base["sizing"],base["k"],base["instrument"],base["contracts"],base["commission_rt"],base["slippage_ticks"],rng) for _ in range(n_oos)]

        is_rates.append(sum(r.outcome=="pass" for r in is_res)/n_is*100)
        oos_rates.append(sum(r.outcome=="pass" for r in oos_res)/n_oos*100)

        # collect sample equity curves
        for pool,store in [(is_res,is_sample),(oos_res,oos_sample)]:
            passes = [r for r in pool if r.outcome=="pass"]
            busts  = [r for r in pool if r.outcome=="bust"]
            for p in (passes[:3]+busts[:2]):
                store.append(p)

    mean_is  = float(np.mean(is_rates))
    mean_oos = float(np.mean(oos_rates))
    wfe      = mean_oos / max(1e-9, mean_is)
    # Stability = 1 - CV(OOS rates), where CV = σ/μ, clamped to [0,1]
    # High CV (variable pass rates across folds) → low stability
    # CV is clipped to 1.0 so stability stays non-negative even for erratic OOS results
    oos_cv    = float(np.std(oos_rates) / max(mean_oos, 1e-3)) if len(oos_rates) > 1 else 0.0
    stability = max(0.0, 1.0 - min(1.0, oos_cv))

    return {
        "is_rates": is_rates, "oos_rates": oos_rates,
        "mean_is": mean_is,   "mean_oos": mean_oos,
        "wfe": wfe,           "stability": stability,
        "is_sample": is_sample[:20], "oos_sample": oos_sample[:20],
        "n_folds": n_folds,
    }

def run_regime_stress(base: dict, n_sims: int) -> dict:
    """Test strategy under 3 regime perturbations + baseline, using shared REGIME_DEFS."""
    rng = np.random.default_rng()
    results = {}
    for name in ["Trending", "Choppy", "Volatile", "No filter (mixed)"]:
        r   = REGIME_DEFS[name]
        wr  = float(np.clip(base["wr"] + r["wr_d"], 15, 90))
        aw  = base["aw"] * r["aw_m"]
        al  = base["al"] * r["al_m"]
        td  = int(np.clip(base["td"] + r["td_d"], 1, MAX_TRADES_DAY))
        res = [simulate_one(wr,aw,al,td,base["sizing"],base["k"],base["instrument"],base["contracts"],base["commission_rt"],base["slippage_ticks"],rng) for _ in range(n_sims)]
        pr  = sum(x.outcome=="pass" for x in res)/n_sims*100
        br  = sum(x.outcome=="bust" for x in res)/n_sims*100
        label = "Baseline" if name == "No filter (mixed)" else name
        results[label] = {"pass_rate":pr,"bust_rate":br,"color":r["color"]}
    return results

# ══════════════════════════════════════════════════════════════════
# SENSITIVITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def sweep_one_param(base: dict, param_name: str, n_pts: int, n_sims: int):
    """Sweep a single parameter, return (values, pass_rates)."""
    rng  = np.random.default_rng()
    cfg  = SENS_PARAMS[param_name]
    vals = np.linspace(cfg["lo"], cfg["hi"], n_pts)
    rates= []
    for v in vals:
        b = dict(base)
        b[cfg["attr"]] = float(v)
        if cfg["attr"] == "td":
            b["td"] = max(1, int(round(v)))
        if cfg["attr"] == "contracts":
            b["contracts"] = max(1, int(round(v)))
        res = [simulate_one(b["wr"],b["aw"],b["al"],b["td"],b["sizing"],b["k"],b["instrument"],b["contracts"],b["commission_rt"],b["slippage_ticks"],rng) for _ in range(n_sims)]
        rates.append(sum(r.outcome=="pass" for r in res)/n_sims*100)
    return list(vals), rates

def run_tornado(base: dict, n_sims: int) -> list:
    """Run OAT sweep for all params, return sorted list of (name, lo_rate, hi_rate, base_rate)."""
    rng = np.random.default_rng()
    # baseline
    res_base = [simulate_one(base["wr"],base["aw"],base["al"],base["td"],base["sizing"],base["k"],base["instrument"],base["contracts"],base["commission_rt"],base["slippage_ticks"],rng) for _ in range(n_sims)]
    base_rate= sum(r.outcome=="pass" for r in res_base)/n_sims*100

    rows = []
    for pname, cfg in SENS_PARAMS.items():
        def _rate(v):
            b = dict(base)
            b[cfg["attr"]] = float(v)
            if cfg["attr"] == "td":       b["td"]        = max(1,int(round(v)))
            if cfg["attr"] == "contracts":b["contracts"] = max(1,int(round(v)))
            r2 = [simulate_one(b["wr"],b["aw"],b["al"],b["td"],b["sizing"],b["k"],b["instrument"],b["contracts"],b["commission_rt"],b["slippage_ticks"],rng) for _ in range(n_sims)]
            return sum(x.outcome=="pass" for x in r2)/n_sims*100
        lo_rate = _rate(cfg["lo"])
        hi_rate = _rate(cfg["hi"])
        rows.append((pname, lo_rate, hi_rate, base_rate))
    rows.sort(key=lambda x: abs(x[2]-x[1]), reverse=True)
    return rows, base_rate

def run_heatmap(base: dict, px: str, py: str, n_grid: int, n_sims: int) -> dict:
    """2D sweep of two params. Returns grid of pass rates + axis labels."""
    rng  = np.random.default_rng()
    cfgx = SENS_PARAMS[px]
    cfgy = SENS_PARAMS[py]
    xv   = np.linspace(cfgx["lo"], cfgx["hi"], n_grid)
    yv   = np.linspace(cfgy["lo"], cfgy["hi"], n_grid)
    grid = np.zeros((n_grid, n_grid))

    for i, x_val in enumerate(xv):
        for j, y_val in enumerate(yv):
            b = dict(base)
            b[cfgx["attr"]] = float(x_val)
            b[cfgy["attr"]] = float(y_val)
            if cfgx["attr"]=="td":        b["td"]=max(1,int(round(x_val)))
            if cfgx["attr"]=="contracts": b["contracts"]=max(1,int(round(x_val)))
            if cfgy["attr"]=="td":        b["td"]=max(1,int(round(y_val)))
            if cfgy["attr"]=="contracts": b["contracts"]=max(1,int(round(y_val)))
            res=[simulate_one(b["wr"],b["aw"],b["al"],b["td"],b["sizing"],b["k"],b["instrument"],b["contracts"],b["commission_rt"],b["slippage_ticks"],rng) for _ in range(n_sims)]
            grid[j, i] = sum(r.outcome=="pass" for r in res)/n_sims*100
    return {"grid":grid,"xv":xv,"yv":yv,"px":px,"py":py}

# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# ALGORITHM CONFIG ENGINE  — pure mathematics, no Monte Carlo
# ══════════════════════════════════════════════════════════════════
def _ror_two_sided(ev_day: float, std_day: float,
                   L: float = DD_LIMIT, T: float = PROFIT_TARGET) -> float:
    """
    Exact two-sided gambler's ruin probability for Brownian motion with drift.
    P(hitting -L before +T | start at 0, drift μ=ev_day, diffusion σ=std_day).

    Formula: P(ruin) = (e^(αL) − 1) / (e^(α(L+T)) − 1)  where α = 2μ/σ²

    Numerically stable: uses asymptotic form exp(−α·T) when α(L+T) > 500
    to avoid float64 overflow while maintaining machine-precision accuracy.
    Zero-drift limit: P(ruin) → L/(L+T) as μ→0.
    """
    if ev_day <= 0.0:
        return 1.0
    if std_day <= 0.0:
        return 0.0
    alpha = 2.0 * ev_day / (std_day ** 2)
    aL, aLT = alpha * L, alpha * (L + T)
    if aLT > 500:
        # Asymptotic: P(ruin) ≈ e^(−α·T) — accurate when both barriers are "distant"
        # in units of the diffusion scale √(σ²/2μ)
        ror = float(np.exp(-alpha * T))
    else:
        ror = float((np.exp(aL) - 1.0) / max(np.exp(aLT) - 1.0, 1e-12))
    return max(0.0, min(1.0, ror))


def _compute_alg_config(bp: dict, n_sims: int) -> dict:
    import scipy.stats as ss
    wr=bp["wr"]/100.0; aw=bp["aw"]; al=bp["al"]; td=bp["td"]
    c=bp["contracts"]; instr=bp["instrument"]; comm=bp["commission_rt"]
    slip=bp["slippage_ticks"]; kf=bp["kelly_frac"]
    friction_total=c*_trade_cost(instr,comm,slip)
    ev_gross=wr*aw-(1-wr)*al
    ev_trade=c*ev_gross-friction_total
    ev_day=td*ev_trade

    # Variance of one trade gross (correct derivation with uniform noise U[0.55,1.45]):
    # E[noise²] = Var[noise] + (E[noise])² = (0.9²/12) + 1 = 1.0675
    var_noise  = ((1.45-0.55)**2)/12.0        # = 0.0675
    e_noise_sq = 1.0 + var_noise               # = 1.0675 = E[noise²]
    e_x2_gross = (wr*aw**2 + (1-wr)*al**2) * e_noise_sq
    var_trade_gross = e_x2_gross - ev_gross**2  # Var = E[X²] - (E[X])²
    var_trade  = c**2 * var_trade_gross         # c contracts, same noise draw
    std_trade  = max(0.001, var_trade**0.5)
    std_day    = max(0.001, (td * var_trade)**0.5)  # CLT: i.i.d. trades per day

    sharpe     = ev_day / std_day if std_day > 0 else 0.0
    days_to_target = PROFIT_TARGET / ev_day if ev_day > 0 else float("inf")

    rr = aw / al if al > 0 else 1.0
    # Kelly criterion: f* = p - (1-p)/b  where b = avg_win/avg_loss
    kelly_f = max(0.0, wr - (1-wr)/rr) if rr > 0 else 0.0
    kelly_rec_frac = kf * kelly_f
    # Optimal contracts: f_applied * DD_buffer / avg_loss_per_contract
    rec_contracts = max(1, min(MAX_CONTRACTS, int(round(kelly_rec_frac * DD_LIMIT / max(al, 1.0)))))

    # P(qualifying day): CLT approximation daily P&L ~ N(ev_day, std_day²)
    # Note: ignores daily loss limit truncation (small conservatism, ~1-5pp)
    z_qual = (QUAL_MIN - ev_day) / std_day if std_day > 0 else float("inf")
    p_qual_day = float(np.clip(ss.norm.sf(z_qual), 0, 1))
    p_enough_quals = float(1 - ss.binom.cdf(QUAL_DAYS_NEED - 1, MAX_DAYS, p_qual_day))

    # Risk of ruin and P(pass) — two-sided gambler's ruin formula (Brownian drift)
    # For drift μ = ev_day, diffusion σ = std_day, barriers at -L and +T:
    # P(ruin | start at 0) = (exp(2μL/σ²) - 1) / (exp(2μ(L+T)/σ²) - 1)  [μ > 0]
    # P(pass | start at 0) = 1 - P(ruin)  [ignoring finite-horizon timeout]
    # Risk of ruin and P(pass) via two-sided gambler's ruin (see _ror_two_sided)
    ror           = _ror_two_sided(ev_day, std_day)
    p_pass_approx = 1.0 - ror

    wr_breakeven = (1.0 / (1.0 + rr)) * 100.0
    rr_breakeven = (1.0 - wr) / wr if wr > 0 else float("inf")
    wr_margin    = bp["wr"] - wr_breakeven
    rr_margin    = rr - rr_breakeven
    edge_snr     = abs(ev_trade) / std_trade if std_trade > 0 else 0

    # Score components — each maps a metric to [0, 100]
    # ev_score: intraday daily Sharpe typically 0.05–0.50; scale so 0.33 Sharpe = 100
    ev_score     = min(100, max(0, sharpe * 300))
    # time_score: full marks at dtt≤5d, zero at dtt≥35d (5d past the 30d limit)
    time_score   = max(0, min(100, (1 - (days_to_target - 5) / MAX_DAYS) * 100))
    qual_score   = min(100, p_enough_quals * 100)
    risk_score   = max(0, (1 - ror) * 100)
    margin_score = min(100, max(0, wr_margin * 4 + rr_margin * 15))
    score = ev_score*0.30 + time_score*0.25 + qual_score*0.20 + risk_score*0.15 + margin_score*0.10

    wfe_actions=[]
    if wr_margin<5: wfe_actions.append("⚠ Win rate barely above break-even — OOS decay collapses\n   performance. Target ≥8pp margin above break-even win rate.")
    if rr<1.0: wfe_actions.append("⚠ R/R < 1.0 — win rate degradation OOS is amplified.\n   Target R/R ≥ 1.0 for parameter-robust strategy.")
    if td>=10: wfe_actions.append("ℹ High trade count amplifies friction OOS. Fewer higher-\n   quality setups may improve cost-adjusted WFE.")
    if kf>0.5: wfe_actions.append("⚠ Kelly fraction > 0.5 is aggressive under OOS uncertainty.\n   Use ≤ 0.25x Kelly for strategies with uncertain parameters.")
    if days_to_target>MAX_DAYS*0.75: wfe_actions.append("⚠ Days-to-target near 30-day limit. OOS degradation in\n   EV/day will push timeout rate above acceptable levels.")
    if edge_snr>0.3: wfe_actions.append(f"✓ Edge/noise ratio {edge_snr:.2f} — strategy signal is strong\n   relative to intraday variance. Should sustain good WFE.")
    if not [a for a in wfe_actions if "⚠" in a]: wfe_actions.append("✓ No major WFE risk factors. Maintain parameter discipline\n   and validate quarterly against live results.")

    stab_actions=[]
    if std_day>ev_day*5: stab_actions.append(f"⚠ σ/day (${std_day:.0f}) is {std_day/max(ev_day,1):.1f}x EV/day — very noisy.\n   High fold-to-fold variance will hurt stability score.")
    if p_qual_day<0.50: stab_actions.append(f"⚠ Only {p_qual_day*100:.0f}% P(qualifying day). Inconsistent qual\n   accumulation → high OOS variance → low stability.")
    if c<rec_contracts and rec_contracts<=MAX_CONTRACTS: stab_actions.append(f"ℹ Kelly recommends {rec_contracts}c → higher EV/day → more\n   consistent qualifying days → better stability.")
    if rr<0.8: stab_actions.append("⚠ Low R/R creates wide daily P&L distribution.\n   Improve avg win / avg loss to stabilise daily outcomes.")
    if not [a for a in stab_actions if "⚠" in a]: stab_actions.append("✓ Daily P&L distribution appears stable.")
    # Analytical stability proxy: expected CV of OOS pass rates across folds
    # Each fold's OOS rate is Binomial(n_oos, p_pass), so σ(rate) = sqrt(p*(1-p)/n_oos)
    # CV = σ / μ = sqrt(p*(1-p)/n_oos) / p  = sqrt((1-p)/(p * n_oos))
    # Stability = 1 - min(CV, 1)  so it remains in [0,1]
    p_safe = max(0.01, min(0.99, p_pass_approx))
    n_oos_approx = 120  # typical OOS sims per fold (30% of 400)
    oos_cv_analytical = float(np.sqrt((1 - p_safe) / max(p_safe * n_oos_approx, 1e-9)))
    stab_analytical = max(0.0, 1.0 - min(1.0, oos_cv_analytical))
    stab_actions.append(f"ℹ Analytical stability estimate: {stab_analytical:.2f}  (CV={oos_cv_analytical:.2f})")

    rng=np.random.default_rng()
    sim_pass_by_c={}
    for cc in sorted(set([1,max(1,rec_contracts-1),rec_contracts,min(MAX_CONTRACTS,rec_contracts+2),min(8,MAX_CONTRACTS)])):
        res=[simulate_one(bp["wr"],aw,al,td,bp["sizing"],bp["k"],instr,cc,comm,slip,rng) for _ in range(n_sims)]
        sim_pass_by_c[cc]=sum(r.outcome=="pass" for r in res)/n_sims*100

    if score>=80:   rdesc="Excellent. Strong edge, comfortable timing, controlled risk. Focus on execution."
    elif score>=65: rdesc="Good — identifiable weaknesses. Address WFE/stability actions to reach Excellent."
    elif score>=50: rdesc="Marginal. Can pass challenges but sensitive to parameter errors. Review actions above."
    elif score>=35: rdesc="Weak. Edge too thin or timing too tight. Significant recalibration needed."
    else:           rdesc="Unsuitable. EV near zero or negative. Do not attempt a challenge."

    preset_name="Custom"
    for k,v in STRATEGY_PRESETS.items():
        if abs(v["wr"]-bp["wr"])<2 and abs(v["aw"]-bp["aw"])<5 and abs(v["al"]-bp["al"])<5:
            preset_name=k; break

    def fmtA(actions):
        return "\n".join("  "+a.replace("\n","\n  ") for a in actions)

    report=(
        f"╔══════════════════════════════════════════════════════════════╗\n"
        f"║  ALGORITHM CONFIGURATION REPORT                              ║\n"
        f"║  Strategy : {preset_name:<50}║\n"
        f"╚══════════════════════════════════════════════════════════════╝\n"
        f"Overall Score : {score:.1f}/100   Grade : {'A' if score>=80 else 'B' if score>=65 else 'C' if score>=50 else 'D' if score>=35 else 'F'}\n"
        f"{rdesc}\n\n"
        f"━━━ CORE PARAMETERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Win rate : {bp['wr']:.0f}%  (break-even: {wr_breakeven:.1f}%  margin: {wr_margin:+.1f}pp)\n"
        f"  R/R      : {rr:.3f}  (break-even: {rr_breakeven:.3f}  margin: {rr_margin:+.3f})\n"
        f"  Avg win  : ${aw:.2f}/ctr    Avg loss: ${al:.2f}/ctr\n"
        f"  Trades/d : {td}    Contracts: {c}    Instrument: {instr}\n\n"
        f"━━━ EXPECTED VALUE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  EV/trade : ${ev_trade:+.4f}    EV/day: ${ev_day:+.2f}    σ/day: ${std_day:.2f}\n"
        f"  Sharpe   : {sharpe:.4f}    Days to $3k: {days_to_target:.1f}  {'✓ within 30d' if days_to_target<=30 else '✗ EXCEEDS 30d'}\n\n"
        f"━━━ KELLY SIZING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Full Kelly f  : {kelly_f:.4f}    Applied ({kf:.2f}x): {kelly_rec_frac:.4f}\n"
        f"  Rec contracts : {rec_contracts}   (currently: {c})  {'✓' if c<=rec_contracts else '⚠ over-leveraged vs Kelly'}\n\n"
        f"━━━ QUALIFYING DAYS  (binomial) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  P(day≥$150)   : {p_qual_day*100:.1f}%    P(≥5 qual/30d): {p_enough_quals*100:.1f}%\n\n"
        f"━━━ RISK METRICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Risk of ruin  : {ror*100:.3f}%    P(pass approx): {p_pass_approx*100:.1f}%\n\n"
        f"━━━ WFE IMPROVEMENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{fmtA(wfe_actions)}\n\n"
        f"━━━ STABILITY IMPROVEMENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{fmtA(stab_actions)}\n\n"
        f"━━━ MONTE CARLO VALIDATION ({n_sims} sims/point) ━━━━━━━━━━━━━━\n"
        + "\n".join(f"  {cc:2d}c : {sim_pass_by_c[cc]:.1f}% pass"+(" ← current" if cc==c else "")+(" ← Kelly rec" if cc==rec_contracts and cc!=c else "") for cc in sorted(sim_pass_by_c))
        + f"\n\n━━━ SCORE BREAKDOWN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  EV quality(30%): {ev_score:.1f} → {ev_score*0.30:.1f}pts\n"
        f"  Timing    (25%): {time_score:.1f} → {time_score*0.25:.1f}pts\n"
        f"  Qual days (20%): {qual_score:.1f} → {qual_score*0.20:.1f}pts\n"
        f"  Risk ctrl (15%): {risk_score:.1f} → {risk_score*0.15:.1f}pts\n"
        f"  Edge margin(10%): {margin_score:.1f} → {margin_score*0.10:.1f}pts\n"
        f"  TOTAL           : {score:.1f}/100\n"
    )
    return dict(score=score,rating_desc=rdesc,
                metrics=dict(ev_trade=ev_trade,ev_day=ev_day,std_day=std_day,sharpe=sharpe,
                             kelly=kelly_f,kelly_rec_frac=kelly_rec_frac,rec_contracts=rec_contracts,
                             days_to_target=days_to_target,p_qual_day=p_qual_day,ror=ror),
                wfe_actions=wfe_actions,stab_actions=stab_actions,
                sim_pass_by_c=sim_pass_by_c,report=report)


# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════
class PropSimApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Prop Firm Monte Carlo  |  v2 Enhanced  ·  $2,000 Trail DD  ·  $3,000 Target")
        self.configure(bg=C["bg"])
        self.geometry("1600x1020")
        self.minsize(1360, 860)
        self._pick_map: Dict[int, str] = {}   # id(Text artist) → INFO key
        self._styles()
        self._build()
        self.after(150, self._run_mc)

    # ─── Styles ───────────────────────────────────────────────────
    def _styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=C["bg"], foreground=C["text"], font=("Helvetica",11))
        s.configure("F.TFrame", background=C["bg"])
        s.configure("P.TFrame", background=C["panel"])
        s.configure("H1.TLabel", background=C["bg"], foreground=C["text"], font=("Helvetica",14,"bold"))
        s.configure("Sub.TLabel", background=C["bg"], foreground=C["muted"], font=("Helvetica",10))
        s.configure("Sec.TLabel", background=C["bg"], foreground=C["muted"], font=("Helvetica",9,"bold"))
        s.configure("EV.TLabel",  background=C["bg"], foreground=C["text"], font=("Helvetica",10))
        for key,col in [("Stat.TLabel",C["text"]),("StatG.TLabel",C["win"]),("StatR.TLabel",C["bust"]),("StatA.TLabel",C["amber"])]:
            s.configure(key, background=C["panel"], foreground=col, font=("Helvetica",20,"bold"))
        s.configure("StatLbl.TLabel", background=C["panel"], foreground=C["muted"], font=("Helvetica",10))
        s.configure("Run.TButton",  background=C["blue"], foreground="white", font=("Helvetica",11,"bold"), padding=(10,6), relief="flat")
        s.configure("Info.TButton", background=C["bg"],   foreground=C["blue"], font=("Helvetica",9),    padding=(2,1),  relief="flat")
        s.map("Run.TButton",  background=[("active","#0C447C"),("disabled",C["border"])])
        s.map("Info.TButton", background=[("active",C["blue_l"])])
        s.configure("H.TScale", background=C["bg"], troughcolor=C["border"], sliderlength=14, sliderrelief="flat")
        s.configure("TCombobox", fieldbackground="white", background=C["bg"], padding=4)
        s.configure("TSpinbox",  fieldbackground="white", background=C["bg"], padding=3)
        s.configure("TNotebook", background=C["panel"], tabposition="nw")
        s.configure("TNotebook.Tab", background=C["panel"], foreground=C["muted"], padding=(14,6), font=("Helvetica",11))
        s.map("TNotebook.Tab",
              background=[("selected",C["bg"]),("active",C["blue_l"])],
              foreground=[("selected",C["blue"]),("active",C["blue"])],
              font=[("selected",("Helvetica",11,"bold"))])

    # ─── Scrollable control panel factory ────────────────────────
    def _make_scroll_ctrl(self, tab, width=314):
        """Return a scrollable Frame (with scrollbar) that fills the left column of tab."""
        outer = ttk.Frame(tab, style="F.TFrame", width=width)
        outer.grid(row=0, column=0, sticky="nsew", padx=(4, 0))
        outer.grid_propagate(False)

        vsb = ttk.Scrollbar(outer, orient="vertical")
        vsb.pack(side="right", fill="y")

        # Canvas fills all remaining space after the scrollbar is packed.
        # highlightthickness=0 removes the canvas focus border (1-2px on Windows)
        # that would otherwise appear to "eat" content on the right edge.
        cvs = tk.Canvas(outer, bg=C["bg"], highlightthickness=0, yscrollcommand=vsb.set)
        cvs.pack(side="left", fill="both", expand=True)
        vsb.configure(command=cvs.yview)

        ctrl = ttk.Frame(cvs, style="F.TFrame")
        win_id = cvs.create_window((0, 0), window=ctrl, anchor="nw")

        def _on_frame_configure(event):
            cvs.configure(scrollregion=cvs.bbox("all"))

        def _on_canvas_configure(event):
            # Pin the inner frame width to the canvas viewport width.
            # Subtract 2px to leave a small right margin so the last
            # character of any label is never clipped by the scrollbar border.
            cvs.itemconfig(win_id, width=max(1, event.width - 2))

        ctrl.bind("<Configure>", _on_frame_configure)
        cvs.bind("<Configure>", _on_canvas_configure)

        def _on_wheel(event):
            delta = getattr(event, "delta", 0)
            if delta:
                cvs.yview_scroll(int(-1 * (delta / 120)), "units")
            elif event.num == 4:
                cvs.yview_scroll(-1, "units")
            elif event.num == 5:
                cvs.yview_scroll(1, "units")

        for widget in (cvs, ctrl):
            widget.bind("<MouseWheel>", _on_wheel)
            widget.bind("<Button-4>",   _on_wheel)
            widget.bind("<Button-5>",   _on_wheel)

        return ctrl, cvs

    # ─── Top-level build ──────────────────────────────────────────
    def _build(self):
        hdr = ttk.Frame(self, style="F.TFrame")
        hdr.pack(fill="x", padx=18, pady=(10,4))
        ttk.Label(hdr, text="Prop Firm  Monte Carlo Simulator", style="H1.TLabel").pack(side="left")
        ttk.Label(hdr, text="  $2,000 trailing DD  ·  $3,000 target  ·  5 qualifying days @ $150+  ·  $400 daily limit  ·  30-day timeout",
                  style="Sub.TLabel").pack(side="left")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=(0,8))

        self.tab_mc   = ttk.Frame(nb, style="F.TFrame")
        self.tab_wf   = ttk.Frame(nb, style="F.TFrame")
        self.tab_sens = ttk.Frame(nb, style="F.TFrame")
        self.tab_alg  = ttk.Frame(nb, style="F.TFrame")

        nb.add(self.tab_mc,   text="  📊  Monte Carlo  ")
        nb.add(self.tab_wf,   text="  🔄  Walk Forward  ")
        nb.add(self.tab_sens, text="  🎯  Sensitivity  ")
        nb.add(self.tab_alg,  text="  ⚙  Algorithm Config  ")

        self._build_mc_tab()
        self._build_wf_tab()
        self._build_sens_tab()
        self._build_alg_tab()

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — MONTE CARLO
    # ══════════════════════════════════════════════════════════════
    def _build_mc_tab(self):
        tab = self.tab_mc
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        # Scrollable controls panel
        ctrl, _cvs_mc = self._make_scroll_ctrl(tab, width=314)

        self._sec(ctrl, "Strategy Preset", info_key="monte_carlo_tab")
        pr = ttk.Frame(ctrl, style="F.TFrame"); pr.pack(fill="x", padx=10, pady=(0,4))
        self.preset_v = tk.StringVar(value="Kalman pairs  (MES/MNQ)")
        cb = ttk.Combobox(pr, textvariable=self.preset_v, values=list(STRATEGY_PRESETS.keys()), state="readonly", width=25)
        cb.pack(side="left", fill="x", expand=True)
        cb.bind("<<ComboboxSelected>>", self._on_preset)
        ttk.Button(pr, text="Details", width=7, command=self._show_strategy_popup).pack(side="left", padx=(4,0))
        self.strategy_lbl = ttk.Label(ctrl, text="", style="Sub.TLabel", justify="left", wraplength=285)
        self.strategy_lbl.pack(fill="x", padx=14, pady=(0,8))

        self._sec(ctrl, "Micro Futures Mode", info_key="futures_mode")
        self.instrument_v = tk.StringVar(value="MES")
        ir = ttk.Frame(ctrl, style="F.TFrame"); ir.pack(fill="x", padx=10, pady=(0,4))
        ttk.Label(ir, text="Instrument", style="Sub.TLabel", width=11).pack(side="left")
        icb = ttk.Combobox(ir, textvariable=self.instrument_v, values=list(CONTRACT_SPECS.keys()), state="readonly", width=12)
        icb.pack(side="left"); icb.bind("<<ComboboxSelected>>", self._on_instrument)
        self.contracts_v = tk.IntVar(value=1)
        self._sl(ctrl, "Contracts / trade", self.contracts_v, 1, MAX_CONTRACTS, fmt="d", res=1.0)
        self.inst_lbl = ttk.Label(ctrl, text="", style="Sub.TLabel", justify="left", wraplength=285)
        self.inst_lbl.pack(fill="x", padx=14, pady=(0,6))

        self._sec(ctrl, "Trade Parameters", info_key="trade_params")
        self.wr_v = tk.DoubleVar(value=60)
        self.aw_v = tk.DoubleVar(value=55)
        self.al_v = tk.DoubleVar(value=72)
        self.td_v = tk.DoubleVar(value=4)
        self._sl(ctrl, "Win rate  (%)",       self.wr_v, 25, 80)
        self._sl(ctrl, "Avg win / ctr ($)",   self.aw_v, 10, 300)
        self._sl(ctrl, "Avg loss / ctr ($)",  self.al_v, 10, 300)
        self._sl(ctrl, "Trades / day",        self.td_v, 1,  15, fmt="d")

        # ── Market Regime Filter ──────────────────────────────────
        self._sec(ctrl, "Market Regime Filter", info_key="regime_select")
        self.regime_v = tk.StringVar(value="No filter (mixed)")
        regime_cb = ttk.Combobox(ctrl, textvariable=self.regime_v,
                                  values=list(REGIME_DEFS.keys()), state="readonly", width=30)
        regime_cb.pack(fill="x", padx=10, pady=(0,4))
        regime_cb.bind("<<ComboboxSelected>>", self._on_regime_change)

        # Effective params strip (shown for fixed regimes)
        self.regime_eff_lbl = ttk.Label(ctrl, text="", style="Sub.TLabel",
                                         wraplength=285, justify="left")
        self.regime_eff_lbl.pack(fill="x", padx=14, pady=(0,2))

        # Blend weight frame (only visible in Custom blend mode)
        self.blend_frame = ttk.Frame(ctrl, style="F.TFrame")
        self.blend_frame.pack(fill="x", padx=10, pady=(2,4))
        self.blend_trending_v  = tk.IntVar(value=25)
        self.blend_choppy_v    = tk.IntVar(value=40)
        self.blend_volatile_v  = tk.IntVar(value=20)
        self.blend_baseline_v  = tk.IntVar(value=15)
        self._blend_sl(self.blend_frame, "Trending  (%)", self.blend_trending_v)
        self._blend_sl(self.blend_frame, "Choppy    (%)", self.blend_choppy_v)
        self.blend_frame.pack_forget()  # hidden by default

        self.blend_frame2 = ttk.Frame(ctrl, style="F.TFrame")
        self.blend_frame2.pack(fill="x", padx=10, pady=(0,4))
        self._blend_sl(self.blend_frame2, "Volatile  (%)", self.blend_volatile_v)
        self._blend_sl(self.blend_frame2, "Baseline  (%)", self.blend_baseline_v)
        self.blend_total_lbl = ttk.Label(self.blend_frame2, text="", style="Sub.TLabel")
        self.blend_total_lbl.pack(fill="x", padx=2, pady=(2,0))
        self.blend_frame2.pack_forget()  # hidden by default

        # Bind blend sliders to total validator
        for bv in (self.blend_trending_v, self.blend_choppy_v,
                    self.blend_volatile_v, self.blend_baseline_v):
            bv.trace_add("write", self._on_blend_change)

        self._sec(ctrl, "Execution Friction", info_key="exec_friction")
        self.commission_v  = tk.DoubleVar(value=0.74)
        self.slippage_v    = tk.DoubleVar(value=0.25)
        self._sl(ctrl, "Commission RT ($)", self.commission_v, 0.0, 5.0, fmt=".2f", res=0.01)
        self._sl(ctrl, "Slippage (ticks)",  self.slippage_v,   0.0, 4.0, fmt=".2f", res=0.05)
        evr = ttk.Frame(ctrl, style="F.TFrame"); evr.pack(fill="x", padx=10, pady=(4,8))
        self.ev_lbl  = ttk.Label(evr, text="EV / trade:  —", style="EV.TLabel"); self.ev_lbl.pack(side="left")
        self.rr_lbl  = ttk.Label(evr, text="R/R: —", style="Sub.TLabel"); self.rr_lbl.pack(side="right")

        self._sec(ctrl, "Position Sizing", info_key="position_sizing")
        self.sizing_v = tk.StringVar(value="Risk geometry")
        sz = ttk.Combobox(ctrl, textvariable=self.sizing_v, values=SIZING_OPTS, state="readonly", width=30)
        sz.pack(fill="x", padx=10, pady=(0,4))
        sz.bind("<<ComboboxSelected>>", self._on_sizing)
        self.k_v = tk.DoubleVar(value=1.5)
        self.k_row = self._sl(ctrl, "Curve exponent  k", self.k_v, 0.2, 4.0, fmt=".1f", res=0.1)
        ttk.Label(ctrl, text="k < 1 → aggressive reduction   k > 1 → gradual", style="Sub.TLabel").pack(anchor="w", padx=14, pady=(0,8))

        self._sec(ctrl, "Calibration Confidence", info_key="calibration_confidence")
        ttk.Label(ctrl, text="Controls parameter estimation uncertainty.\n'None' = exact params (unrealistically high pass rates).",
                  style="Sub.TLabel", wraplength=285, justify="left").pack(fill="x", padx=14, pady=(0,4))
        self.calib_v = tk.StringVar(value="Typical  (±5pp WR, ±20% aw/al)")
        calib_cb = ttk.Combobox(ctrl, textvariable=self.calib_v, state="readonly", width=30,
                                 values=["None  (exact parameters — textbook)",
                                         "Tight  (±2pp WR, ±10% aw/al)",
                                         "Typical  (±5pp WR, ±20% aw/al)",
                                         "Wide  (±10pp WR, ±35% aw/al)",
                                         "Conservative  (±15pp WR, ±50% aw/al)"])
        calib_cb.pack(fill="x", padx=10, pady=(0,6))

        self._sec(ctrl, "Simulations")
        sr = ttk.Frame(ctrl, style="F.TFrame"); sr.pack(fill="x", padx=10, pady=(0,6))
        ttk.Label(sr, text="Runs:", style="Sub.TLabel").pack(side="left")
        self.nsim_v = tk.IntVar(value=1000)
        ttk.Spinbox(sr, from_=200, to=5000, increment=200, textvariable=self.nsim_v, width=8, font=("Helvetica",10)).pack(side="left", padx=(8,0))
        self.run_mc_btn = ttk.Button(ctrl, text="▶  Run Monte Carlo", style="Run.TButton", command=self._run_mc)
        self.run_mc_btn.pack(fill="x", padx=10, pady=(4,4))
        br2 = ttk.Frame(ctrl, style="F.TFrame"); br2.pack(fill="x", padx=10, pady=(0,4))
        ttk.Button(br2, text="Strategy guide", command=self._show_strategy_popup).pack(side="left", fill="x", expand=True)
        ttk.Button(br2, text="Sizing guide",   command=self._show_sizing_popup).pack(side="left", fill="x", expand=True, padx=(4,0))
        self.mc_status = ttk.Label(ctrl, text="Ready", style="Sub.TLabel"); self.mc_status.pack(padx=10, pady=(0,8))

        self._sec(ctrl, "Challenge Rules  (fixed)")
        for line in [
            f"Profit target   ${PROFIT_TARGET:,.0f}",
            f"Trailing DD     ${DD_LIMIT:,.0f}",
            f"Daily loss lim  ${DAILY_LOSS_LIM:,.0f}",
            f"Qual days       {QUAL_DAYS_NEED} × ${QUAL_MIN:.0f}+",
            f"Max days        {MAX_DAYS}",
            f"Max contracts   {MAX_CONTRACTS}",
        ]:
            ttk.Label(ctrl, text=line, style="Sub.TLabel", font=("Courier",9)).pack(anchor="w", padx=14, pady=1)

        # Charts area
        cf = ttk.Frame(tab, style="F.TFrame")
        cf.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        self.fig_mc = Figure(facecolor=C["bg"])
        self.fig_mc.subplots_adjust(left=0.06, right=0.975, top=0.94, bottom=0.08, hspace=0.42, wspace=0.28)
        gs = self.fig_mc.add_gridspec(3, 2)
        self.ax_paths = self.fig_mc.add_subplot(gs[0,0])
        self.ax_dist  = self.fig_mc.add_subplot(gs[0,1])
        self.ax_geom  = self.fig_mc.add_subplot(gs[1,0])
        self.ax_days  = self.fig_mc.add_subplot(gs[1,1])
        self.ax_daily = self.fig_mc.add_subplot(gs[2,0])
        self.ax_sweep = self.fig_mc.add_subplot(gs[2,1])
        self._style_all_axes(self.fig_mc)
        self.canvas_mc = FigureCanvasTkAgg(self.fig_mc, master=cf)
        self.canvas_mc.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_mc.mpl_connect("pick_event", self._on_chart_pick)

        # Stats bar
        sbar = ttk.Frame(tab, style="P.TFrame")
        sbar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(2,6))
        specs = [
            ("s_pass","Pass rate","StatG.TLabel"),("s_bust","Bust rate","StatR.TLabel"),
            ("s_to","Timeout rate","Stat.TLabel"),("s_days","Avg days (pass)","Stat.TLabel"),
            ("s_ev","EV / trade","StatA.TLabel"),("s_qual","Qualifying days","Stat.TLabel"),
        ]
        self._sl_map: Dict[str,ttk.Label] = {}
        for key,lbl,style in specs:
            card = ttk.Frame(sbar, style="P.TFrame"); card.pack(side="left", expand=True, fill="both", padx=6, pady=6)
            ttk.Label(card, text=lbl, style="StatLbl.TLabel").pack(anchor="w", padx=10, pady=(4,0))
            w = ttk.Label(card, text="—", style=style); w.pack(anchor="w", padx=10, pady=(0,4))
            self._sl_map[key] = w

        self._update_strategy_summary()
        self._update_instrument_summary()

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — WALK FORWARD
    # ══════════════════════════════════════════════════════════════
    def _build_wf_tab(self):
        tab = self.tab_wf
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        # Scrollable controls panel
        ctrl, _cvs_wf = self._make_scroll_ctrl(tab, width=314)

        self._sec(ctrl, "Walk Forward Testing", info_key="wf_main")
        ttk.Label(ctrl, text="Uses parameters from Monte Carlo tab.\nRun MC first to set your parameters.",
                  style="Sub.TLabel", wraplength=285, justify="left").pack(fill="x", padx=14, pady=(0,8))

        self.wf_folds_v   = tk.IntVar(value=6)
        self.wf_noise_v   = tk.DoubleVar(value=10.0)
        self.wf_deg_v     = tk.DoubleVar(value=15.0)
        self.wf_oos_v     = tk.DoubleVar(value=30.0)
        self.wf_nsim_v    = tk.IntVar(value=400)

        self._sl(ctrl, "Folds (N)",           self.wf_folds_v, 3, 12, fmt="d", res=1.0)
        self._sl(ctrl, "Param noise (%)",     self.wf_noise_v, 0, 35, fmt=".0f", res=1.0)
        self._sl(ctrl, "OOS degradation (%)", self.wf_deg_v,   0, 40, fmt=".0f", res=1.0)
        self._sl(ctrl, "OOS fraction (%)",    self.wf_oos_v,   20, 50, fmt=".0f", res=5.0)

        ttk.Label(ctrl, text=(
            "Param noise: fold-to-fold regime variation\n"
            "OOS degradation: live IS→OOS edge decay\n"
            "OOS fraction: % of sims in OOS window"
        ), style="Sub.TLabel", wraplength=285, justify="left").pack(fill="x", padx=14, pady=(0,8))

        sr2 = ttk.Frame(ctrl, style="F.TFrame"); sr2.pack(fill="x", padx=10, pady=(0,6))
        ttk.Label(sr2, text="Sims/fold:", style="Sub.TLabel").pack(side="left")
        ttk.Spinbox(sr2, from_=100, to=2000, increment=100, textvariable=self.wf_nsim_v, width=8, font=("Helvetica",10)).pack(side="left", padx=(8,0))

        self.run_wf_btn = ttk.Button(ctrl, text="▶  Run Walk Forward", style="Run.TButton", command=self._run_wf)
        self.run_wf_btn.pack(fill="x", padx=10, pady=(4,4))
        self.wf_status = ttk.Label(ctrl, text="Ready — run Monte Carlo first", style="Sub.TLabel", wraplength=280)
        self.wf_status.pack(padx=10, pady=(0,8))

        # WFE results block
        self._sec(ctrl, "Walk Forward Results")
        self.wfe_frame = ttk.Frame(ctrl, style="P.TFrame")
        self.wfe_frame.pack(fill="x", padx=10, pady=(0,8))
        metrics = [("wf_wfe","WFE Score","StatA.TLabel"),("wf_stability","Stability","StatG.TLabel"),("wf_is","IS Pass Rate","Stat.TLabel"),("wf_oos","OOS Pass Rate","Stat.TLabel")]
        self._wf_map: Dict[str,ttk.Label] = {}
        for key,lbl,style in metrics:
            row = ttk.Frame(self.wfe_frame, style="P.TFrame"); row.pack(fill="x", padx=6, pady=2)
            ttk.Label(row, text=lbl+":", style="StatLbl.TLabel", width=16).pack(side="left")
            w = ttk.Label(row, text="—", style=style, font=("Helvetica",13,"bold")); w.pack(side="left")
            self._wf_map[key] = w

        self._sec(ctrl, "Regime Stress Controls", info_key="regime_stress")
        self.regime_nsim_v = tk.IntVar(value=400)
        rs3 = ttk.Frame(ctrl, style="F.TFrame"); rs3.pack(fill="x", padx=10, pady=(0,4))
        ttk.Label(rs3, text="Sims:", style="Sub.TLabel").pack(side="left")
        ttk.Spinbox(rs3, from_=100, to=2000, increment=100, textvariable=self.regime_nsim_v, width=8, font=("Helvetica",10)).pack(side="left", padx=(8,0))
        ttk.Button(ctrl, text="▶  Run Regime Stress Test", command=self._run_regime).pack(fill="x", padx=10, pady=(0,4))
        self.regime_status = ttk.Label(ctrl, text="Ready", style="Sub.TLabel"); self.regime_status.pack(padx=10)

        self._sec(ctrl, "WFE Interpretation", info_key="wf_main")
        for line in ["WFE ≥ 0.80  → Excellent", "WFE 0.65–0.80 → Good", "WFE 0.50–0.65 → Marginal", "WFE < 0.50  → Poor"]:
            ttk.Label(ctrl, text=line, style="Sub.TLabel", font=("Courier",9)).pack(anchor="w", padx=14, pady=1)

        # Charts
        cf = ttk.Frame(tab, style="F.TFrame")
        cf.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        self.fig_wf = Figure(facecolor=C["bg"])
        self.fig_wf.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.08, hspace=0.46, wspace=0.30)
        gs2 = self.fig_wf.add_gridspec(3, 2)
        self.ax_wf_folds   = self.fig_wf.add_subplot(gs2[0,0])
        self.ax_wf_wfe     = self.fig_wf.add_subplot(gs2[0,1])
        self.ax_wf_regime  = self.fig_wf.add_subplot(gs2[1,0])
        self.ax_wf_equity  = self.fig_wf.add_subplot(gs2[1,1])
        self.ax_wf_deg     = self.fig_wf.add_subplot(gs2[2,0])
        self.ax_wf_stab    = self.fig_wf.add_subplot(gs2[2,1])
        self._style_all_axes(self.fig_wf)
        self.canvas_wf = FigureCanvasTkAgg(self.fig_wf, master=cf)
        self.canvas_wf.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_wf.mpl_connect("pick_event", self._on_chart_pick)

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — SENSITIVITY
    # ══════════════════════════════════════════════════════════════
    def _build_sens_tab(self):
        tab = self.tab_sens
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        # Scrollable controls panel
        ctrl, _cvs_sens = self._make_scroll_ctrl(tab, width=314)

        self._sec(ctrl, "Parameter Sensitivity", info_key="sensitivity_main")
        ttk.Label(ctrl, text="Uses parameters from Monte Carlo tab.\nRun MC first to establish your baseline.",
                  style="Sub.TLabel", wraplength=285, justify="left").pack(fill="x", padx=14, pady=(0,8))

        self._sec(ctrl, "Tornado Chart", info_key="tornado")
        ttk.Label(ctrl, text="Sweeps all parameters LOW→HIGH.\nAll others held at baseline.",
                  style="Sub.TLabel", wraplength=285, justify="left").pack(fill="x", padx=14, pady=(0,6))
        self.tornado_nsim_v = tk.IntVar(value=200)
        tr = ttk.Frame(ctrl, style="F.TFrame"); tr.pack(fill="x", padx=10, pady=(0,6))
        ttk.Label(tr, text="Sims/point:", style="Sub.TLabel").pack(side="left")
        ttk.Spinbox(tr, from_=50, to=500, increment=50, textvariable=self.tornado_nsim_v, width=7, font=("Helvetica",10)).pack(side="left", padx=(8,0))
        self.run_tornado_btn = ttk.Button(ctrl, text="▶  Run Tornado Chart", style="Run.TButton", command=self._run_tornado)
        self.run_tornado_btn.pack(fill="x", padx=10, pady=(0,4))

        self._sec(ctrl, "Sensitivity Curve", info_key="sens_curve")
        self.sens_param_v = tk.StringVar(value="Win rate (%)")
        ttk.Combobox(ctrl, textvariable=self.sens_param_v, values=list(SENS_PARAMS.keys()), state="readonly", width=28).pack(fill="x", padx=10, pady=(0,4))
        self.sens_pts_v   = tk.IntVar(value=20)
        self.sens_nsim_v  = tk.IntVar(value=200)
        sr3 = ttk.Frame(ctrl, style="F.TFrame"); sr3.pack(fill="x", padx=10, pady=(0,4))
        ttk.Label(sr3, text="Points:", style="Sub.TLabel").pack(side="left")
        ttk.Spinbox(sr3, from_=10, to=40, increment=5, textvariable=self.sens_pts_v, width=5, font=("Helvetica",10)).pack(side="left", padx=(6,0))
        ttk.Label(sr3, text="  Sims:", style="Sub.TLabel").pack(side="left")
        ttk.Spinbox(sr3, from_=100, to=500, increment=100, textvariable=self.sens_nsim_v, width=6, font=("Helvetica",10)).pack(side="left", padx=(4,0))
        self.run_curve_btn = ttk.Button(ctrl, text="▶  Run Sensitivity Curve", style="Run.TButton", command=self._run_sens_curve)
        self.run_curve_btn.pack(fill="x", padx=10, pady=(0,4))

        self._sec(ctrl, "2-D Heatmap", info_key="heatmap")
        ttk.Label(ctrl, text="Select two parameters to vary simultaneously.\nAll others held at baseline.",
                  style="Sub.TLabel", wraplength=285, justify="left").pack(fill="x", padx=14, pady=(0,6))
        self.hm_x_v = tk.StringVar(value="Win rate (%)")
        self.hm_y_v = tk.StringVar(value="Avg win / ctr ($)")
        px_row = ttk.Frame(ctrl, style="F.TFrame"); px_row.pack(fill="x", padx=10, pady=(0,4))
        ttk.Label(px_row, text="X axis:", style="Sub.TLabel", width=8).pack(side="left")
        ttk.Combobox(px_row, textvariable=self.hm_x_v, values=list(SENS_PARAMS.keys()), state="readonly", width=22).pack(side="left")
        py_row = ttk.Frame(ctrl, style="F.TFrame"); py_row.pack(fill="x", padx=10, pady=(0,4))
        ttk.Label(py_row, text="Y axis:", style="Sub.TLabel", width=8).pack(side="left")
        ttk.Combobox(py_row, textvariable=self.hm_y_v, values=list(SENS_PARAMS.keys()), state="readonly", width=22).pack(side="left")
        hm_ctrl = ttk.Frame(ctrl, style="F.TFrame"); hm_ctrl.pack(fill="x", padx=10, pady=(0,4))
        ttk.Label(hm_ctrl, text="Grid:", style="Sub.TLabel").pack(side="left")
        self.hm_grid_v = tk.IntVar(value=10)
        ttk.Spinbox(hm_ctrl, from_=6, to=16, increment=2, textvariable=self.hm_grid_v, width=5, font=("Helvetica",10)).pack(side="left", padx=(6,0))
        ttk.Label(hm_ctrl, text="  Sims/cell:", style="Sub.TLabel").pack(side="left")
        self.hm_nsim_v = tk.IntVar(value=100)
        ttk.Spinbox(hm_ctrl, from_=50, to=300, increment=50, textvariable=self.hm_nsim_v, width=6, font=("Helvetica",10)).pack(side="left", padx=(4,0))
        self.run_hmap_btn = ttk.Button(ctrl, text="▶  Run Heatmap", style="Run.TButton", command=self._run_heatmap)
        self.run_hmap_btn.pack(fill="x", padx=10, pady=(0,4))

        self.sens_status = ttk.Label(ctrl, text="Ready — run tools above", style="Sub.TLabel", wraplength=280)
        self.sens_status.pack(padx=10, pady=(0,8))

        self._sec(ctrl, "Robustness Score")
        self.robustness_lbl = ttk.Label(ctrl, text="Run Tornado to calculate", style="Sub.TLabel", wraplength=280)
        self.robustness_lbl.pack(fill="x", padx=14, pady=(0,8))

        # Charts
        cf = ttk.Frame(tab, style="F.TFrame")
        cf.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        self.fig_sens = Figure(facecolor=C["bg"])
        self.fig_sens.subplots_adjust(left=0.1, right=0.97, top=0.94, bottom=0.08, hspace=0.46, wspace=0.32)
        gs3 = self.fig_sens.add_gridspec(3, 2)
        self.ax_tornado = self.fig_sens.add_subplot(gs3[0, :])   # full width
        self.ax_curve   = self.fig_sens.add_subplot(gs3[1, 0])
        self.ax_hmap    = self.fig_sens.add_subplot(gs3[1, 1])
        self.ax_pf      = self.fig_sens.add_subplot(gs3[2, 0])
        self.ax_robust  = self.fig_sens.add_subplot(gs3[2, 1])
        self._style_all_axes(self.fig_sens)
        self.canvas_sens = FigureCanvasTkAgg(self.fig_sens, master=cf)
        self.canvas_sens.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_sens.mpl_connect("pick_event", self._on_chart_pick)

    # ══════════════════════════════════════════════════════════════
    # SHARED UI HELPERS
    # ══════════════════════════════════════════════════════════════
    def _sec(self, parent, text: str, info_key: str = None):
        f = ttk.Frame(parent, style="F.TFrame")
        f.pack(fill="x", padx=10, pady=(10,2))
        ttk.Label(f, text=text.upper(), style="Sec.TLabel").pack(side="left", anchor="w")
        if info_key and info_key in INFO:
            btn = ttk.Button(f, text=" ℹ ", style="Info.TButton",
                             command=lambda k=info_key: self._popup(k, INFO[k]))
            btn.pack(side="left", padx=(4,0))
        tk.Canvas(parent, height=1, bg=C["border"], highlightthickness=0).pack(fill="x", padx=10, pady=(2,4))

    def _sl(self, parent, label: str, var: tk.Variable, lo, hi, fmt=".0f", res=1.0):
        row = ttk.Frame(parent, style="F.TFrame"); row.pack(fill="x", padx=10, pady=2)
        ttk.Label(row, text=label, style="Sub.TLabel", width=18).pack(side="left")
        val_lbl = ttk.Label(row, style="Sub.TLabel", width=7, anchor="e"); val_lbl.pack(side="right")
        def _upd(*_):
            v = var.get()
            val_lbl.config(text=str(int(round(v))) if fmt=="d" else f"{v:{fmt}}")
            self._on_param_change()
        var.trace_add("write", _upd)
        ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal", style="Horizontal.TScale",
                  command=lambda v, r=res, va=var: va.set(round(float(v)/r)*r)
                  ).pack(side="left", fill="x", expand=True, padx=(4,4))
        _upd()
        return row

    def _ax_base(self, ax, title: str, subtitle: str = ""):
        ax.clear()
        ax.set_facecolor(C["bg"])
        full_title = f"{title}  —  {subtitle}" if subtitle else title
        ax.set_title(full_title, fontsize=8.5, color=C["text"], fontweight="bold", pad=5)
        ax.tick_params(colors=C["muted"], labelsize=7.5)
        ax.xaxis.label.set_color(C["muted"]); ax.yaxis.label.set_color(C["muted"])
        for sp in ax.spines.values(): sp.set_edgecolor(C["border"]); sp.set_linewidth(0.5)

    def _style_all_axes(self, fig):
        for ax in fig.get_axes():
            ax.set_facecolor(C["bg"])
            ax.tick_params(colors=C["muted"], labelsize=7.5)
            for sp in ax.spines.values(): sp.set_edgecolor(C["border"]); sp.set_linewidth(0.5)

    def _popup(self, title_key: str, body: str):
        display_title = title_key.replace("_", " ").title()
        top = tk.Toplevel(self); top.title(display_title)
        top.configure(bg=C["bg"]); top.geometry("760x680")
        top.transient(self); top.grab_set()
        container = ttk.Frame(top, style="F.TFrame"); container.pack(fill="both", expand=True, padx=12, pady=12)
        txt = tk.Text(container, wrap="word", bg="white", fg=C["text"], relief="flat", font=("Helvetica",10), padx=10, pady=10)
        scroll = ttk.Scrollbar(container, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=scroll.set)
        txt.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
        txt.insert("1.0", body); txt.config(state="disabled")
        ttk.Button(top, text="Close", command=top.destroy).pack(pady=(0,12))

    def _show_strategy_popup(self):
        pieces = ["Strategy & Regime Guide\n════════════════════════════\n"]
        for name, info in STRATEGY_LIBRARY.items():
            pieces.append(f"► {name}")
            pieces.append(f"  Type: {info['type']}")
            pieces.append(f"  Edge: {info['what_it_entails']}")
            pieces.append(f"  Win rate band: {info['win_rate_band']}")
            pieces.append(f"  Risk profile: {info['risk_profile']}")
            pieces.append(f"  Best regime: {info['best_regime']}")
            pieces.append(f"  Notes: {info['notes']}\n")
        for sym in ("MES","MNQ"):
            s=CONTRACT_SPECS[sym]
            pieces.append(f"► {sym} — {s['name']}")
            pieces.append(f"  ${s['point_value']:.2f}/pt  |  {s['tick_size']}pt tick = ${s['tick_value']:.2f}  |  {s['context']}\n")
        self._popup("strategy_and_regime_guide", "\n".join(pieces))

    def _show_sizing_popup(self):
        pieces = ["Sizing & Risk Geometry Guide\n════════════════════════════\n"]
        for name, info in SIZING_LIBRARY.items():
            pieces.append(f"► {name}")
            pieces.append(f"  {info['descriptor']}")
            pieces.append(f"  Geometry: {info['geometry']}")
            pieces.append(f"  Risk: {info['risk_note']}\n")
        self._popup("sizing_guide", "\n".join(pieces))

    # ══════════════════════════════════════════════════════════════
    # STATE / PARAM HELPERS
    # ══════════════════════════════════════════════════════════════
    def _blend_sl(self, parent, label: str, var: tk.IntVar):
        """Compact integer slider for blend weights."""
        row = ttk.Frame(parent, style="F.TFrame"); row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, style="Sub.TLabel", width=14).pack(side="left")
        val_lbl = ttk.Label(row, style="Sub.TLabel", width=4, anchor="e"); val_lbl.pack(side="right")
        def _upd(*_):
            val_lbl.config(text=f"{var.get()}%")
        var.trace_add("write", _upd); _upd()
        ttk.Scale(row, from_=0, to=100, variable=var, orient="horizontal",
                  command=lambda v,va=var: va.set(max(0,min(100,int(round(float(v))))))).pack(
                  side="left", fill="x", expand=True, padx=(4,4))

    def _on_regime_change(self, _=None):
        regime = self.regime_v.get()
        is_blend = (regime == "Custom blend")
        if is_blend:
            self.blend_frame.pack(fill="x", padx=10, pady=(2,0))
            self.blend_frame2.pack(fill="x", padx=10, pady=(0,4))
            self.regime_eff_lbl.config(text="Each session day samples from your blend weights below.")
        else:
            self.blend_frame.pack_forget()
            self.blend_frame2.pack_forget()
            self._update_regime_eff_label()

    def _on_blend_change(self, *_):
        total = (self.blend_trending_v.get() + self.blend_choppy_v.get() +
                 self.blend_volatile_v.get() + self.blend_baseline_v.get())
        col = C["win"] if total == 100 else C["bust"]
        self.blend_total_lbl.config(text=f"Total: {total}%  {'✓ OK' if total==100 else '⚠ must equal 100%'}",
                                     foreground=col)

    def _update_regime_eff_label(self):
        regime = self.regime_v.get()
        if regime == "No filter (mixed)":
            self.regime_eff_lbl.config(text="No adjustment — base params used as-is.")
            return
        try:
            r = REGIME_DEFS[regime]
            wr = float(np.clip(self.wr_v.get() + r["wr_d"], 10, 95))
            aw = self.aw_v.get() * r["aw_m"]
            al = self.al_v.get() * r["al_m"]
            td = int(np.clip(self.td_v.get() + r["td_d"], 1, MAX_TRADES_DAY))
            self.regime_eff_lbl.config(
                text=f"Effective → WR: {wr:.0f}%  AW: ${aw:.0f}  AL: ${al:.0f}  TD: {td}")
        except Exception:
            pass

    def _get_regime_blend(self) -> Optional[Dict[str, float]]:
        """Return blend weights dict if in custom blend mode and weights sum to ~100."""
        if self.regime_v.get() != "Custom blend":
            return None
        t = float(self.blend_trending_v.get() + self.blend_choppy_v.get() +
                  self.blend_volatile_v.get() + self.blend_baseline_v.get())
        if t <= 0:
            return None
        return {
            "Trending":          self.blend_trending_v.get() / t,
            "Choppy":            self.blend_choppy_v.get()   / t,
            "Volatile":          self.blend_volatile_v.get() / t,
            "No filter (mixed)": self.blend_baseline_v.get() / t,
        }

    def _get_calib_sigma(self) -> tuple:
        """Return (sigma_wr_pp, sigma_payoff_frac) for calibration confidence level."""
        level = self.calib_v.get()
        # Returns: (win rate σ in pp, aw/al fractional σ)
        if "None" in level:         return (0.0,  0.00)
        elif "Tight" in level:      return (2.0,  0.10)
        elif "Typical" in level:    return (5.0,  0.20)
        elif "Wide" in level:       return (10.0, 0.35)
        else:                       return (15.0, 0.50)  # Conservative

    def _get_base_params(self) -> dict:
        return dict(
            wr=self.wr_v.get(), aw=self.aw_v.get(), al=self.al_v.get(),
            td=int(round(self.td_v.get())), sizing=self.sizing_v.get(),
            k=self.k_v.get(), instrument=self.instrument_v.get(),
            contracts=int(round(self.contracts_v.get())),
            commission_rt=self.commission_v.get(), slippage_ticks=self.slippage_v.get(),
        )

    def _on_param_change(self):
        try:
            wr=self.wr_v.get()/100; aw=self.aw_v.get(); al=self.al_v.get()
            c=int(round(self.contracts_v.get()))
            cost=c*_trade_cost(self.instrument_v.get(),self.commission_v.get(),self.slippage_v.get())
            ev=c*(wr*aw-(1-wr)*al)-cost
            rr=aw/al if al>0 else 0
            sign="+" if ev>=0 else ""
            col=C["win"] if ev>0 else (C["amber"] if ev==0 else C["bust"])
            self.ev_lbl.config(text=f"EV / trade:  {sign}${ev:.1f}", foreground=col)
            self.rr_lbl.config(text=f"R/R: {rr:.2f}")
            self._update_regime_eff_label()
        except Exception: pass

    def _on_preset(self, _=None):
        p = STRATEGY_PRESETS.get(self.preset_v.get(), {})
        if p:
            self.wr_v.set(p["wr"]); self.aw_v.set(p["aw"])
            self.al_v.set(p["al"]); self.td_v.set(p["td"])
        self._update_strategy_summary()

    def _on_instrument(self, _=None):
        self._update_instrument_summary(); self._on_param_change()

    def _on_sizing(self, _=None):
        is_geom = self.sizing_v.get() == "Risk geometry"
        for w in self.k_row.winfo_children():
            try: w.configure(state="normal" if is_geom else "disabled")
            except Exception: pass
        self._redraw_geom()

    def _update_strategy_summary(self):
        info = STRATEGY_LIBRARY.get(self.preset_v.get(), {})
        self.strategy_lbl.config(text=f"{info.get('type','')}\nWin band: {info.get('win_rate_band','')}")

    def _update_instrument_summary(self):
        spec = CONTRACT_SPECS[self.instrument_v.get()]
        self.inst_lbl.config(text=f"{spec['name']}  ·  ${spec['point_value']:.2f}/pt  ·  ${spec['tick_value']:.2f}/tick\n{spec['contract_note']}")

    # ══════════════════════════════════════════════════════════════
    # MC TAB — RUN + DRAW
    # ══════════════════════════════════════════════════════════════
    def _run_mc(self):
        self.run_mc_btn.config(state="disabled"); self.mc_status.config(text="Running…"); self.update_idletasks()
        bp     = self._get_base_params()
        regime    = self.regime_v.get()
        blend     = self._get_regime_blend()
        sig_wr, sig_pay = self._get_calib_sigma()
        results, sample = run_mc(
            bp["wr"], bp["aw"], bp["al"], bp["td"], bp["sizing"], bp["k"],
            bp["instrument"], bp["contracts"], bp["commission_rt"], bp["slippage_ticks"],
            self.nsim_v.get(), regime=regime, blend_weights=blend,
            sigma_wr=sig_wr, sigma_pay=sig_pay,
        )
        stats   = compute_stats(results, self.nsim_v.get())
        sweep_n = min(150, max(60, self.nsim_v.get()//8))
        sweep   = sweep_contracts(bp["wr"],bp["aw"],bp["al"],bp["td"],bp["sizing"],bp["k"],
                                   bp["instrument"],bp["commission_rt"],bp["slippage_ticks"],sweep_n)
        self._mc_draw(sample, stats, sweep, bp, regime)
        self.canvas_mc.draw_idle()
        self._update_mc_stats(stats, bp)
        regime_tag  = "" if regime == "No filter (mixed)" else f"  [{regime}]"
        calib_short = self.calib_v.get().split("(")[0].strip()
        calib_tag   = "" if "None" in calib_short else f"  · {calib_short}"
        self.mc_status.config(text=f"{self.nsim_v.get():,} sims complete{regime_tag}{calib_tag}")
        self.run_mc_btn.config(state="normal")

    def _mc_draw(self, sample, stats, sweep, bp, regime="No filter (mixed)"):
        self._draw_paths(sample, stats, regime)
        self._draw_outcomes(stats)
        self._redraw_geom()
        self._draw_days(stats)
        self._draw_daily_pnl(stats)
        self._draw_sweep(sweep)

    def _draw_paths(self, sample, stats, regime="No filter (mixed)"):
        ax = self.ax_paths
        regime_tag = "" if regime == "No filter (mixed)" else f"  |  {regime}"
        # Note: with calibration uncertainty each path may use different parameters
        self._ax_base(ax, "Sample Equity Paths", f"green=pass  red=bust  grey=timeout{regime_tag}")
        col_map = {"pass":C["win"],"bust":C["bust"],"timeout":C["timeout"]}
        for r in sample:
            ax.plot(range(len(r.equity)), r.equity, color=col_map[r.outcome], alpha=0.27, lw=0.9, zorder=2)
        ax.axhline(PROFIT_TARGET, color=C["win"],  lw=1.2, ls="--", alpha=0.85, zorder=4)
        ax.axhline(-DD_LIMIT,     color=C["bust"], lw=1.2, ls="--", alpha=0.85, zorder=4)
        ax.axhline(0, color=C["muted"], lw=0.5, alpha=0.4, zorder=1)
        ax.text(0.5, PROFIT_TARGET+60,  f"Target  ${PROFIT_TARGET:,.0f}", fontsize=6.5, color=C["win"], va="bottom")
        ax.text(0.5, -DD_LIMIT-70, f"Drawdown limit  ${DD_LIMIT:,.0f}", fontsize=6.5, color=C["bust"], va="top")
        # Regime badge
        if regime != "No filter (mixed)":
            r_def = REGIME_DEFS.get(regime, {})
            col   = r_def.get("color", C["muted"])
            ax.text(0.99, 0.99, f"⬤  {regime}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=C["bg"], ec=col, lw=0.8, alpha=0.9))
        ax.legend(handles=[Patch(color=C["win"],alpha=0.55,label=f"Pass ({stats['n_pass']})"),
                            Patch(color=C["bust"],alpha=0.55,label=f"Bust ({stats['n_bust']})"),
                            Patch(color=C["timeout"],alpha=0.55,label=f"Timeout ({stats['n_timeout']})")],
                  fontsize=7,framealpha=0,loc="upper left",handlelength=1.2,borderpad=0.3)
        ax.set_xlabel("Day",fontsize=8); ax.set_ylabel("Cumulative P&L ($)",fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_:f"${x:,.0f}"))
        self._info_note(ax, "equity_paths")

    def _draw_outcomes(self, stats):
        ax = self.ax_dist
        self._ax_base(ax, "Outcome Distribution", "% of all simulations")
        vals=[stats["pass_rate"],stats["bust_rate"],stats["to_rate"]]
        bars=ax.bar(["Pass","Bust","Timeout"],vals,color=[C["win"],C["bust"],C["timeout"]],width=0.5,zorder=3,edgecolor="none")
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{v:.0f}%",ha="center",va="bottom",fontsize=9,color=C["text"],fontweight="bold")
        ax.set_ylim(0,max(vals)*1.35+5 if max(vals)>0 else 5)
        ax.set_ylabel("% of simulations",fontsize=8); ax.tick_params(labelsize=8.5)
        ax.yaxis.grid(True,color=C["border"],lw=0.4,zorder=0); ax.set_axisbelow(True)
        self._info_note(ax, "outcome_dist")

    def _redraw_geom(self):
        sizing=self.sizing_v.get(); k=self.k_v.get()
        ax = self.ax_geom
        self._ax_base(ax, "Risk Geometry", "size multiplier vs. remaining buffer")
        x = np.linspace(0,1,300)
        methods = {
            "Fixed size":        (lambda b:np.ones_like(b), C["muted"], ":", 0.7),
            "Linear scale-down": (lambda b:np.maximum(0.05,b), C["blue"], "-", 0.7),
            "Half Kelly":        (lambda b:np.maximum(0.05,0.5+0.5*b), C["win"], "-", 0.7),
            "Risk geometry":     (lambda b:np.maximum(0.05,b**(1/max(k,0.05))), C["amber"], "-", 0.7),
            "Martingale":        (lambda b:np.minimum(3.0,2.0-b), C["bust"], "--", 0.7),
        }
        for name,(fn,col,ls,_al) in methods.items():
            active=(name==sizing)
            ax.plot(x*100,fn(x),color=col,lw=(2.3 if active else 0.9),ls=ls,alpha=(1.0 if active else 0.25),label=name+("  ← active" if active else ""),zorder=(5 if active else 2))
        ax.axvspan(0,25,alpha=0.06,color=C["bust"],zorder=0)
        ax.text(12.5,0.12,"Danger\nzone",ha="center",fontsize=6,color=C["bust"],alpha=0.8)
        ax.axhline(1.0,color=C["muted"],lw=0.5,ls=":",alpha=0.6)
        ax.set_xlim(0,100); ax.set_ylim(0,3.3)
        ax.set_xlabel("Remaining DD buffer (%)",fontsize=8); ax.set_ylabel("Size multiplier",fontsize=8)
        ax.legend(fontsize=6.2,framealpha=0,ncol=1,loc="upper left",borderpad=0.3)
        ax.yaxis.grid(True,color=C["border"],lw=0.4,zorder=0); ax.set_axisbelow(True)
        self._info_note(ax,"risk_geom_chart")
        self.canvas_mc.draw_idle()

    def _draw_days(self, stats):
        ax = self.ax_days
        self._ax_base(ax, "Days to Outcome", "green = pass, red = bust")
        bins=range(1,MAX_DAYS+2,2)
        if stats["pass_days"]: ax.hist(stats["pass_days"],bins=bins,color=C["win"],alpha=0.65,label=f"Pass (n={stats['n_pass']})",zorder=3)
        if stats["bust_days"]: ax.hist(stats["bust_days"],bins=bins,color=C["bust"],alpha=0.55,label=f"Bust (n={stats['n_bust']})",zorder=2)
        if stats["avg_days"]>0:
            ax.axvline(stats["avg_days"],color=C["win"],lw=1.3,ls="--",alpha=0.9,zorder=5)
            ax.text(stats["avg_days"]+0.5,ax.get_ylim()[1]*0.9,f"avg {stats['avg_days']:.0f}d",fontsize=7,color=C["win"],va="top")
        ax.set_xlabel("Day",fontsize=8); ax.set_ylabel("Count",fontsize=8)
        handles, lbls = ax.get_legend_handles_labels()
        if handles: ax.legend(fontsize=7,framealpha=0)
        ax.yaxis.grid(True,color=C["border"],lw=0.4); ax.set_axisbelow(True)
        self._info_note(ax,"days_chart")

    def _draw_daily_pnl(self, stats):
        ax = self.ax_daily
        self._ax_base(ax, "Daily P&L Distribution", "all days, all simulations")
        vals=stats["all_daily"]
        if vals:
            bins=max(12,min(40,int(math.sqrt(len(vals)))*2))
            ax.hist(vals,bins=bins,alpha=0.72,color=C["blue"],edgecolor="none",label="Daily P&L count")
            ax.axvline(float(np.mean(vals)),color=C["amber"],lw=1.2,ls="--",label=f"Mean: ${np.mean(vals):.0f}")
            ax.axvline(0,           color=C["muted"],lw=0.7,ls=":",alpha=0.7,label="Break-even")
            ax.axvline(QUAL_MIN,    color=C["win"],  lw=0.8,ls=":",alpha=0.8,label=f"Qual threshold ${QUAL_MIN:.0f}")
            ax.axvline(-DAILY_LOSS_LIM,color=C["bust"],lw=0.8,ls=":",alpha=0.6,label=f"Daily loss limit -${DAILY_LOSS_LIM:.0f}")
            ax.text(0.02,0.97,f"mean: ${np.mean(vals):.0f}  /  std: ${np.std(vals,ddof=1):.0f}",
                    transform=ax.transAxes,fontsize=7.5,va="top",color=C["text"])
        ax.set_xlabel("Daily P&L ($)",fontsize=8); ax.set_ylabel("Count",fontsize=8)
        handles,lbls = ax.get_legend_handles_labels()
        if handles: ax.legend(fontsize=7,framealpha=0,loc="upper right")
        ax.yaxis.grid(True,color=C["border"],lw=0.4); ax.set_axisbelow(True)
        self._info_note(ax,"daily_pnl")

    def _draw_sweep(self, sweep):
        ax = self.ax_sweep
        # Remove any stale twinx axes from previous runs before calling _ax_base.
        # twinx() creates a sibling axes that persists on the figure; without removal
        # each redraw stacks another copy of the P&L line on top.
        for sibling in ax.figure.get_axes():
            if sibling is not ax and sibling.get_shared_x_axes().joined(ax, sibling):
                sibling.remove()
        self._ax_base(ax, "Contracts Sensitivity", "pass / bust rates and mean P&L vs. contract count")
        x = sweep["contracts"]
        ax.plot(x, sweep["pass_rates"], marker="o", markersize=3, lw=1.1, label="Pass rate", color=C["win"])
        ax.plot(x, sweep["bust_rates"], marker="o", markersize=3, lw=1.1, label="Bust rate", color=C["bust"])
        ax.set_xlabel("Contracts", fontsize=8); ax.set_ylabel("Rate (%)", fontsize=8)
        ax.set_xlim(1, MAX_CONTRACTS)
        max_pct = max(max(sweep["pass_rates"], default=0), max(sweep["bust_rates"], default=0))
        ax.set_ylim(0, max_pct * 1.25 + 5)
        ax.yaxis.grid(True, color=C["border"], lw=0.4); ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.plot(x, sweep["evs"], marker=".", markersize=3, lw=1.0, label="Mean P&L", color=C["amber"])
        ax2.set_ylabel("Mean final P&L ($)", fontsize=8); ax2.tick_params(colors=C["muted"], labelsize=7.5)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"${y:,.0f}"))
        for sp in ax2.spines.values(): sp.set_edgecolor(C["border"]); sp.set_linewidth(0.5)
        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=6.8, framealpha=0, loc="upper left")
        self._info_note(ax, "sweep_chart")

    def _info_note(self, ax, key: str):
        """Adds a clickable ℹ label in the top-right of the axes title area."""
        if key not in INFO:
            return
        t = ax.text(1.0, 1.02, " ℹ ", transform=ax.transAxes,
                    fontsize=9, color=C["blue"], ha="right", va="bottom",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=C["blue_l"], ec=C["blue"], lw=0.6, alpha=0.85),
                    picker=True, zorder=10)
        self._pick_map[id(t)] = key

    def _on_chart_pick(self, event):
        """Handles click on ℹ text artists embedded in chart axes."""
        key = self._pick_map.get(id(event.artist))
        if key and key in INFO:
            self._popup(key, INFO[key])

    def _update_mc_stats(self, stats, bp):
        pr=stats["pass_rate"]; br=stats["bust_rate"]
        self._sl_map["s_pass"].config(text=f"{pr:.0f}%",style="StatG.TLabel" if pr>=40 else ("StatA.TLabel" if pr>=20 else "StatR.TLabel"))
        self._sl_map["s_bust"].config(text=f"{br:.0f}%",style="StatG.TLabel" if br<=25 else ("StatA.TLabel" if br<=50 else "StatR.TLabel"))
        self._sl_map["s_to"].config(text=f"{stats['to_rate']:.0f}%")
        self._sl_map["s_days"].config(text=(f"{stats['avg_days']:.0f}" if stats['avg_days']>0 else "—"))
        wr=bp["wr"]/100; aw=bp["aw"]; al=bp["al"]; c=bp["contracts"]
        # EV formula matches simulate_one exactly: gross*contracts - cost, no instrument factor
        cost=c*_trade_cost(bp["instrument"],bp["commission_rt"],bp["slippage_ticks"])
        ev=c*(wr*aw-(1-wr)*al)-cost
        sign="+" if ev>=0 else ""
        self._sl_map["s_ev"].config(text=f"{sign}${ev:.1f}",style="StatG.TLabel" if ev>5 else ("StatA.TLabel" if ev>=0 else "StatR.TLabel"))
        self._sl_map["s_qual"].config(text=f"{stats['avg_qual']:.1f} / {QUAL_DAYS_NEED}")

    # ══════════════════════════════════════════════════════════════
    # WF TAB — RUN + DRAW
    # ══════════════════════════════════════════════════════════════
    def _run_wf(self):
        self.run_wf_btn.config(state="disabled"); self.wf_status.config(text="Running walk forward…"); self.update_idletasks()
        bp = self._get_base_params()
        wf = run_walk_forward(bp, int(self.wf_folds_v.get()), float(self.wf_noise_v.get()),
                              float(self.wf_deg_v.get()), float(self.wf_oos_v.get())/100, int(self.wf_nsim_v.get()))
        self._draw_wf_folds(wf)
        self._draw_wfe_gauge(wf)
        self._draw_wf_equity(wf)
        self._draw_wf_degradation(wf)
        self._draw_wf_stability(wf)
        self.canvas_wf.draw_idle()
        # Update results panel
        wfe_val = wf["wfe"]
        wfe_style = "StatG.TLabel" if wfe_val>=0.8 else ("StatA.TLabel" if wfe_val>=0.5 else "StatR.TLabel")
        self._wf_map["wf_wfe"].config(text=f"{wfe_val:.2f}", style=wfe_style)
        stab_val = wf["stability"]
        stab_style = "StatG.TLabel" if stab_val>=0.7 else ("StatA.TLabel" if stab_val>=0.5 else "StatR.TLabel")
        self._wf_map["wf_stability"].config(text=f"{stab_val:.2f}", style=stab_style)
        self._wf_map["wf_is"].config(text=f"{wf['mean_is']:.0f}%")
        self._wf_map["wf_oos"].config(text=f"{wf['mean_oos']:.0f}%")
        self.wf_status.config(text=f"Done — WFE: {wfe_val:.2f}  Stability: {stab_val:.2f}")
        self.run_wf_btn.config(state="normal")

    def _run_regime(self):
        self.regime_status.config(text="Running regime stress test…"); self.update_idletasks()
        bp = self._get_base_params()
        res = run_regime_stress(bp, int(self.regime_nsim_v.get()))
        self._draw_regime(res)
        self.canvas_wf.draw_idle()
        self.regime_status.config(text="Regime stress test complete")

    def _draw_wf_folds(self, wf):
        ax = self.ax_wf_folds
        self._ax_base(ax, "IS vs OOS Pass Rate by Fold")
        n = wf["n_folds"]; x = np.arange(n)
        w = 0.35
        bars_is  = ax.bar(x-w/2, wf["is_rates"],  w, color=C["blue"],  alpha=0.8, label="In-Sample (IS)",  zorder=3)
        bars_oos = ax.bar(x+w/2, wf["oos_rates"], w, color=C["amber"], alpha=0.8, label="Out-of-Sample (OOS)", zorder=3)
        for b in list(bars_is)+list(bars_oos):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.0f}",ha="center",va="bottom",fontsize=6,color=C["text"])
        ax.axhline(40, color=C["win"], lw=0.8, ls=":", alpha=0.7); ax.text(n-0.5,41,"40% target",fontsize=6.5,color=C["win"])
        ax.set_xticks(x); ax.set_xticklabels([f"F{i+1}" for i in range(n)], fontsize=7.5)
        ax.set_ylabel("Pass rate (%)",fontsize=8); ax.set_ylim(0,100)
        ax.legend(fontsize=7,framealpha=0,loc="upper right")
        ax.yaxis.grid(True,color=C["border"],lw=0.4); ax.set_axisbelow(True)
        self._info_note(ax, "wf_folds")

    def _draw_wfe_gauge(self, wf):
        ax = self.ax_wf_wfe
        self._ax_base(ax, "Walk Forward Efficiency (WFE) Gauge")
        wfe=wf["wfe"]
        zones=[(0,0.5,"Poor\n(<0.50)",C["bust"]),(0.5,0.65,"Marginal\n(0.50–0.65)",C["amber"]),
               (0.65,0.8,"Good\n(0.65–0.80)",C["gold"]),(0.8,1.0,"Excellent\n(≥0.80)",C["win"])]
        for lo,hi,label,col in zones:
            ax.barh(0, hi-lo, left=lo, height=0.5, color=col, alpha=0.75, zorder=2)
            ax.text((lo+hi)/2,-0.42,label,ha="center",fontsize=6.5,color=C["text"],fontweight="bold",linespacing=1.3)
        needle = min(wfe,1.0)
        ax.axvline(needle, color=C["text"], lw=2.5, zorder=5)
        ax.text(needle, 0.36, f"▼ {wfe:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=C["text"])
        ax.set_xlim(0,1.0); ax.set_ylim(-0.72,0.8)
        ax.set_yticks([]); ax.set_xlabel("WFE = mean OOS pass rate ÷ mean IS pass rate", fontsize=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.text(0.5,0.97,f"Mean IS: {wf['mean_is']:.1f}%   |   Mean OOS: {wf['mean_oos']:.1f}%   |   Stability: {wf['stability']:.2f}",
                ha="center",va="top",transform=ax.transAxes,fontsize=8,color=C["muted"])
        self._info_note(ax, "wf_wfe")

    def _draw_regime(self, res):
        ax = self.ax_wf_regime
        self._ax_base(ax, "Regime Stress Test — Pass & Bust Rates by Market Condition")
        names=list(res.keys()); prs=[res[n]["pass_rate"] for n in names]; brs=[res[n]["bust_rate"] for n in names]
        # Use consistent pass/bust colours with hatch to distinguish regimes
        regime_hatches = ['', '//', 'xx', '..']
        x=np.arange(len(names)); w=0.38
        for i, (name, pr, br) in enumerate(zip(names, prs, brs)):
            hatch = regime_hatches[i % len(regime_hatches)]
            ax.bar(x[i]-w/2, pr, w, color=C["win"],  alpha=0.80, hatch=hatch, edgecolor=C["bg"],   zorder=3, label=f"{name} — pass" if i==0 else "_")
            ax.bar(x[i]+w/2, br, w, color=C["bust"], alpha=0.65, hatch=hatch, edgecolor=C["bg"],   zorder=3, label=f"{name} — bust" if i==0 else "_")
            ax.text(x[i]-w/2, pr+0.5, f"{pr:.0f}",ha="center",va="bottom",fontsize=7,color=C["text"])
            ax.text(x[i]+w/2, br+0.5, f"{br:.0f}",ha="center",va="bottom",fontsize=7,color=C["text"])
        ax.set_xticks(x); ax.set_xticklabels(names,fontsize=8,fontweight="bold"); ax.set_ylabel("Rate (%)",fontsize=8)
        ax.set_ylim(0,100); ax.axhline(30,color=C["muted"],lw=0.8,ls=":",alpha=0.6)
        ax.text(len(names)-0.5,31,"30% floor",fontsize=6.5,color=C["muted"])
        ax.legend(handles=[Patch(color=C["win"], alpha=0.8, label="Pass rate (left bar)"),
                            Patch(color=C["bust"],alpha=0.65,label="Bust rate (right bar)")],
                  fontsize=7, framealpha=0, loc="upper right")
        ax.yaxis.grid(True,color=C["border"],lw=0.4); ax.set_axisbelow(True)
        self._info_note(ax, "regime_stress")

    def _draw_wf_equity(self, wf):
        ax = self.ax_wf_equity
        self._ax_base(ax, "IS vs OOS Sample Equity Curves")
        for r in wf.get("is_sample", []):
            col = C["win"] if r.outcome=="pass" else C["bust"]
            ax.plot(range(len(r.equity)),r.equity,color=col,alpha=0.22,lw=0.8,zorder=2)
        for r in wf.get("oos_sample", []):
            col = C["amber"] if r.outcome=="pass" else "#CC8888"
            ax.plot(range(len(r.equity)),r.equity,color=col,alpha=0.22,lw=0.8,ls="--",zorder=2)
        ax.axhline(PROFIT_TARGET,color=C["win"],  lw=1.2,ls="--",alpha=0.8,label=f"+${PROFIT_TARGET:,.0f} target")
        ax.axhline(-DD_LIMIT,    color=C["bust"], lw=1.2,ls="--",alpha=0.8,label=f"-${DD_LIMIT:,.0f} bust")
        ax.axhline(0,            color=C["muted"],lw=0.5,alpha=0.4)
        ax.legend(handles=[
            Patch(color=C["win"],   alpha=0.7, label="IS pass  (solid green)"),
            Patch(color=C["bust"],  alpha=0.6, label="IS bust  (solid red)"),
            Patch(color=C["amber"], alpha=0.7, label="OOS pass (dashed amber)"),
            Patch(facecolor="#CC8888",alpha=0.6,label="OOS bust (dashed pink)"),
            Patch(color=C["win"],   alpha=0.0, label=f"── +${PROFIT_TARGET:,.0f} target"),
            Patch(color=C["bust"],  alpha=0.0, label=f"── -${DD_LIMIT:,.0f} bust"),
        ], fontsize=6.5, framealpha=0, loc="upper left", ncol=2)
        ax.set_xlabel("Day",fontsize=8); ax.set_ylabel("Cumulative P&L ($)",fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_:f"${x:,.0f}"))
        self._info_note(ax, "wf_equity")

    def _draw_wf_degradation(self, wf):
        ax = self.ax_wf_deg
        self._ax_base(ax, "IS→OOS Degradation by Fold")
        n=wf["n_folds"]; x=np.arange(n)
        gaps=[wf["is_rates"][i]-wf["oos_rates"][i] for i in range(n)]
        cols=[C["win"] if g<=15 else (C["amber"] if g<=30 else C["bust"]) for g in gaps]
        bars=ax.bar(x,gaps,color=cols,alpha=0.82,zorder=3,edgecolor="none")
        for b,g in zip(bars,gaps):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f"{g:.1f}pp",ha="center",va="bottom",fontsize=7,color=C["text"])
        ax.axhline(0,color=C["muted"],lw=0.8)
        ax.axhline(15,color=C["amber"],lw=0.8,ls=":",alpha=0.8)
        ax.text(n-0.5,16,"15pp warning",fontsize=6.5,color=C["amber"])
        ax.set_xticks(x); ax.set_xticklabels([f"F{i+1}" for i in range(n)],fontsize=7.5)
        ax.set_ylabel("IS − OOS gap (pp)",fontsize=8)
        ax.legend(handles=[Patch(color=C["win"],  alpha=0.82,label="Good  ≤ 15pp"),
                            Patch(color=C["amber"],alpha=0.82,label="Watch 15–30pp"),
                            Patch(color=C["bust"], alpha=0.82,label="High  > 30pp")],
                  fontsize=6.5, framealpha=0, loc="upper right")
        ax.yaxis.grid(True,color=C["border"],lw=0.4); ax.set_axisbelow(True)
        self._info_note(ax, "wf_degradation")

    def _draw_wf_stability(self, wf):
        ax = self.ax_wf_stab
        self._ax_base(ax, "OOS Pass Rate — Fold-by-Fold Consistency")
        n=wf["n_folds"]; x=list(range(1,n+1))
        oos=wf["oos_rates"]; is_=wf["is_rates"]
        oos_std = float(np.std(oos))
        ax.fill_between(x,[max(0,m-oos_std) for m in oos],[min(100,m+oos_std) for m in oos],
                        alpha=0.18,color=C["amber"],label="±1 std dev band")
        ax.plot(x,oos, marker="o",markersize=5,lw=1.5,color=C["amber"],label=f"OOS  (mean {wf['mean_oos']:.0f}%)",zorder=4)
        ax.plot(x,is_, marker="s",markersize=4,lw=1.0,color=C["blue"],ls="--",label=f"IS   (mean {wf['mean_is']:.0f}%)",zorder=3,alpha=0.8)
        ax.axhline(wf["mean_oos"],color=C["amber"],lw=1,ls=":",alpha=0.6)
        ax.axhline(wf["mean_is"], color=C["blue"], lw=1,ls=":",alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels([f"F{i}" for i in x],fontsize=7.5)
        ax.set_ylabel("Pass rate (%)",fontsize=8); ax.set_ylim(0,100)
        ax.legend(fontsize=7,framealpha=0,loc="upper right")
        ax.yaxis.grid(True,color=C["border"],lw=0.4); ax.set_axisbelow(True)
        ax.text(0.98,0.04,f"OOS std: {oos_std:.1f}pp  ({'stable' if oos_std<10 else 'variable'})",
                transform=ax.transAxes,fontsize=7.5,ha="right",color=C["muted"])
        self._info_note(ax, "wf_stability")

    # ══════════════════════════════════════════════════════════════
    # SENSITIVITY TAB — RUN + DRAW
    # ══════════════════════════════════════════════════════════════
    def _run_tornado(self):
        self.run_tornado_btn.config(state="disabled"); self.sens_status.config(text="Running tornado…"); self.update_idletasks()
        bp = self._get_base_params()
        n  = int(self.tornado_nsim_v.get())
        rows, base_rate = run_tornado(bp, n)
        self._draw_tornado(rows, base_rate)
        self._draw_param_profile_data(rows, base_rate)
        self.canvas_sens.draw_idle()
        # Robustness score: use normalized inverse of mean relative sensitivity
        impacts = [abs(hi-lo) for _,lo,hi,_ in rows]
        mean_impact = float(np.mean(impacts)) if impacts else 0
        # Robustness is higher when mean sensitivity impact is lower
        robustness = max(0.0, min(100.0, 100.0 - mean_impact * 1.5))
        quality = "Robust" if robustness>70 else ("Moderate" if robustness>45 else "Fragile")
        col = C["win"] if robustness>70 else (C["amber"] if robustness>45 else C["bust"])
        self.robustness_lbl.config(text=f"Robustness Score: {robustness:.0f}/100\n{quality} — mean sensitivity: {mean_impact:.0f}pp",
                                   foreground=col)
        self.sens_status.config(text=f"Tornado done — baseline: {base_rate:.0f}%")
        self.run_tornado_btn.config(state="normal")

    def _run_sens_curve(self):
        self.run_curve_btn.config(state="disabled"); self.sens_status.config(text="Running sensitivity curve…"); self.update_idletasks()
        bp     = self._get_base_params()
        pname  = self.sens_param_v.get()
        n_pts  = int(self.sens_pts_v.get())
        n_sims = int(self.sens_nsim_v.get())
        vals, rates = sweep_one_param(bp, pname, n_pts, n_sims)
        self._draw_sens_curve(vals, rates, pname, bp)
        self.canvas_sens.draw_idle()
        self.sens_status.config(text=f"Curve done — {pname}")
        self.run_curve_btn.config(state="normal")

    def _run_heatmap(self):
        px = self.hm_x_v.get(); py = self.hm_y_v.get()
        if px == py:
            self.sens_status.config(text="X and Y parameters must differ"); return
        self.run_hmap_btn.config(state="disabled"); self.sens_status.config(text="Running heatmap…"); self.update_idletasks()
        bp     = self._get_base_params()
        n_grid = int(self.hm_grid_v.get())
        n_sims = int(self.hm_nsim_v.get())
        hm     = run_heatmap(bp, px, py, n_grid, n_sims)
        self._draw_heatmap(hm, bp)
        self.canvas_sens.draw_idle()
        self.sens_status.config(text=f"Heatmap done — {px} × {py}")
        self.run_hmap_btn.config(state="normal")

    def _draw_tornado(self, rows, base_rate):
        ax = self.ax_tornado
        self._ax_base(ax, "Tornado Chart — Parameter Impact on Pass Rate")
        n = len(rows)
        labels = [r[0] for r in rows]
        for i, (name, lo_rate, hi_rate, _base) in enumerate(rows):
            left  = min(lo_rate, hi_rate) - base_rate
            right = max(lo_rate, hi_rate) - base_rate
            if left < 0:
                ax.barh(i, left,  left=0, height=0.6, color=C["bust"], alpha=0.75, zorder=3)
            if right > 0:
                ax.barh(i, right, left=0, height=0.6, color=C["win"],  alpha=0.75, zorder=3)
            ax.text(-1.5, i, f"{lo_rate:.0f}%", ha="right", va="center", fontsize=7.5, color=C["bust"])
            ax.text( 1.5, i, f"{hi_rate:.0f}%", ha="left",  va="center", fontsize=7.5, color=C["win"])
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8.5)
        ax.axvline(0, color=C["text"], lw=1.0, zorder=4)
        ax.set_xlabel(f"Pass rate change from baseline ({base_rate:.0f}%)", fontsize=8)
        ax.text(0.02, 0.02, "← lower param value is WORSE", transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=C["bust"])
        ax.text(0.98, 0.02, "higher param value is BETTER →", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=C["win"])
        ax.xaxis.grid(True, color=C["border"], lw=0.4, zorder=0); ax.set_axisbelow(True)
        ax.legend(handles=[Patch(color=C["win"], alpha=0.75, label="HIGH value of parameter"),
                            Patch(color=C["bust"],alpha=0.75, label="LOW value of parameter"),],
                  fontsize=7.5, framealpha=0, loc="lower right")
        self._info_note(ax, "tornado")

    def _draw_param_profile(self, bp, base_rate):
        """Sensitivity summary bar chart — replaces broken polar subplot."""
        ax = self.ax_pf
        self._ax_base(ax, "Sensitivity Impact Summary")
        ax.text(0.5, 0.5, "Run Tornado to populate\nthis chart", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=C["muted"])
        self._info_note(ax, "sens_summary")

    def _draw_param_profile_data(self, rows, base_rate):
        """Called after tornado runs — draws ranked impact bars."""
        ax = self.ax_pf
        self._ax_base(ax, "Sensitivity Impact Summary")
        if not rows:
            return
        impacts = [abs(hi-lo) for _,lo,hi,_ in rows]
        max_imp = max(impacts) if impacts else 1
        labels  = [r[0] for r in rows]
        cols    = []
        for imp in impacts:
            ratio = imp / max_imp if max_imp > 0 else 0
            cols.append(C["bust"] if ratio > 0.70 else (C["amber"] if ratio > 0.35 else C["win"]))
        y = np.arange(len(rows))
        ax.barh(y, impacts, height=0.6, color=cols, alpha=0.82, zorder=3)
        for yi, imp in enumerate(impacts):
            ax.text(imp+0.3, yi, f"{imp:.0f}pp", va="center", fontsize=7, color=C["text"])
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Pass rate swing: HIGH minus LOW value (pp)", fontsize=7.5)
        ax.xaxis.grid(True, color=C["border"], lw=0.4); ax.set_axisbelow(True)
        ax.legend(handles=[Patch(color=C["bust"], alpha=0.82, label="High impact — critical"),
                            Patch(color=C["amber"],alpha=0.82, label="Medium impact"),
                            Patch(color=C["win"],  alpha=0.82, label="Low impact — robust")],
                  fontsize=6.5, framealpha=0, loc="lower right")
        self._info_note(ax, "sens_summary")

    def _draw_sens_curve(self, vals, rates, pname, bp):
        ax = self.ax_curve
        self._ax_base(ax, f"Sensitivity Curve — {pname}")
        ax.plot(vals, rates, color=C["blue"], lw=2.0, marker="o", markersize=3, zorder=4, label="Pass rate")
        ax.fill_between(vals, rates, alpha=0.12, color=C["blue"])
        cfg = SENS_PARAMS[pname]
        base_val = float(bp[cfg["attr"]])
        ax.axvline(base_val, color=C["amber"], lw=1.5, ls="--", alpha=0.9, label=f"Your baseline: {base_val:{cfg['fmt']}}")
        ax.axhline(40, color=C["muted"], lw=0.8, ls=":", alpha=0.7)
        ax.text(vals[0], 41, "40% benchmark", fontsize=6.5, color=C["muted"])
        best_idx = int(np.argmax(rates))
        ax.scatter([vals[best_idx]], [rates[best_idx]], s=60, color=C["win"], zorder=5,
                   label=f"Peak: {rates[best_idx]:.0f}% @ {vals[best_idx]:{cfg['fmt']}}")
        ax.set_xlabel(pname, fontsize=8); ax.set_ylabel("Pass rate (%)", fontsize=8)
        ax.set_ylim(0, max(max(rates)*1.15+5, 15))
        ax.legend(fontsize=7.5, framealpha=0)
        ax.yaxis.grid(True, color=C["border"], lw=0.4); ax.set_axisbelow(True)
        self._info_note(ax, "sens_curve")

    def _draw_heatmap(self, hm, bp):
        ax = self.ax_hmap
        # Remove any existing colorbars attached to this axes before redrawing
        if hasattr(self, "_hmap_colorbar") and self._hmap_colorbar is not None:
            try:
                self._hmap_colorbar.remove()
            except Exception:
                pass
            self._hmap_colorbar = None
        self._ax_base(ax, f"Heatmap — {hm['px']}  ×  {hm['py']}")
        grid  = hm["grid"]; xv = hm["xv"]; yv = hm["yv"]
        cmap  = mcolors.LinearSegmentedColormap.from_list("rg", [C["bust"],"#CC5555","#DDAA44","#88BB44",C["win"]])
        im    = ax.imshow(grid, origin="lower", cmap=cmap, aspect="auto",
                          extent=[xv[0],xv[-1],yv[0],yv[-1]], vmin=0, vmax=100)
        cb    = ax.figure.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
        self._hmap_colorbar = cb
        cb.set_label("Pass rate (%)", fontsize=7.5); cb.ax.tick_params(labelsize=7)
        # Annotate colorbar zones
        for val, label in [(15,"red zone"),(40,"target"),(70,"strong")]:
            cb.ax.axhline(val, color="white", lw=0.8, alpha=0.6)
        cfgx = SENS_PARAMS[hm["px"]]; cfgy = SENS_PARAMS[hm["py"]]
        bx = float(bp[cfgx["attr"]]); by = float(bp[cfgy["attr"]])
        ax.scatter([bx],[by],marker="*",s=240,color="white",zorder=5,edgecolors=C["text"],linewidths=0.8)
        ax.set_xlabel(hm["px"], fontsize=8); ax.set_ylabel(hm["py"], fontsize=8)
        ax.text(0.97,0.03,"★ = your current settings",transform=ax.transAxes,
                ha="right",va="bottom",fontsize=7,color="white",fontweight="bold")
        self._info_note(ax, "heatmap")
        self._draw_robust_summary(hm, bp)

    def _draw_robust_summary(self, hm, bp):
        ax = self.ax_robust
        self._ax_base(ax, "Heatmap — Pass Rate Distribution")
        flat = hm["grid"].flatten()
        mean_v = float(np.mean(flat))
        ax.hist(flat, bins=15, color=C["blue"], alpha=0.72, edgecolor="none", zorder=3)
        ax.axvline(mean_v, color=C["amber"], lw=1.5, ls="--", label=f"Mean: {mean_v:.0f}%")
        ax.axvline(40, color=C["win"], lw=1.0, ls=":", alpha=0.8)
        ax.text(42, 0, "40% target", fontsize=6.5, color=C["win"], va="bottom")
        pct_above = float(np.mean(flat>=40)*100)
        ax.text(0.98,0.95,f"{pct_above:.0f}% of cells ≥ 40%",
                transform=ax.transAxes,ha="right",va="top",fontsize=8,color=C["text"],fontweight="bold")
        ax.set_xlabel("Pass rate (%)",fontsize=8); ax.set_ylabel("Cell count",fontsize=8)
        ax.legend(fontsize=7.5,framealpha=0)
        ax.yaxis.grid(True,color=C["border"],lw=0.4); ax.set_axisbelow(True)
        self._info_note(ax, "sens_dist")


    # ══════════════════════════════════════════════════════════════
    # TAB 4 — ALGORITHM CONFIG  (pure-maths analysis engine)
    # ══════════════════════════════════════════════════════════════
    def _build_alg_tab(self):
        tab = self.tab_alg
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        # ── Left control panel ────────────────────────────────────
        ctrl, _ = self._make_scroll_ctrl(tab, width=314)

        self._sec(ctrl, "Analysis Source")
        src_f = ttk.Frame(ctrl, style="F.TFrame"); src_f.pack(fill="x", padx=10, pady=(0,4))
        ttk.Label(src_f, text="Uses MC tab parameters live.", style="Sub.TLabel").pack(anchor="w")

        self._sec(ctrl, "Target Configuration")
        self.alg_target_pass_v = tk.DoubleVar(value=50.0)
        self._sl(ctrl, "Target pass rate (%)", self.alg_target_pass_v, 20, 90, fmt=".0f")

        self.alg_max_bust_v = tk.DoubleVar(value=20.0)
        self._sl(ctrl, "Max bust rate (%)", self.alg_max_bust_v, 5, 50, fmt=".0f")

        self._sec(ctrl, "Kelly Sizing Parameters")
        self.alg_kelly_frac_v = tk.DoubleVar(value=0.25)
        self._sl(ctrl, "Kelly fraction", self.alg_kelly_frac_v, 0.05, 1.0, fmt=".2f", res=0.05)
        ttk.Label(ctrl, text="0.25 = Quarter Kelly  ·  0.5 = Half Kelly\n1.0 = Full Kelly (never recommended)",
                  style="Sub.TLabel", wraplength=285).pack(anchor="w", padx=14, pady=(0,6))

        self._sec(ctrl, "Monte Carlo Approximation")
        self.alg_nsim_v = tk.IntVar(value=500)
        sr = ttk.Frame(ctrl, style="F.TFrame"); sr.pack(fill="x", padx=10, pady=(0,6))
        ttk.Label(sr, text="Sims for validation:", style="Sub.TLabel").pack(side="left")
        ttk.Spinbox(sr, from_=200, to=2000, increment=200, textvariable=self.alg_nsim_v,
                    width=7, font=("Helvetica",10)).pack(side="left", padx=(8,0))

        self.run_alg_btn = ttk.Button(ctrl, text="▶  Generate Algorithm Config",
                                       style="Run.TButton", command=self._run_alg_analysis)
        self.run_alg_btn.pack(fill="x", padx=10, pady=(4,4))
        self.alg_status = ttk.Label(ctrl, text="Ready — configure above and click Generate",
                                     style="Sub.TLabel", wraplength=285)
        self.alg_status.pack(padx=10, pady=(0,8))

        self._sec(ctrl, "Mathematical Metrics")
        self.alg_metrics_lbl = ttk.Label(ctrl, text="Run analysis to populate.",
                                          style="Sub.TLabel", wraplength=285, justify="left")
        self.alg_metrics_lbl.pack(fill="x", padx=14, pady=(0,4))

        self._sec(ctrl, "Algorithm Rating")
        self.alg_rating_lbl = ttk.Label(ctrl, text="—",
                                         font=("Helvetica",22,"bold"),
                                         background=C["bg"], foreground=C["muted"])
        self.alg_rating_lbl.pack(padx=14, pady=(2,2))
        self.alg_rating_desc = ttk.Label(ctrl, text="",
                                          style="Sub.TLabel", wraplength=285, justify="left")
        self.alg_rating_desc.pack(fill="x", padx=14, pady=(0,8))

        self._sec(ctrl, "WFE Improvement Actions")
        self.alg_wfe_lbl = ttk.Label(ctrl, text="Run analysis to populate.",
                                      style="Sub.TLabel", wraplength=285, justify="left")
        self.alg_wfe_lbl.pack(fill="x", padx=14, pady=(0,4))

        self._sec(ctrl, "Stability Score Actions")
        self.alg_stab_lbl = ttk.Label(ctrl, text="Run analysis to populate.",
                                       style="Sub.TLabel", wraplength=285, justify="left")
        self.alg_stab_lbl.pack(fill="x", padx=14, pady=(0,4))

        self._sec(ctrl, "AI Prompt Generator  —  Topstep / ProjectX")
        ttk.Label(ctrl, text="Generates a complete Python prompt using the\nproject-x-py SDK for Topstep/TopstepX.",
                  style="Sub.TLabel", wraplength=285, justify="left").pack(fill="x", padx=14, pady=(0,4))
        ttk.Label(ctrl, text="pip install project-x-py",
                  font=("Courier", 9), background=C["panel"], foreground=C["blue"],
                  relief="flat").pack(anchor="w", padx=14, pady=(0,6))

        # Timeframe selector (framework is always Python/project-x-py)
        tf_row = ttk.Frame(ctrl, style="F.TFrame"); tf_row.pack(fill="x", padx=10, pady=(0,4))
        ttk.Label(tf_row, text="Timeframe:", style="Sub.TLabel", width=12).pack(side="left")
        self.alg_timeframe_v = tk.StringVar(value="5-minute")
        ttk.Combobox(tf_row, textvariable=self.alg_timeframe_v, state="readonly", width=18,
                     values=["1-minute","2-minute","5-minute","15-minute","30-minute",
                              "1-hour","4-hour","Daily"]).pack(side="left")
        # Fixed framework — not shown as selector
        self.alg_framework_v = tk.StringVar(value="Python / project-x-py (Topstep)")

        # Extra context field
        self._sec(ctrl, "Extra Signal Context  (optional)")
        ttk.Label(ctrl, text="Describe your entry/exit logic or indicators.\nLeave blank for AI to infer from preset type.",
                  style="Sub.TLabel", wraplength=285, justify="left").pack(fill="x", padx=14, pady=(0,4))
        self.alg_context_txt = tk.Text(ctrl, height=4, wrap="word", font=("Helvetica",9),
                                        bg="white", fg=C["text"], relief="flat",
                                        padx=6, pady=4, highlightthickness=1,
                                        highlightbackground=C["border"])
        self.alg_context_txt.pack(fill="x", padx=10, pady=(0,4))
        self.alg_context_txt.insert("1.0", "e.g. Fast EMA(9) crosses above Slow EMA(21) on 5-min chart with volume confirmation")

        # What the AI still needs checklist
        self._sec(ctrl, "What the AI still needs from you")
        self.alg_needs_lbl = ttk.Label(ctrl,
            text="Run analysis first to see the checklist.",
            style="Sub.TLabel", wraplength=285, justify="left")
        self.alg_needs_lbl.pack(fill="x", padx=14, pady=(0,6))

        # Buttons
        ttk.Button(ctrl, text="📋  Copy stats report",
                   command=self._copy_alg_config).pack(fill="x", padx=10, pady=(0,4))
        ttk.Button(ctrl, text="🤖  Copy AI algorithm prompt",
                   style="Run.TButton",
                   command=self._copy_ai_prompt).pack(fill="x", padx=10, pady=(0,6))
        self._last_config_text = ""
        self._last_ai_prompt   = ""

        # ── Right: figures (top) + report text (bottom) ──────────
        right = ttk.Frame(tab, style="F.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(8,4))
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        # Chart panel
        cf_alg = ttk.Frame(right, style="F.TFrame")
        cf_alg.grid(row=0, column=0, sticky="nsew")
        self.fig_alg = Figure(facecolor=C["bg"])
        self.fig_alg.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.09,
                                      hspace=0.50, wspace=0.35)
        gs4 = self.fig_alg.add_gridspec(2, 3)
        self.ax_alg_kelly    = self.fig_alg.add_subplot(gs4[0, 0])
        self.ax_alg_beven    = self.fig_alg.add_subplot(gs4[0, 1])
        self.ax_alg_qual     = self.fig_alg.add_subplot(gs4[0, 2])
        self.ax_alg_ror      = self.fig_alg.add_subplot(gs4[1, 0])
        self.ax_alg_wfe_map  = self.fig_alg.add_subplot(gs4[1, 1])
        self.ax_alg_frontier = self.fig_alg.add_subplot(gs4[1, 2])
        self._style_all_axes(self.fig_alg)
        self.canvas_alg = FigureCanvasTkAgg(self.fig_alg, master=cf_alg)
        self.canvas_alg.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_alg.mpl_connect("pick_event", self._on_chart_pick)

        # Report text panel
        rp = ttk.Frame(right, style="F.TFrame")
        rp.grid(row=1, column=0, sticky="nsew", pady=(6,0))
        rp.rowconfigure(0, weight=1); rp.columnconfigure(0, weight=1)
        self.alg_report_txt = tk.Text(rp, wrap="word", bg=C["panel"], fg=C["text"],
                                       relief="flat", font=("Courier",9), padx=12, pady=8,
                                       state="disabled")
        rsb = ttk.Scrollbar(rp, orient="vertical", command=self.alg_report_txt.yview)
        self.alg_report_txt.configure(yscrollcommand=rsb.set)
        self.alg_report_txt.grid(row=0, column=0, sticky="nsew")
        rsb.grid(row=0, column=1, sticky="ns")

    # ── Algorithm analysis engine ─────────────────────────────────
    def _get_alg_params(self):
        """Gather current MC params plus alg-tab targets."""
        bp = self._get_base_params()
        bp["target_pass"] = self.alg_target_pass_v.get()
        bp["max_bust"]    = self.alg_max_bust_v.get()
        bp["kelly_frac"]  = self.alg_kelly_frac_v.get()
        return bp

    def _run_alg_analysis(self):
        self.run_alg_btn.config(state="disabled")
        self.alg_status.config(text="Computing…"); self.update_idletasks()
        try:
            bp = self._get_alg_params()
            result = _compute_alg_config(bp, int(self.alg_nsim_v.get()))
            self._draw_alg_charts(result, bp)
            self.canvas_alg.draw_idle()
            self._populate_alg_panel(result, bp)
        except Exception as e:
            self.alg_status.config(text=f"Error: {e}")
        finally:
            self.run_alg_btn.config(state="normal")

    def _populate_alg_panel(self, r, bp):
        # Rating
        score = r["score"]
        if score >= 80:   rating, rcol = "A",  C["win"]
        elif score >= 65: rating, rcol = "B",  C["win"]
        elif score >= 50: rating, rcol = "C",  C["amber"]
        elif score >= 35: rating, rcol = "D",  C["amber"]
        else:             rating, rcol = "F",  C["bust"]
        self.alg_rating_lbl.config(text=f"{rating}  ({score:.0f}/100)", foreground=rcol)
        self.alg_rating_desc.config(text=r["rating_desc"])

        # Metrics
        m = r["metrics"]
        self.alg_metrics_lbl.config(text=(
            f"EV/trade: ${m['ev_trade']:+.2f}   EV/day: ${m['ev_day']:+.1f}\n"
            f"σ/day: ${m['std_day']:.1f}   Sharpe proxy: {m['sharpe']:.3f}\n"
            f"Kelly f: {m['kelly']:.4f}   Rec. contracts: {m['rec_contracts']}\n"
            f"Days to target (EV): {m['days_to_target']:.1f}\n"
            f"P(qual day): {m['p_qual_day']*100:.1f}%   Risk of ruin: {m['ror']*100:.2f}%"
        ))

        # WFE actions
        self.alg_wfe_lbl.config(text="\n".join(r["wfe_actions"]))
        # Stability actions
        self.alg_stab_lbl.config(text="\n".join(r["stab_actions"]))
        self.alg_status.config(text=f"Done — Score: {score:.0f}/100  ({rating})")

        # Report text
        self._last_config_text = r["report"]
        self.alg_report_txt.config(state="normal")
        self.alg_report_txt.delete("1.0", "end")
        self.alg_report_txt.insert("1.0", r["report"])
        self.alg_report_txt.config(state="disabled")

        # Build AI prompt and checklist
        self._last_ai_prompt = self._generate_ai_prompt(r, bp)
        self._update_needs_checklist(r, bp)

    def _update_needs_checklist(self, r, bp):
        """Populate the 'What the AI still needs' checklist."""
        m = r["metrics"]
        needs = []
        # Always included
        needs.append("✅ Statistical targets  — INCLUDED in prompt")
        needs.append("✅ Risk management rules — INCLUDED in prompt")
        needs.append("✅ Challenge constraints — INCLUDED in prompt")
        needs.append("✅ Position sizing formula — INCLUDED in prompt")
        needs.append("✅ Kelly recommendation  — INCLUDED in prompt")
        needs.append("✅ Gambler's ruin pass probability — INCLUDED")
        needs.append("✅ EV formula (no instrument factor) — INCLUDED")
        # Calibration confidence
        calib = self.calib_v.get().split("(")[0].strip() if hasattr(self, "calib_v") else "Unknown"
        if "None" in calib:
            needs.append("⚠️  Calibration = None — pass rates are 90-100%\n   Switch to 'Typical' for realistic estimates")
        else:
            needs.append(f"✅ Calibration confidence: {calib}")

        # Signal logic assessment
        context = self.alg_context_txt.get("1.0", "end").strip()
        default_ctx = "e.g. Fast EMA(9) crosses above Slow EMA(21) on 5-min chart with volume confirmation"
        if context and context != default_ctx:
            needs.append("✅ Entry/exit signal logic — PROVIDED by you")
        else:
            needs.append("⚠️  Entry/exit signal logic — NOT provided\n   Fill in 'Extra Signal Context' for precise code")

        # Data
        needs.append("⚠️  Historical OHLCV data — you must supply\n   (or AI will generate synthetic test data)")

        # Framework note
        fw = self.alg_framework_v.get()
        if "Pine Script" in fw:
            needs.append("ℹ️  TradingView account needed to run Pine Script")
        elif "NinjaScript" in fw:
            needs.append("ℹ️  NinjaTrader licence needed to compile NinjaScript")
        elif "MQL5" in fw:
            needs.append("ℹ️  MetaTrader 5 needed to run MQL5 code")

        # Edge warnings
        if m["days_to_target"] > 30:
            needs.append("⚠️  EV/day is below challenge requirement — AI code\n   will reflect these params but may still timeout")
        if m["p_qual_day"] < 0.4:
            needs.append("⚠️  Low qualifying day probability — AI strategy may\n   need regime filter added to concentrate activity")

        self.alg_needs_lbl.config(text="\n".join(needs))

    def _generate_ai_prompt(self, r, bp) -> str:
        """Build a complete, Topstep/project-x-py ready AI prompt with anti-overfitting rules."""
        m           = r["metrics"]
        tf          = self.alg_timeframe_v.get()
        context     = self.alg_context_txt.get("1.0", "end").strip()
        default_ctx = "e.g. Fast EMA(9) crosses above Slow EMA(21) on 5-min chart with volume confirmation"
        has_context = context and context != default_ctx

        # Detect preset
        preset_name = "Custom"
        for k, v in STRATEGY_PRESETS.items():
            if abs(v["wr"]-bp["wr"])<2 and abs(v["aw"]-bp["aw"])<5 and abs(v["al"]-bp["al"])<5:
                preset_name = k; break

        strat_lib   = STRATEGY_LIBRARY.get(preset_name, {})
        strat_type  = strat_lib.get("type", "trend-following / momentum")
        strat_desc  = strat_lib.get("what_it_entails", "")
        best_regime = strat_lib.get("best_regime", "")
        notes       = strat_lib.get("notes", "")
        instr       = bp["instrument"]
        instr_spec  = CONTRACT_SPECS.get(instr, {})
        tick_val    = instr_spec.get("tick_value", 1.25)
        pt_val      = instr_spec.get("point_value", 5.0)
        rr          = bp["aw"] / bp["al"] if bp["al"] > 0 else 1.0

        # Sizing pseudocode
        k_exp = bp["k"]
        sizing_code = {
            "Fixed size":        "size_mult = 1.0",
            "Linear scale-down": "size_mult = max(0.05, buffer_pct)",
            "Half Kelly":        "size_mult = max(0.05, 0.5 + 0.5 * buffer_pct)",
            "Risk geometry":     f"size_mult = max(0.05, buffer_pct ** (1 / {k_exp:.1f}))",
            "Martingale":        "size_mult = min(3.0, max(0.05, 2.0 - buffer_pct))  # STRESS TEST ONLY",
        }.get(bp["sizing"], "size_mult = 1.0")

        # Anti-overfitting parameter guidance per strategy type
        if "mean-reversion" in strat_type or "scalp" in strat_type:
            signal_guidance = f"""SIGNAL ARCHITECTURE REQUIREMENTS (anti-overfitting):
  Use 2-3 simple, interpretable indicators maximum.
  Recommended for mean-reversion scalping:
    - Lookback period: 10–20 bars  (do NOT optimise to a single value — use round numbers)
    - RSI: period 9–14, overbought 65–70, oversold 30–35  (wide bands, not curve-fit)
    - Entry trigger: price deviation from VWAP or short-term SMA, ≥ 0.5× ATR
    - Exit trigger: return to mean OR time-based (max 10 bars held)
  CRITICAL: Do NOT use >3 filter conditions. Each additional filter risks reducing
  live trade frequency to near zero on OOS data (overfitting).
  Minimum trade frequency guard: algorithm MUST generate ≥ {bp['td']//2} trades/session.
  If 3 consecutive sessions produce zero trades, print a warning and relax
  entry thresholds by 20%."""
        elif "trend" in strat_type or "momentum" in strat_type or "EMA" in strat_type.upper() or "crossover" in strat_type:
            signal_guidance = f"""SIGNAL ARCHITECTURE REQUIREMENTS (anti-overfitting):
  Use 2-3 simple, interpretable indicators maximum.
  Recommended for EMA trend-following:
    - Fast EMA: 8–12 period  (use 9 as default — round number, don't optimise)
    - Slow EMA: 18–26 period  (use 21 as default)
    - Trend filter: 50-period SMA slope (positive = long-only, negative = short-only)
    - Volume filter: volume > 20-bar average (1 condition only — not multiple)
    - Entry: fast EMA crosses slow EMA in direction of trend filter
    - Stop: 1.5× ATR(14) from entry price
  CRITICAL: Do NOT add >1 additional filter (e.g. no RSI AND MACD AND Stoch together).
  Minimum trade frequency guard: algorithm MUST generate ≥ {bp['td']//2} trades/session.
  If 3 consecutive sessions produce zero trades, relax entry by widening EMA gap threshold."""
        elif "breakout" in strat_type:
            signal_guidance = f"""SIGNAL ARCHITECTURE REQUIREMENTS (anti-overfitting):
  Use 2 simple conditions maximum.
  Recommended for momentum breakout:
    - Range: high/low of first {15 if '5' in tf else 20} bars of session (opening range)
    - Entry: close above range high (long) or below range low (short)
    - Volume confirmation: current bar volume > 1.5× 10-bar average (single filter)
    - Stop: opposite side of opening range
  CRITICAL: No additional filters beyond volume. Breakout systems with many filters
  stop trading entirely on OOS data. Keep it simple.
  Minimum trade frequency guard: ≥ {bp['td']//2} trades/session required."""
        else:
            signal_guidance = f"""SIGNAL ARCHITECTURE REQUIREMENTS (anti-overfitting):
  Use 2-3 simple, interpretable indicators maximum with ROUND NUMBER parameters.
  Examples: SMA(20), RSI(14), ATR(14)  — not SMA(17), RSI(11), ATR(13).
  Minimum trade frequency guard: algorithm MUST generate ≥ {bp['td']//2} trades/session.
  If 3 consecutive sessions produce zero trades, relax entry conditions by 20%."""

        prompt = f"""You are an expert algorithmic trading developer specialising in Python and the Topstep/ProjectX API.

Write a complete, LIVE-READY trading algorithm using the project-x-py SDK (pip install project-x-py).
The algorithm must connect to TopstepX via the ProjectX API and trade {instr} futures.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOPSTEP / PROJECTX API — REQUIRED PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Authentication (use environment variables — NEVER hardcode credentials):
  import os
  # Set before running: export PROJECT_X_API_KEY="your_key"
  #                     export PROJECT_X_USERNAME="your_username"

Initialisation pattern (ALWAYS use TradingSuite):
  from project_x_py import TradingSuite
  suite = await TradingSuite.create("{instr}")
  # suite.client  — authenticated ProjectX client
  # suite.orders  — OrderManager for placing/cancelling orders
  # suite.positions — PositionManager for live P&L and position tracking
  # suite.data    — RealtimeDataManager for live OHLCV bars
  # suite.events  — EventBus for order fill / position callbacks

Market data (returns Polars DataFrame):
  bars = await suite.data.get_data("{tf.replace("-minute","min").replace("-hour","h")}")
  # Columns: timestamp, open, high, low, close, volume
  # Fallback if real-time not ready:
  bars = await suite.client.get_bars("{instr}", days=1, interval={tf.split("-")[0]})

Order placement (bracket order — entry + stop + target in one call):
  order = await suite.orders.place_bracket_order(
      contract_id=suite.instrument.id,
      side=0,          # 0=Buy, 1=Sell
      size=contracts,  # integer, max {MAX_CONTRACTS}
      entry_price=entry_price,
      stop_price=stop_price,
      target_price=target_price,
      order_type=2,    # 2=Market, 1=Limit
      custom_tag="strategy-entry",
  )

Flat/close all positions:
  await suite.positions.close_all_positions()

Real-time order fill callback:
  from project_x_py import EventType
  async def on_fill(event):
      print(f"Filled: {{event.data}}")
  await suite.on(EventType.ORDER_FILL, on_fill)

API endpoints (TopstepX):
  Base URL   : https://api.topstepx.com/api
  User hub   : https://rtc.topstepx.com/hubs/user
  Market hub : https://rtc.topstepx.com/hubs/market
  Rate limit : 400 concurrent requests

IMPORTANT TOPSTEP COMPLIANCE RULES:
  - Run ONLY on your own physical device (no VPS, no VPN, no cloud servers)
  - All API orders are FINAL — no reversal by Topstep support
  - All trading must comply with Topstep Terms of Use
  - Topstep does NOT provide technical support for custom code

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY SPECIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Preset     : {preset_name}
  Type       : {strat_type}
  Description: {strat_desc}
  Instrument : {instr}  (${pt_val}/pt, tick=${tick_val})
  Timeframe  : {tf} bars
  Best regime: {best_regime}
  Notes      : {notes}

{"USER-PROVIDED SIGNAL LOGIC:" if has_context else "SIGNAL LOGIC (infer from strategy type — see requirements below):"}
  {context if has_context else "[Implement signals as described in requirements below]"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{signal_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CALIBRATED STATISTICAL TARGETS  (from Monte Carlo simulation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These are the LIVE TRADING statistics your algorithm must achieve.
Avg win/loss are true dollar amounts per contract — already scaled
for {instr} ({instr_spec.get("name","micro futures")}, ${pt_val}/pt).

  Target win rate        : {bp["wr"]:.0f}%   (acceptable live range: {max(0,bp["wr"]-8):.0f}–{min(95,bp["wr"]+8):.0f}%)
  Target avg win / ctr   : ${bp["aw"]:.2f}   (range: ${bp["aw"]*0.80:.2f}–${bp["aw"]*1.25:.2f})
  Target avg loss / ctr  : ${bp["al"]:.2f}   (range: ${bp["al"]*0.80:.2f}–${bp["al"]*1.25:.2f})
  Target R/R             : {rr:.2f}   (minimum acceptable: {max(0.5,rr-0.3):.2f})
  Target trades / day    : {bp["td"]}         (range: {max(1,bp["td"]-2)}–{bp["td"]+3}, NEVER below {max(1,bp["td"]//2)})
  Net EV / trade         : ${m["ev_trade"]:+.4f}  (must remain POSITIVE — formula: c*(wr*aw − (1−wr)*al) − friction)
  Net EV / day           : ${m["ev_day"]:+.2f}
  Daily P&L σ            : ≈${m["std_day"]:.0f}  (daily Sharpe ≈ {m["sharpe"]:.3f})
  Days to $3k (EV)       : {m["days_to_target"]:.1f}  (challenge limit: 30 days)
  Gambler's ruin P(bust) : {m["ror"]*100:.3f}%  (two-sided barrier formula, exact)
  P(pass, no timeout)    : {(1-m["ror"])*100:.1f}%  (requires calibration confidence adjustment in live use)

CALIBRATION NOTE:
  The pass rates above assume parameters are KNOWN EXACTLY.
  In live trading with typical estimation uncertainty (±5pp win rate,
  ±20% payoffs), expect actual pass rate to be approximately:
    {max(0,(1-m["ror"])*100 - 15):.0f}–{min(99,(1-m["ror"])*100 - 5):.0f}%
  Use more trades (100+) to calibrate parameters before live deployment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROP FIRM RISK MANAGEMENT  (TOPSTEP CHALLENGE — HARD RULES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implement ALL of the following in a RiskManager class:

  1. TRAILING DRAWDOWN  = ${DD_LIMIT:,.0f}
     peak_equity = max(peak_equity, account.balance)
     if (peak_equity - account.balance) >= {DD_LIMIT:,.0f}:
         await suite.positions.close_all_positions()
         HALT — do not place any further orders today

  2. DAILY LOSS LIMIT   = ${DAILY_LOSS_LIM:,.0f}
     session_pnl = account.balance - session_open_balance
     if session_pnl <= -{DAILY_LOSS_LIM:,.0f}:
         await suite.positions.close_all_positions()
         HALT — do not trade for remainder of this session

  3. PROFIT TARGET      = ${PROFIT_TARGET:,.0f}  AND qualifying_days >= {QUAL_DAYS_NEED}
     if account.balance >= session_open_balance + {PROFIT_TARGET:,.0f}:
         if qualifying_days >= {QUAL_DAYS_NEED}:
             PASS — stop trading, challenge complete

  4. QUALIFYING DAYS    = {QUAL_DAYS_NEED} sessions with session P&L >= ${QUAL_MIN:.0f}
     Track daily: if session_close_pnl >= {QUAL_MIN:.0f}: qualifying_days += 1

  5. MAX CONTRACTS      = {MAX_CONTRACTS} per order
     contracts = min({MAX_CONTRACTS}, calculated_size)

  6. SESSION HOURS      = CME Globex regular hours for {instr}
     Only trade during active hours (approx 08:30–15:15 CT for equity index)
     No positions held overnight (flatten before session close)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSITION SIZING  ({bp["sizing"]})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Base contracts     = {m["rec_contracts"]}  (Kelly-recommended)
  Full Kelly f       = {m["kelly"]:.4f}
  Applied Kelly      = {m["kelly_rec_frac"]:.4f}  ({m["kelly_rec_frac"]*100:.2f}% of DD buffer)

  Python function:
    def calculate_size(account_balance: float, peak_equity: float,
                       base_contracts: int = {m["rec_contracts"]}) -> int:
        buffer_remaining = {DD_LIMIT:.0f} - max(0, peak_equity - account_balance)
        buffer_pct = max(0.0, buffer_remaining / {DD_LIMIT:.0f})
        {sizing_code}
        return max(1, min({MAX_CONTRACTS}, round(base_contracts * size_mult)))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANTI-OVERFITTING REQUIREMENTS  (mandatory)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The following rules MUST be implemented to prevent the strategy from
fitting historical noise and producing no trades on live data:

  1. PARAMETER ROBUSTNESS
     All indicator periods must be ROUND NUMBERS (9, 10, 14, 20, 21, 50).
     Never use specific optimised values like 13, 17, or 23.
     Define all parameters in a PARAMS dict at the top of the file.

  2. MINIMUM TRADE FREQUENCY GUARD
     Track trades per session. If session_trade_count < {max(1, bp["td"]//2)} by
     1 hour before session close, log a WARNING:
     "Trade frequency below minimum — check if conditions are too restrictive."
     Do NOT automatically relax conditions in live trading (only alert).

  3. PARAMETER SENSITIVITY COMMENT BLOCK
     After the PARAMS dict, include a comment block showing:
     - Which parameters most affect trade frequency
     - Safe range for each parameter (±20% from default)
     - What to widen if no trades occur

  4. SIGNAL SIMPLICITY RULE
     No more than 3 entry conditions combined with AND.
     No more than 1 exit condition (stop or target — not both chained with multiple conditions).
     Time-based exits are preferred over complex indicator exits.

  5. OUT-OF-SAMPLE VALIDATION NOTE
     Include a comment at the top: the strategy was calibrated on historical data.
     Before live deployment, test on at least 2 weeks of OOS data (after your
     calibration period) and verify win rate is within ±8pp of target.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXECUTION & FRICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Commission (RT)     : ${bp["commission_rt"]:.2f}/contract
  Slippage assumption : {bp["slippage_ticks"]:.2f} ticks (${bp["slippage_ticks"]*tick_val:.2f}/contract)
  Total friction/trade: ${bp["contracts"]*(bp["commission_rt"]+bp["slippage_ticks"]*tick_val):.2f} at {bp["contracts"]} contracts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED CODE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate a single Python file with this structure:

  1. IMPORTS & DEPENDENCIES (project-x-py, polars, asyncio, logging, python-dotenv)
  2. PARAMS dict — all tuneable parameters as round numbers with comments
  3. RiskManager class — trailing DD, daily loss, qualifying day tracking
  4. SignalGenerator class — indicator calculations on Polars DataFrames
  5. PositionSizer function — Kelly-based sizing using calculate_size()
  6. Strategy class — main loop integrating signals + risk + sizing
  7. main() async function — TradingSuite.create(), event subscriptions, loop
  8. if __name__ == "__main__": asyncio.run(main())

Each class must have docstrings explaining parameters and what triggers an entry/exit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BACKTEST ACCEPTANCE CRITERIA  (OOS forward-test required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test on at least 2 weeks of out-of-sample data after your calibration
period. Do NOT attempt a live challenge until ALL pass:

  ✓  Win rate          ≥ {max(0,bp["wr"]-5):.0f}%  (target: {bp["wr"]:.0f}%)
  ✓  R/R               ≥ {max(0.5, bp["aw"]/bp["al"]-0.2):.2f}  (target: {bp["aw"]/bp["al"]:.2f})
  ✓  Net EV/trade      > $0  (c × (wr×aw − (1−wr)×al) − friction)
  ✓  Max drawdown      < ${DD_LIMIT*0.8:,.0f}  (80% of ${DD_LIMIT:,.0f} trailing limit)
  ✓  MC pass rate      ≥ {max(0, bp["target_pass"]-10):.0f}%  at "Typical" calibration confidence
  ✓  Bust rate         < {bp["max_bust"]+5:.0f}%  at "Typical" calibration confidence
  ✓  WFE score         ≥ 0.65  (run Walk Forward test in simulator)
  ✓  Qualifying days   ≥ 5 in ≥ 70% of challenge simulations
  ✓  Trade count       ≥ {max(1,bp["td"]//2)} trades/session in OOS period
  ✓  Days to $3k (EV)  ≤ 25  (5-day buffer before 30-day cutoff)

CALIBRATION WARNING:
  Monte Carlo pass rates at "None" calibration confidence are THEORETICAL
  upper bounds (90–100%). With "Typical" uncertainty (±5pp WR, ±20% payoffs),
  realistic pass rates are {max(0,int((1-m["ror"])*100)-15):.0f}–{min(99,int((1-m["ror"])*100)-5):.0f}%.
  Do not use "None" mode to justify a challenge attempt.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION CHECKLIST  (AI self-check before outputting code)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [ ] Uses project-x-py TradingSuite — no raw HTTP calls
  [ ] Credentials loaded from environment variables only
  [ ] All 5 Topstep risk rules implemented in RiskManager
  [ ] calculate_size() uses {bp["sizing"]} formula exactly as specified
  [ ] Max contracts hard-capped at {MAX_CONTRACTS}
  [ ] Avg win/loss in TRUE DOLLARS per contract (no instrument scaling)
  [ ] EV formula: c × (wr×aw − (1−wr)×al) − c×friction — NO other factors
  [ ] Entry conditions ≤ 3 (anti-overfitting rule)
  [ ] Round-number indicator parameters only (9, 10, 14, 20, 21, 50)
  [ ] Minimum trade frequency warning implemented ({max(1,bp["td"]//2)} trades/session threshold)
  [ ] Session hours enforced — no overnight positions
  [ ] OOS validation note in file header

Generate the complete, runnable Python file now.
{"If no entry/exit logic was specified above, implement PLACEHOLDER signals labelled clearly with # PLACEHOLDER — replace with your own tested logic" if not has_context else ""}
"""
        return prompt


    def _copy_alg_config(self):
        if self._last_config_text:
            self.clipboard_clear()
            self.clipboard_append(self._last_config_text)
            self.alg_status.config(text="Stats report copied to clipboard!")
        else:
            self.alg_status.config(text="Run analysis first.")

    def _copy_ai_prompt(self):
        if self._last_ai_prompt:
            self.clipboard_clear()
            self.clipboard_append(self._last_ai_prompt)
            self.alg_status.config(text="AI prompt copied — paste into Claude or GPT-4!")
        else:
            self.alg_status.config(text="Run analysis first to generate the prompt.")

    def _draw_alg_charts(self, r, bp):
        m = r["metrics"]
        C2 = C  # colour palette alias

        # ── 1. Kelly Frontier: pass rate vs contracts ─────────────
        ax = self.ax_alg_kelly
        self._ax_base(ax, "Kelly Sizing Frontier")
        contracts_range = np.arange(1, MAX_CONTRACTS + 1)
        kelly_f = m["kelly"]
        kfrac   = bp["kelly_frac"]
        rec_c   = m["rec_contracts"]
        # Analytical expected P&L growth rate ∝ f*(EV/trade) scaled by contracts
        # Use Gaussian approximation pass probability
        pass_approx = []
        for c in contracts_range:
            ev_d  = c * bp["td"] * (bp["wr"]/100 * bp["aw"] - (1-bp["wr"]/100) * bp["al"]) -                     c * _trade_cost(bp["instrument"], bp["commission_rt"], bp["slippage_ticks"]) * bp["td"]
            var_t = bp["wr"]/100 * bp["aw"]**2 + (1-bp["wr"]/100) * bp["al"]**2 -                     (bp["wr"]/100 * bp["aw"] - (1-bp["wr"]/100) * bp["al"])**2
            sig_d = c * np.sqrt(bp["td"] * var_t)
            if sig_d > 0:
                import scipy.stats as _ss
                z = (PROFIT_TARGET - ev_d * MAX_DAYS) / (sig_d * np.sqrt(MAX_DAYS))
                pa = float(_ss.norm.sf(z)) * 0.85  # rough correction for DD constraint
            else:
                pa = 0.0
            pass_approx.append(min(95, max(0, pa * 100)))
        ax.plot(contracts_range, pass_approx, color=C2["blue"], lw=2, label="Pass rate (approx)")
        ax.axvline(rec_c, color=C2["win"], lw=2, ls="--", label=f"Rec. ({rec_c}c)")
        ax.fill_between(contracts_range, pass_approx, alpha=0.10, color=C2["blue"])
        if r["sim_pass_by_c"]:
            sc = list(r["sim_pass_by_c"].keys())
            sv = [r["sim_pass_by_c"][c] for c in sc]
            ax.scatter(sc, sv, color=C2["amber"], s=18, zorder=5, label="Simulated", alpha=0.8)
        ax.set_xlabel("Contracts", fontsize=8); ax.set_ylabel("Pass rate (%)", fontsize=8)
        ax.set_ylim(0, 100); ax.legend(fontsize=6.5, framealpha=0)
        ax.yaxis.grid(True, color=C2["border"], lw=0.4); ax.set_axisbelow(True)
        self._info_note(ax, "alg_kelly")

        # ── 2. Break-even map: win rate vs R/R ───────────────────
        ax = self.ax_alg_beven
        self._ax_base(ax, "Break-Even Map  (win rate vs R/R)")
        wr_grid = np.linspace(20, 85, 120)
        rr_grid = np.linspace(0.3, 5.0, 120)
        WR, RR  = np.meshgrid(wr_grid, rr_grid)
        EV = (WR/100) * RR - (1 - WR/100)  # normalised EV (units of avg loss)
        cmap_be = mcolors.LinearSegmentedColormap.from_list("be", [C2["bust"], "#DDAA44", C2["win"]])
        im = ax.contourf(WR, RR, EV, levels=20, cmap=cmap_be, alpha=0.82)
        ax.contour(WR, RR, EV, levels=[0], colors=[C2["text"]], linewidths=[1.2])
        cur_rr = bp["aw"] / bp["al"] if bp["al"] > 0 else 1
        ax.scatter([bp["wr"]], [cur_rr], marker="*", s=220, color="white",
                   zorder=5, edgecolors=C2["text"], linewidths=0.8)
        ax.set_xlabel("Win rate (%)", fontsize=8); ax.set_ylabel("R/R (avg win / avg loss)", fontsize=8)
        ax.text(0.97, 0.97, "★ = current", transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="white", fontweight="bold")
        ax.text(0.97, 0.03, "Contour line = break-even", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color=C2["text"])
        self._info_note(ax, "alg_beven")

        # ── 3. Qualifying day probability ─────────────────────────
        ax = self.ax_alg_qual
        self._ax_base(ax, "Qualifying Day Probability")
        import scipy.stats as ss
        p_q = m["p_qual_day"]
        # Binomial distribution: P(≥5 qual days in k days available)
        days_avail = np.arange(5, MAX_DAYS + 1)
        prob_enough = [float(1 - ss.binom.cdf(QUAL_DAYS_NEED - 1, d, p_q)) for d in days_avail]
        ax.plot(days_avail, [p*100 for p in prob_enough], color=C2["blue"], lw=2)
        ax.fill_between(days_avail, [p*100 for p in prob_enough], alpha=0.12, color=C2["blue"])
        # Mark days-to-target
        dtt = m["days_to_target"]
        if 5 <= dtt <= MAX_DAYS:
            ax.axvline(dtt, color=C2["amber"], lw=1.5, ls="--",
                       label=f"EV days-to-target: {dtt:.0f}")
        ax.axhline(90, color=C2["muted"], lw=0.8, ls=":", alpha=0.7)
        ax.text(MAX_DAYS*0.98, 91, "90%", fontsize=7, color=C2["muted"], ha="right")
        ax.set_xlabel("Trading days available", fontsize=8)
        ax.set_ylabel("P(≥5 qualifying days) %", fontsize=8)
        ax.set_ylim(0, 105); ax.legend(fontsize=7, framealpha=0)
        ax.yaxis.grid(True, color=C2["border"], lw=0.4); ax.set_axisbelow(True)
        self._info_note(ax, "alg_qual")

        # ── 4. Risk of ruin curve ─────────────────────────────────
        ax = self.ax_alg_ror
        self._ax_base(ax, "Risk of Ruin  (vs fraction risked per trade)")
        wr_f = bp["wr"] / 100
        rr   = cur_rr
        if wr_f > 0 and rr > 0 and wr_f * rr > (1 - wr_f):
            fracs = np.linspace(0.01, 0.5, 200)
            ror_vals = []
            for f in fracs:
                # Each fraction f scales the aw/al by f (fraction of DD buffer per trade)
                # Trade EV at fraction f: p * f*aw - (1-p) * f*al  (in DD-buffer units)
                # Convert to daily P&L: EV_day = td * (p*f*aw - (1-p)*f*al)
                # σ_day = sqrt(td) * f * sqrt(p*aw² + (1-p)*al² - ev_trade²)
                # Then use two-sided gambler's ruin against barriers DD and TARGET
                ev_trade_f  = wr_f * f * bp["aw"] - (1 - wr_f) * f * bp["al"]
                var_trade_f = (wr_f * (f*bp["aw"])**2 + (1-wr_f) * (f*bp["al"])**2
                               - ev_trade_f**2)
                ev_day_f    = bp["td"] * ev_trade_f
                std_day_f   = max(0.001, (bp["td"] * var_trade_f) ** 0.5)
                r_ror = _ror_two_sided(ev_day_f, std_day_f)
                ror_vals.append(r_ror * 100)
            ax.plot(fracs * 100, ror_vals, color=C2["bust"], lw=2)
            ax.fill_between(fracs * 100, ror_vals, alpha=0.12, color=C2["bust"])
            cur_f = m.get("kelly_rec_frac", 0.05) * 100
            ax.axvline(cur_f, color=C2["win"], lw=1.5, ls="--",
                       label=f"Rec. fraction: {cur_f:.1f}%")
            ax.axhline(5, color=C2["amber"], lw=0.8, ls=":", alpha=0.8)
            ax.text(49, 6, "5% RoR threshold", fontsize=6.5, color=C2["amber"], ha="right")
        else:
            ax.text(0.5, 0.5, "Negative EV — RoR = 100%", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=C2["bust"])
        ax.set_xlabel("Fraction of capital risked per trade (%)", fontsize=8)
        ax.set_ylabel("Risk of ruin (%)", fontsize=8)
        ax.legend(fontsize=7, framealpha=0)
        ax.yaxis.grid(True, color=C2["border"], lw=0.4); ax.set_axisbelow(True)
        self._info_note(ax, "alg_ror")

        # ── 5. WFE sensitivity map: noise% vs degradation% ───────
        ax = self.ax_alg_wfe_map
        self._ax_base(ax, "WFE Sensitivity  (noise vs degradation)")
        noise_vals = np.linspace(0, 30, 40)
        deg_vals   = np.linspace(0, 40, 40)
        N2, D2 = np.meshgrid(noise_vals, deg_vals)
        # Analytical WFE estimate:
        # WFE ≈ (1 - d*0.8) / (1 + n*0.5)   where d=degradation fraction, n=noise fraction
        WFE_grid = (1 - D2/100 * 0.8) / (1 + N2/100 * 0.5)
        WFE_grid = np.clip(WFE_grid, 0, 1.3)
        cmap_wfe = mcolors.LinearSegmentedColormap.from_list("wfe",
            [C2["bust"], C2["amber"], C2["gold"], C2["win"]])
        im2 = ax.contourf(N2, D2, WFE_grid, levels=np.linspace(0, 1.2, 24), cmap=cmap_wfe)
        ax.contour(N2, D2, WFE_grid, levels=[0.50, 0.65, 0.80], colors=["white"],
                   linewidths=[0.8], linestyles=["--"])
        ax.text(15, 5,  "0.80", fontsize=7, color="white", ha="center")
        ax.text(22, 18, "0.65", fontsize=7, color="white", ha="center")
        ax.text(26, 33, "0.50", fontsize=7, color="white", ha="center")
        # Mark recommended operating point
        ax.scatter([10], [15], marker="*", s=200, color="white", zorder=5,
                   edgecolors=C2["text"], linewidths=0.6)
        ax.set_xlabel("Param noise % (fold variation)", fontsize=8)
        ax.set_ylabel("IS→OOS degradation %", fontsize=8)
        ax.text(0.97, 0.97, "* = rec. target", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="white")
        self._info_note(ax, "alg_wfe_map")

        # ── 6. Efficiency frontier: EV/day vs σ/day by contracts ─
        ax = self.ax_alg_frontier
        self._ax_base(ax, "Risk-Return Frontier  (by contract count)")
        ev_pts, std_pts, c_labels = [], [], []
        for c in range(1, min(MAX_CONTRACTS+1, 16)):
            ev_d  = c * bp["td"] * (bp["wr"]/100 * bp["aw"] - (1-bp["wr"]/100) * bp["al"]) -                     c * _trade_cost(bp["instrument"], bp["commission_rt"], bp["slippage_ticks"]) * bp["td"]
            var_t = bp["wr"]/100 * bp["aw"]**2 + (1-bp["wr"]/100) * bp["al"]**2 -                     (bp["wr"]/100 * bp["aw"] - (1-bp["wr"]/100) * bp["al"])**2
            sig_d = c * np.sqrt(bp["td"] * var_t)
            ev_pts.append(ev_d); std_pts.append(sig_d); c_labels.append(c)
        ax.plot(std_pts, ev_pts, color=C2["blue"], lw=1.5, alpha=0.6, zorder=2)
        for i, (s, e, cl) in enumerate(zip(std_pts, ev_pts, c_labels)):
            col = C2["win"] if cl == rec_c else C2["blue"]
            sz  = 60 if cl == rec_c else 22
            ax.scatter([s], [e], color=col, s=sz, zorder=4)
            if cl in (1, rec_c, min(15, MAX_CONTRACTS)):
                ax.text(s+2, e, f"{cl}c", fontsize=6.5, color=C2["text"], va="center")
        ax.axhline(0, color=C2["muted"], lw=0.8, ls=":", alpha=0.7)
        ax.set_xlabel("Daily P&L std dev ($)", fontsize=8)
        ax.set_ylabel("Daily EV ($)", fontsize=8)
        ax.yaxis.grid(True, color=C2["border"], lw=0.4)
        ax.xaxis.grid(True, color=C2["border"], lw=0.4)
        ax.set_axisbelow(True)
        self._info_note(ax, "alg_frontier")


if __name__ == "__main__":
    app = PropSimApp()
    app.mainloop()
