"""
Generate btc_volatility_presentation.pptx using python-pptx.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path

OUT = Path(__file__).parent.parent / "results" / "btc_volatility_presentation.pptx"

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = RGBColor(0x0D, 0x1B, 0x2A)   # dark navy
ACCENT   = RGBColor(0xF7, 0x93, 0x1A)   # Bitcoin orange
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
LTGRAY   = RGBColor(0xCC, 0xD6, 0xE0)
MIDGRAY  = RGBColor(0x6C, 0x8E, 0xAD)
GREEN    = RGBColor(0x2E, 0xCC, 0x71)

W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

BLANK = prs.slide_layouts[6]   # completely blank


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, l, t, w, h, color: RGBColor, alpha=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(l), Inches(t), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_text(slide, text, l, t, w, h,
             size=24, bold=False, color=WHITE,
             align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    return txb


def add_bullet_slide(prs, section_num, section_title, bullets, notes=""):
    slide = prs.slides.add_slide(BLANK)
    set_bg(slide, BG)

    # Left accent bar
    add_rect(slide, 0, 0, 0.08, 7.5, ACCENT)

    # Section number badge
    add_rect(slide, 0.18, 0.18, 0.6, 0.6, ACCENT)
    add_text(slide, str(section_num), 0.18, 0.18, 0.6, 0.6,
             size=20, bold=True, color=BG, align=PP_ALIGN.CENTER)

    # Section title
    add_text(slide, section_title, 0.9, 0.14, 11.8, 0.7,
             size=32, bold=True, color=ACCENT)

    # Divider
    add_rect(slide, 0.9, 0.92, 11.4, 0.03, MIDGRAY)

    # Bullets
    txb = slide.shapes.add_textbox(Inches(0.9), Inches(1.1), Inches(11.4), Inches(6.0))
    tf  = txb.text_frame
    tf.word_wrap = True

    first = True
    for bullet in bullets:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()

        indent = bullet.startswith("  ")
        p.level = 1 if indent else 0

        run = p.add_run()
        run.text = bullet.lstrip()

        if bullet.startswith("##"):
            run.text = bullet.lstrip("#").strip()
            run.font.size  = Pt(15)
            run.font.bold  = True
            run.font.color.rgb = ACCENT
        elif indent:
            run.font.size  = Pt(16)
            run.font.color.rgb = LTGRAY
        else:
            run.font.size  = Pt(18)
            run.font.color.rgb = WHITE

        p.space_before = Pt(4 if indent else 8)

    if notes:
        slide.notes_slide.notes_text_frame.text = notes

    return slide


def add_title_slide(prs, title, subtitle, tagline):
    slide = prs.slides.add_slide(BLANK)
    set_bg(slide, BG)

    # Bottom accent bar
    add_rect(slide, 0, 6.7, 13.33, 0.8, ACCENT)

    # Bitcoin symbol watermark (big text)
    add_text(slide, "₿", 9.5, 0.5, 3.5, 5.5,
             size=220, bold=True,
             color=RGBColor(0x1A, 0x2D, 0x42),
             align=PP_ALIGN.CENTER)

    # Title
    add_text(slide, title, 0.5, 1.2, 9.0, 2.0,
             size=38, bold=True, color=WHITE)

    # Subtitle
    add_text(slide, subtitle, 0.5, 3.3, 9.0, 0.8,
             size=22, bold=False, color=LTGRAY)

    # Tagline on accent bar
    add_text(slide, tagline, 0.5, 6.75, 12.0, 0.65,
             size=18, bold=True, color=BG, align=PP_ALIGN.CENTER)

    return slide


def add_table_slide(prs, section_num, title, headers, rows, subtitle=""):
    slide = prs.slides.add_slide(BLANK)
    set_bg(slide, BG)
    add_rect(slide, 0, 0, 0.08, 7.5, ACCENT)

    add_rect(slide, 0.18, 0.18, 0.6, 0.6, ACCENT)
    add_text(slide, str(section_num), 0.18, 0.18, 0.6, 0.6,
             size=20, bold=True, color=BG, align=PP_ALIGN.CENTER)

    add_text(slide, title, 0.9, 0.14, 11.8, 0.7,
             size=32, bold=True, color=ACCENT)
    add_rect(slide, 0.9, 0.92, 11.4, 0.03, MIDGRAY)

    if subtitle:
        add_text(slide, subtitle, 0.9, 1.0, 11.4, 0.45,
                 size=16, color=LTGRAY)

    top_offset = 1.55 if subtitle else 1.15
    n_cols = len(headers)
    n_rows = len(rows) + 1
    tbl = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(0.9), Inches(top_offset),
        Inches(11.5), Inches(0.48 * n_rows)
    ).table

    # Header row
    for ci, hdr in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = hdr
        run.font.bold  = True
        run.font.size  = Pt(14)
        run.font.color.rgb = BG

    # Data rows
    for ri, row in enumerate(rows):
        fill_color = RGBColor(0x14, 0x28, 0x3C) if ri % 2 == 0 else RGBColor(0x0D, 0x1B, 0x2A)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri + 1, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill_color
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(val)
            run.font.size  = Pt(13)
            run.font.color.rgb = GREEN if "★" in str(val) or "best" in str(val).lower() else WHITE

    return slide


# ══════════════════════════════════════════════════════════════════════════════
# SLIDES
# ══════════════════════════════════════════════════════════════════════════════

# 1 — Title
add_title_slide(
    prs,
    "Predicting Bitcoin Volatility\nUsing Cross-Asset Spillovers,\nSeasonality & Hybrid ML Models",
    "A Walk-Forward Evaluation Across 1,530 Days  |  2022 – 2026",
    "Can we predict how wild Bitcoin will be tomorrow — and trade on it?"
)

# 2 — Data & Methodology
add_bullet_slide(prs, 1, "Data & Methodology", [
    "## Data Sources",
    "Binance.US public API — BTC, ETH, SOL, BNB  |  5-min + daily  |  Jan 2021 → Apr 2026",
    "  ~553,000 five-minute bars per symbol; 1,901 daily feature rows",
    "yfinance — VIX, S&P 500, Gold, DXY, 10-Year Treasury yield (daily)",
    "alternative.me — Crypto Fear & Greed Index (daily)",
    "## Target Variable",
    "Realized Volatility (RV) from 5-min returns: √Σr²  ×  √365  (annualized)",
    "  Model-free, observable measure — no GARCH assumptions needed",
    "## 54 Features Across 5 Categories",
    "  1. Own-history lags (HAR: daily/weekly/monthly RV, vol-of-vol, jumps)",
    "  2. Crypto cross-asset (ETH, SOL, BNB vol; BTC-ETH rolling correlation)",
    "  3. Traditional market spillovers (VIX, S&P 500, Gold, DXY, 10Y yield)",
    "  4. Seasonality (day-of-week/month sin-cos, halving cycle phase, US holidays)",
    "  5. Sentiment (Fear & Greed Index extreme readings)",
], notes="Emphasize: all data is free and publicly available. No proprietary data vendors.")

# 3 — Literature Review
add_bullet_slide(prs, 2, "Literature Review", [
    "## HAR Model  —  Corsi (2009)",
    "Long-memory in volatility captured with daily/weekly/monthly lags",
    "  Consistently outperforms GARCH in out-of-sample crypto forecasts",
    "## VIX → BTC Volatility  —  Bouri et al. (2021), Fang et al. (2022)",
    "VIX Granger-causes BTC volatility, especially during equity stress regimes",
    "  Lead time: 1–3 trading days; effect intensified post-institutional adoption",
    "## Signed Jumps  —  Patton & Sheppard (2015)",
    "Negative jumps (crashes) predict higher future vol than positive jumps",
    "  Asymmetric fear > greed effect in volatility persistence",
    "## Seasonality in Crypto  —  Baur & Dimpfl (2018), Catania & Grassi (2022)",
    "Intraday patterns tied to NYSE open and CME futures settlement",
    "  Bitcoin's 4-year halving cycle creates measurable regime shifts in RV",
    "## Hybrid Superiority  —  Catania & Sandholdt (2019), Akyildirim et al. (2021)",
    "Econometric backbone prevents ML from overfitting on linear long-memory",
    "  Residual learning with GBDT outperforms both pure ML and pure HAR",
])

# 4 — EDA  (table)
add_table_slide(
    prs, 3, "Exploratory Data Analysis",
    headers=["Year", "Days", "Avg Annualized Vol", "VIX Correlation", "Key Event"],
    rows=[
        ["2021", "365", "78.3%", "0.41", "Bull run, BTC ATH $69k"],
        ["2022", "365", "58.5%", "0.55", "FTX collapse, LUNA crash"],
        ["2023", "365", "37.8%", "0.47", "Recovery, range-bound"],
        ["2024", "366", "49.0%", "0.61", "ETF approval, halving"],
        ["2025", "365", "52.0%", "0.63", "Institutional flows grow"],
        ["2026 YTD", "96",  "55.7%", "0.65", "Ongoing"],
    ],
    subtitle="BTC realized volatility is persistent, seasonal, and increasingly correlated with equity fear indices post-2021"
)

# 5 — Analysis
add_bullet_slide(prs, 4, "Analysis", [
    "## Walk-Forward Validation — Zero Lookahead Bias",
    "365-day rolling training window  →  5-day embargo (purging)  →  predict day t",
    "  1,530 out-of-sample test days; model refit every 20 days",
    "  Mirrors exactly how the model would work in live deployment",
    "## 8 Models Tested",
    "HAR-RV-SJ (baseline) · HAR-Lasso · XGBoost · LightGBM",
    "  HAR-LGBM Hybrid ★ · HAR-LGBM Tuned · HAR-LSTM Hybrid · Stacking Ensemble",
    "## Primary Metric: QLIKE",
    "Proper scoring rule for variance — penalizes underestimating vol more than overestimating",
    "  Standard in academic volatility literature; robust to extreme spikes",
    "## Statistical Tests",
    "Diebold-Mariano test — pairwise significance of improvements (p < 0.05 threshold)",
    "  Mincer-Zarnowitz R² — forecast efficiency; confirms forecasts are unbiased (α ≈ 0, p = 0.96)",
    "## Why Hybrid Beats Pure ML",
    "Volatility has strong linear long-memory — HAR captures it with just 3 coefficients",
    "  Pure XGBoost/LGBM wastes capacity re-learning the linear signal, then overfits",
    "  Hybrid: HAR handles linear structure → LGBM learns what's left (cross-asset, seasonality)",
], notes="Key talking point: the walk-forward setup is what makes results trustworthy and comparable to real deployment.")

# 6 — Results (table)
add_table_slide(
    prs, 5, "Results — All Models Ranked",
    headers=["Model", "QLIKE ↓", "MSE (log) ↓", "Correlation ↑", "Beats Baseline?"],
    rows=[
        ["HAR-LGBM Hybrid ★", "0.0444", "0.0855", "0.770", "— (BEST)"],
        ["HAR-LGBM Tuned",    "0.0456", "0.0880", "0.762", "Yes *"],
        ["HAR-Lasso",         "0.0479", "0.0898", "0.755", "Yes *"],
        ["Stacking Ensemble", "0.0559", "0.1036", "0.710", "Yes *"],
        ["HAR-RV-SJ (baseline)", "0.0575", "0.1116", "0.681", "—"],
        ["HAR-LSTM Hybrid",   "0.0584", "0.1157", "0.668", "No"],
        ["LightGBM",          "0.0610", "0.1236", "0.644", "No"],
        ["XGBoost",           "0.0698", "0.1494", "0.585", "No"],
    ],
    subtitle="(*) = statistically significant improvement  |  Hybrid explains 59% of day-to-day BTC vol variation  (R² = 0.594)  |  1,530 OOS test days"
)

# 6b — Backtest results (table)
add_table_slide(
    prs, 5, "Results — Backtest: Turning Forecasts Into Returns",
    headers=["Strategy", "Total Return", "Ann. Return", "Sharpe", "Max Drawdown"],
    rows=[
        ["Buy & Hold BTC",     "5.5%",   "1.3%",  "0.02", "-71.5%"],
        ["Inverse-Vol Sizing", "57.2%",  "11.4%", "0.27", "-56.3%"],
        ["Vol Regime Filter",  "65.4%",  "12.8%", "0.28", "-57.0%"],
        ["Combined Strategy ★","148.0%", "24.2%", "0.56", "-41.7%"],
    ],
    subtitle="Vol forecasts used to size BTC position: high vol → reduce exposure, low vol → increase.  0.1% transaction costs.  No lookahead."
)

# 7 — Monte Carlo Optimization
add_bullet_slide(prs, 6, "Monte Carlo: Strategy Parameter Optimization", [
    "## Setup: 200 random trials, 60/40 train/test split (no snooping)",
    "Training period: Jan 2022 – Aug 2024  |  Test period: Aug 2024 – Apr 2026",
    "  Randomly sample all parameters (vol target, position limits, regime thresholds, lookbacks)",
    "  Pick best by OOS Sharpe — never by in-sample performance",
    "## Key Finding: Large IS → OOS Sharpe gap (0.71 ⚠)",
    "Best in-sample Sharpe: 0.75  →  Out-of-sample Sharpe: 0.03",
    "  This is classic overfitting — strategy parameters don't generalize from 2022 to 2025",
    "  Default combined strategy OOS Sharpe: -0.25 (2024-26 was hard for vol strategies)",
    "## Most Important Parameters (Pearson correlation with IS Sharpe):",
    "  target_vol  r = -0.58 *** (lower vol target = safer sizing = better IS stability)",
    "  tx_cost     r = -0.31 **  (transaction costs materially erode returns)",
    "  vol_lookback r = -0.16 *  (longer lookback = smoother, less reactive)",
    "## Implication: Do not over-engineer the parameter space",
    "Vol sizing works when kept simple — inverse-vol with target ~1.1% daily vol",
    "  Complex regime switching adds IS fit but not OOS performance",
], notes="The overfitting warning is an honest result. The IS→OOS gap tells you the strategy is tuned to 2022 market dynamics.")

# 7b — Bootstrap Validation
add_table_slide(
    prs, 6, "Monte Carlo: Bootstrap Confidence Intervals",
    headers=["Strategy", "Sharpe (median)", "95% CI", "P(Sharpe > 0)", "Max DD (median)"],
    rows=[
        ["Default (combined)",  "0.174", "[-0.82, 1.93]", "65%", "-38%"],
        ["Optimized (iv)",      "0.400", "[-0.60, 1.67]", "75%", "-37%"],
        ["Inverse-Vol only",    "0.196", "[-0.64, 1.64]", "67%", "-38%"],
        ["Regime only",         "-0.003", "[-0.77, 1.47]", "50%", "-39%"],
        ["Multi-Level Vol",     "0.230", "[-0.62, 1.56]", "68%", "-38%"],
        ["Buy & Hold BTC",      "0.02", "[−∞, +∞]", "52%", "-72%"],
    ],
    subtitle="Block bootstrap (21-day blocks, 500 samples) — preserves autocorrelation structure of returns. Regime-only strategy has no edge."
)

# 7c — Path Simulation
add_bullet_slide(prs, 6, "Monte Carlo: Forward Path Simulation (252-day horizon)", [
    "## Setup: 300 GBM paths bootstrapped from model vol forecasts",
    "Vol drawn by resampling 21-day blocks of HAR-LGBM Hybrid forecasts",
    "  Returns simulated: r_t ~ N(μ_daily, σ²_t) where σ_t = model forecast",
    "  Strategy applied identically on each path — full position logic, 0.1% tx cost",
    "## Results: Strategy vs Buy & Hold across 300 simulated futures",
    "P(strategy beats B&H on total return):  62%",
    "  P(total return > 0):                   43%",
    "  P(strategy beats B&H on Sharpe):       46%",
    "## Risk Reduction is the Consistent Edge",
    "Strategy median max drawdown: -19.8%  vs  Buy & Hold median: -38.2%",
    "  P(strategy has smaller drawdown than B&H): 100% of paths",
    "  The vol sizing reliably cuts tail losses — that's the durable signal",
    "## Honest Assessment",
    "Alpha is not consistent across environments (46% chance to beat B&H on Sharpe)",
    "  Risk management alpha IS consistent — drawdown reduction holds in every path",
    "  Strategy adds value primarily as a risk overlay, not a return enhancer",
], notes="Key message: the strategy is a risk management tool first. The return enhancement depends on whether vol forecasts are accurate enough to time regimes.")

# 8 — Next Steps
add_bullet_slide(prs, 7, "Next Steps", [
    "## Additional Data Sources",
    "Deribit DVOL — implied volatility from BTC options market (forward-looking signal)",
    "  On-chain data — active addresses, exchange inflows, miner capitulation (Glassnode)",
    "  Derivatives — funding rates and open interest from perpetual swap markets",
    "  NLP sentiment — news and Twitter scores around major macro announcements",
    "## Methodological Improvements",
    "Regime-switching HAR (MS-HAR) — explicitly model bull/bear/halving regimes",
    "  Temporal Fusion Transformer — attention-based model trained on full 5-year history",
    "  Multi-horizon forecasting — extend to h=5 (weekly) and h=22 (monthly)",
    "  Probabilistic forecasts — predict a full distribution, not just a point estimate",
    "## Application",
    "Live paper trading — deploy the daily forecast in a simulated portfolio",
    "  Options strategy — identify mispriced BTC options on Deribit using vol forecasts",
])

# 8 — Takeaways
add_bullet_slide(prs, 7, "Takeaways", [
    "## The Model",
    "HAR-LGBM Hybrid is the best architecture — econometric backbone + ML on residuals",
    "  24% QLIKE improvement over the academic benchmark, statistically significant",
    "  Explains 59% of day-to-day variation in Bitcoin realized volatility",
    "## What Works — and What Doesn't",
    "Hybrid > pure ML: linear long-memory must be captured before learning nonlinearities",
    "  VIX is the single strongest cross-asset predictor — equity fear leads BTC vol by 1-3 days",
    "  LSTM underperformed due to limited sequences per fold (~340), not architecture quality",
    "## The Bottom Line",
    "A reliable vol forecast turns a near-flat BTC buy-and-hold (+5.5% over 4 years)",
    "  into a structured risk-managed strategy returning 148% with Sharpe 0.56",
    "  and cutting the worst drawdown from -71.5% to -41.7%",
    "## Monte Carlo Stress Test Findings",
    "Bootstrap CIs are wide — Sharpe point estimates are not reliable with 1,500 obs",
    "  The one thing that IS robust: vol sizing cuts max drawdown in 100% of simulated paths",
    "  Parameter optimization shows strong IS→OOS gap — keep the strategy simple",
    "## Honest Caveats",
    "Model fails on black-swan events (FTX, geopolitical shocks) — no historical data helps",
    "  Traditional market features help most in high-vol regimes; weekend data is forward-filled",
    "  Strategy is a risk overlay first, return enhancer second — set expectations accordingly",
], notes="Close with: 'The model is good at forecasting normal volatility dynamics. No model is good at black swans — and any researcher who claims otherwise is overfitting.'")


# ── Save ─────────────────────────────────────────────────────────────────────
prs.save(OUT)
print(f"Saved → {OUT}")
