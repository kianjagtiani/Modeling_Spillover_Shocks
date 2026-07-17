"""
Add charts and visuals to btc_volatility_presentation.pptx
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import tempfile, os, shutil, json
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Color palette ──────────────────────────────────────────────────────────────
BG       = "#0D1B2A"
BG2      = "#1A2D42"
ORANGE   = "#F7931A"
WHITE    = "#FFFFFF"
LTBLUE   = "#CCD6E0"
GREEN    = "#27AE60"
RED      = "#E74C3C"
CYAN     = "#3DBDCA"
PURPLE   = "#9B59B6"
GOLD     = "#F1C40F"

RESULTS = "/Users/trishatjokrosapoetro/Documents/crypto/results"
PPTX_IN  = f"{RESULTS}/btc_volatility_presentation.pptx"
PPTX_OUT = f"{RESULTS}/btc_volatility_presentation.pptx"

tmpdir = tempfile.mkdtemp()

def savefig(name):
    path = os.path.join(tmpdir, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close('all')
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 – BTC Realized Volatility Time Series  (for Slide 4 EDA)
# ══════════════════════════════════════════════════════════════════════════════
def chart_btc_vol_timeseries():
    df = pd.read_parquet(f"{RESULTS}/preds_har_lgbm_hybrid_h1.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    rv_pct = np.exp(df['actual']) * 100          # annualised vol in %
    pred_pct = np.exp(df['predicted']) * 100

    fig, ax = plt.subplots(figsize=(11.2, 2.1), facecolor=BG)
    ax.set_facecolor(BG)
    ax.fill_between(rv_pct.index, rv_pct, alpha=0.25, color=ORANGE)
    ax.plot(rv_pct.index, rv_pct,   color=ORANGE, lw=1.1, label='Actual RV')
    ax.plot(pred_pct.index, pred_pct, color=CYAN,  lw=0.9, alpha=0.85, label='HAR-LGBM Hybrid Forecast')

    # Annotate key events
    events = {
        "FTX Collapse\n(Nov 2022)": "2022-11-08",
        "BTC ATH\n(Mar 2024)": "2024-03-05",
        "Carry Unwind\n(Aug 2024)": "2024-08-05",
    }
    for label, date in events.items():
        dt = pd.Timestamp(date, tz='UTC')
        yval = rv_pct.asof(dt)
        ax.annotate(label, xy=(dt, yval), xytext=(0, 18), textcoords='offset points',
                    fontsize=6, color=RED, ha='center',
                    arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

    ax.set_ylabel("Annualised Vol (%)", color=LTBLUE, fontsize=8)
    ax.tick_params(colors=LTBLUE, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(BG2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
    ax.legend(fontsize=7, facecolor=BG2, edgecolor='none', labelcolor=LTBLUE, loc='upper right')
    ax.set_title("BTC Realized Volatility — Actual vs Forecast  (2022–2026)", color=WHITE, fontsize=9, pad=6)
    fig.tight_layout(pad=0.4)
    return savefig("btc_vol_ts.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 – Year-by-Year QLIKE Improvement  (for Slide 5 Analysis)
# ══════════════════════════════════════════════════════════════════════════════
def chart_yearly_improvement():
    years     = ['2022', '2023', '2024', '2025', '2026*']
    har_q     = [0.0622, 0.0698, 0.0529, 0.0445, 0.0607]
    hybrid_q  = [0.0575, 0.0510, 0.0404, 0.0308, 0.0395]
    improve   = [7.6,    27.0,   23.8,   30.7,   34.9]
    avg_vol   = [58.5,   37.8,   49.0,   52.0,   55.7]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.7), facecolor=BG)
    x = np.arange(len(years))
    w = 0.35

    # Top: grouped QLIKE
    ax1.set_facecolor(BG)
    bars1 = ax1.bar(x - w/2, har_q,    w, color=LTBLUE,  label='HAR-RV-SJ Baseline', alpha=0.85)
    bars2 = ax1.bar(x + w/2, hybrid_q, w, color=ORANGE, label='HAR-LGBM Hybrid',     alpha=0.90)
    ax1.set_ylabel("QLIKE", color=LTBLUE, fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(years, color=LTBLUE, fontsize=8)
    ax1.tick_params(colors=LTBLUE, labelsize=7)
    for spine in ax1.spines.values(): spine.set_edgecolor(BG2)
    ax1.set_facecolor(BG)
    ax1.legend(fontsize=7, facecolor=BG2, edgecolor='none', labelcolor=LTBLUE)
    ax1.set_title("QLIKE by Year: Baseline vs Hybrid", color=WHITE, fontsize=9, pad=4)
    for bar, val in zip(bars2, hybrid_q):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f'{val:.3f}',
                 ha='center', va='bottom', color=ORANGE, fontsize=6.5)

    # Bottom: improvement %
    ax2.set_facecolor(BG)
    bar_colors = [GREEN if v > 20 else CYAN for v in improve]
    bars3 = ax2.bar(x, improve, color=bar_colors, alpha=0.85)
    ax2.set_ylabel("QLIKE Improvement %", color=LTBLUE, fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(years, color=LTBLUE, fontsize=8)
    ax2.tick_params(colors=LTBLUE, labelsize=7)
    for spine in ax2.spines.values(): spine.set_edgecolor(BG2)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
    ax2.set_title("Hybrid Improvement Over Baseline (% QLIKE reduction)", color=WHITE, fontsize=9, pad=4)
    for bar, val in zip(bars3, improve):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.1f}%',
                 ha='center', va='bottom', color=WHITE, fontsize=7.5, fontweight='bold')

    fig.tight_layout(pad=0.5, h_pad=1.5)
    return savefig("yearly_improvement.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 – QLIKE Model Comparison Bar Chart  (for Slide 6 Results)
# ══════════════════════════════════════════════════════════════════════════════
def chart_model_qlike():
    with open(f"{RESULTS}/scores_h1_sliding.json") as f:
        scores = json.load(f)

    models = list(scores.keys())
    qlikes = [scores[m]['qlike'] for m in models]
    # Sort ascending (lower = better)
    sorted_data = sorted(zip(qlikes, models))
    qlikes_s, models_s = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(5.6, 3.0), facecolor=BG)
    ax.set_facecolor(BG)
    colors = [ORANGE if m == 'HAR-LGBM Hybrid' else (CYAN if 'HAR' in m else LTBLUE) for m in models_s]
    bars = ax.barh(models_s, qlikes_s, color=colors, alpha=0.88, height=0.6)
    ax.axvline(x=qlikes_s[-1], color=RED, lw=0.8, linestyle='--', alpha=0.6, label='Worst model')

    for bar, val, m in zip(bars, qlikes_s, models_s):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left', color=WHITE, fontsize=7)
        if m == 'HAR-LGBM Hybrid':
            ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2 - 0.35,
                    '★ Best', va='center', ha='left', color=ORANGE, fontsize=6.5)

    ax.set_xlabel("QLIKE  (lower is better)", color=LTBLUE, fontsize=8)
    ax.tick_params(colors=LTBLUE, labelsize=7.5)
    for spine in ax.spines.values(): spine.set_edgecolor(BG2)
    ax.set_title("Out-of-Sample QLIKE — All 8 Models  (1,531 test days)", color=WHITE, fontsize=9, pad=5)
    fig.tight_layout(pad=0.5)
    return savefig("model_qlike.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 – Volatility Regime Performance  (for Slide 7 Backtest)
# ══════════════════════════════════════════════════════════════════════════════
def chart_regime_performance():
    regimes  = ['Low Vol\n(<33rd pct)', 'Normal Vol\n(33rd–67th)', 'High Vol\n(>67th pct)']
    qlike    = [0.0458, 0.0209, 0.0671]
    mae      = [0.2634, 0.1644, 0.2508]
    avg_act  = [28.1,   45.9,   74.8]
    avg_pred = [35.3,   48.5,   62.2]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.9), facecolor=BG)
    titles = ['QLIKE by Regime', 'MAE (log) by Regime', 'Avg Vol: Actual vs Predicted']
    ax1, ax2, ax3 = axes

    # QLIKE bar
    ax1.set_facecolor(BG)
    colors_r = [CYAN, GREEN, RED]
    ax1.bar(regimes, qlike, color=colors_r, alpha=0.85)
    for i, v in enumerate(qlike):
        ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', color=WHITE, fontsize=7.5, fontweight='bold')
    ax1.set_ylabel('QLIKE', color=LTBLUE, fontsize=8)
    ax1.tick_params(colors=LTBLUE, labelsize=7)
    for s in ax1.spines.values(): s.set_edgecolor(BG2)

    # MAE bar
    ax2.set_facecolor(BG)
    ax2.bar(regimes, mae, color=colors_r, alpha=0.85)
    for i, v in enumerate(mae):
        ax2.text(i, v + 0.003, f'{v:.3f}', ha='center', color=WHITE, fontsize=7.5, fontweight='bold')
    ax2.set_ylabel('MAE (log RV)', color=LTBLUE, fontsize=8)
    ax2.tick_params(colors=LTBLUE, labelsize=7)
    for s in ax2.spines.values(): s.set_edgecolor(BG2)

    # Actual vs Predicted grouped
    x = np.arange(len(regimes))
    w = 0.35
    ax3.set_facecolor(BG)
    ax3.bar(x - w/2, avg_act,  w, color=ORANGE, label='Actual',    alpha=0.88)
    ax3.bar(x + w/2, avg_pred, w, color=CYAN,   label='Predicted', alpha=0.85)
    ax3.set_ylabel('Avg Ann. Vol (%)', color=LTBLUE, fontsize=8)
    ax3.set_xticks(x); ax3.set_xticklabels(regimes, color=LTBLUE, fontsize=7)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
    ax3.legend(fontsize=7, facecolor=BG2, edgecolor='none', labelcolor=LTBLUE)
    ax3.tick_params(colors=LTBLUE, labelsize=7)
    for s in ax3.spines.values(): s.set_edgecolor(BG2)

    for ax, t in zip(axes, titles):
        ax.set_title(t, color=WHITE, fontsize=8.5, pad=4)

    fig.suptitle("HAR-LGBM Hybrid: Performance Across Volatility Regimes", color=WHITE, fontsize=9.5, y=1.02)
    fig.tight_layout(pad=0.5)
    return savefig("regime_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 – Actual vs Predicted Scatter  (for Slide 9 Takeaways)
# ══════════════════════════════════════════════════════════════════════════════
def chart_actual_vs_pred():
    df = pd.read_parquet(f"{RESULTS}/preds_har_lgbm_hybrid_h1.parquet")
    actual = df['actual'].values
    pred   = df['predicted'].values

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=BG)
    ax.set_facecolor(BG)

    # Density scatter via hexbin
    hb = ax.hexbin(pred, actual, gridsize=45, cmap='YlOrBr', mincnt=1, alpha=0.9)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label('Count', color=LTBLUE, fontsize=7)
    cb.ax.yaxis.set_tick_params(color=LTBLUE, labelcolor=LTBLUE)

    # Perfect-fit line
    lo = min(actual.min(), pred.min()) - 0.1
    hi = max(actual.max(), pred.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], color=WHITE, lw=1.2, linestyle='--', alpha=0.7, label='Perfect fit')

    # R² annotation
    from numpy.polynomial import polynomial as P
    corr = np.corrcoef(actual, pred)[0, 1]
    ax.text(0.05, 0.95, f'R² = {corr**2:.3f}\nCorr = {corr:.3f}',
            transform=ax.transAxes, color=ORANGE, fontsize=9,
            va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor=BG2, edgecolor=ORANGE, alpha=0.8))

    ax.set_xlabel("Predicted log(RV)", color=LTBLUE, fontsize=9)
    ax.set_ylabel("Actual log(RV)",    color=LTBLUE, fontsize=9)
    ax.tick_params(colors=LTBLUE, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor(BG2)
    ax.legend(fontsize=8, facecolor=BG2, edgecolor='none', labelcolor=LTBLUE)
    ax.set_title("HAR-LGBM Hybrid: Actual vs Predicted\n(1,531 OOS days, log-RV scale)",
                 color=WHITE, fontsize=9.5, pad=6)
    fig.tight_layout(pad=0.5)
    return savefig("actual_vs_pred.png")


# ══════════════════════════════════════════════════════════════════════════════
# INSERT CHARTS INTO PPTX
# ══════════════════════════════════════════════════════════════════════════════
def insert_image(slide, img_path, left_in, top_in, width_in, height_in):
    from pptx.util import Inches
    slide.shapes.add_picture(img_path, Inches(left_in), Inches(top_in),
                             Inches(width_in), Inches(height_in))


def main():
    print("Generating charts...")
    p1 = chart_btc_vol_timeseries()
    print(f"  [1/5] BTC vol time series: {p1}")
    p2 = chart_yearly_improvement()
    print(f"  [2/5] Year-by-year improvement: {p2}")
    p3 = chart_model_qlike()
    print(f"  [3/5] Model QLIKE bar: {p3}")
    p4 = chart_regime_performance()
    print(f"  [4/5] Regime performance: {p4}")
    p5 = chart_actual_vs_pred()
    print(f"  [5/5] Actual vs predicted scatter: {p5}")

    print("\nOpening presentation...")
    prs = Presentation(PPTX_IN)

    # ── Slide 4 (idx=3): EDA ──────────────────────────────────────────────────
    # Table ends at ~4.91". Add BTC vol time series below: top=5.05", h=2.20"
    slide4 = prs.slides[3]
    insert_image(slide4, p1, left_in=0.90, top_in=5.05, width_in=11.40, height_in=2.20)
    print("  Added BTC vol time series to Slide 4 (EDA)")

    # ── Slide 5 (idx=4): Analysis ─────────────────────────────────────────────
    # Shrink TextBox 6 to left half, add yearly chart on right
    slide5 = prs.slides[4]
    tb6 = next(s for s in slide5.shapes if s.name == "TextBox 6")
    tb6.width  = Inches(5.50)
    tb6.height = Inches(5.80)
    insert_image(slide5, p2, left_in=6.55, top_in=1.05, width_in=5.60, height_in=5.85)
    print("  Added yearly improvement chart to Slide 5 (Analysis)")

    # ── Slide 6 (idx=5): Results – All Models ─────────────────────────────────
    # Resize table to left side, add QLIKE bar on right
    slide6 = prs.slides[5]
    tbl7 = next(s for s in slide6.shapes if s.name == "Table 7")
    # Move table to left half and shrink width
    tbl7.width = Inches(5.80)
    insert_image(slide6, p3, left_in=6.90, top_in=1.55, width_in=5.40, height_in=4.20)
    print("  Added QLIKE bar chart to Slide 6 (Results – Models)")

    # ── Slide 7 (idx=6): Backtest ─────────────────────────────────────────────
    # Table ends at 3.95". Add regime chart below: top=4.10", h=3.10"
    slide7 = prs.slides[6]
    insert_image(slide7, p4, left_in=0.90, top_in=4.10, width_in=11.40, height_in=3.15)
    print("  Added regime performance chart to Slide 7 (Backtest)")

    # Also embed backtest_results.png on slide 7 right side alongside the table
    tbl7_s7 = next(s for s in slide7.shapes if s.name == "Table 7")
    tbl7_s7.width = Inches(5.80)
    insert_image(slide7, f"{RESULTS}/backtest_results.png",
                 left_in=6.90, top_in=1.50, width_in=5.40, height_in=2.55)
    print("  Added backtest_results.png to Slide 7 (Backtest)")

    # ── Slide 9 (idx=8): Takeaways ────────────────────────────────────────────
    # Shrink TextBox 6 to left half, add scatter on right
    slide9 = prs.slides[8]
    tb6_s9 = next(s for s in slide9.shapes if s.name == "TextBox 6")
    tb6_s9.width  = Inches(6.20)
    tb6_s9.height = Inches(5.80)
    insert_image(slide9, p5, left_in=7.30, top_in=1.05, width_in=5.50, height_in=5.85)
    print("  Added actual vs predicted scatter to Slide 9 (Takeaways)")

    print(f"\nSaving to {PPTX_OUT} ...")
    prs.save(PPTX_OUT)
    print("Done!")

    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
