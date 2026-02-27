#!/usr/bin/env python3
"""
Business analysis charts for jobsearch.az dataset.
Produces 9 executive-grade charts saved to charts/.
"""

import re
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "jobsearch.csv"
CHARTS_DIR = BASE_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
PRIMARY_BLUE = "#1f6aa5"
SECONDARY_BLUE = "#4a9fd4"
LIGHT_GRAY = "#d0d5dd"
DARK_GRAY = "#374151"
GREEN = "#2d7a4f"
LIGHT_GREEN = "#a8d5b5"

FIGSIZE = (12, 7)
DPI = 150

# Use Arial on Windows for full Unicode/Azerbaijani character support
matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlepad": 16,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.framealpha": 0.9,
        "legend.edgecolor": LIGHT_GRAY,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_view_count(v):
    v = str(v).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*[Kk]$", v)
    if m:
        return float(m.group(1)) * 1000
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Dates
    df["date"] = pd.to_datetime(df["created_at"].str[:10])
    df["day_of_week"] = df["date"].dt.day_name()
    df["week"] = df["date"].dt.to_period("W")

    # Salary
    df["salary_num"] = pd.to_numeric(df["salary"], errors="coerce")
    df["has_salary"] = df["salary_num"].notna()

    # Views
    df["views"] = df["view_count"].apply(parse_view_count)

    # Company display name
    df["company_display"] = df["company.title"].replace(
        {"Company": "Anonymous / Undisclosed"}
    )

    return df


def save_chart(fig: plt.Figure, filename: str) -> None:
    out = CHARTS_DIR / filename
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filename}")


def annotate_bars_v(ax, bars, fmt="{:.0f}", offset=2, fontsize=8.5, color=DARK_GRAY):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=color,
                fontweight="bold",
            )


def annotate_bars_h(ax, bars, fmt="{:.0f}", offset=1, fontsize=8.5, color=DARK_GRAY):
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(
                w + offset,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(w),
                ha="left",
                va="center",
                fontsize=fontsize,
                color=color,
                fontweight="bold",
            )


# ---------------------------------------------------------------------------
# Chart 1 — Daily Job Posting Trend
# ---------------------------------------------------------------------------
def chart_01_daily_trend(df: pd.DataFrame) -> None:
    df_main = df[df["date"] >= "2026-01-28"].copy()
    daily = df_main.groupby("date").size().reset_index(name="count")
    daily = daily.sort_values("date")
    daily["rolling_avg"] = daily["count"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.bar(daily["date"], daily["count"], color=LIGHT_GRAY, width=0.8, label="Daily postings")
    ax.plot(
        daily["date"],
        daily["rolling_avg"],
        color=PRIMARY_BLUE,
        linewidth=2.5,
        label="7-day rolling average",
        zorder=5,
    )

    # Overall average annotation
    avg = daily["count"].mean()
    ax.axhline(avg, color=SECONDARY_BLUE, linestyle=":", linewidth=1.4, alpha=0.7)
    ax.text(
        daily["date"].iloc[-1],
        avg + 4,
        f"  Avg: {avg:.0f} jobs/day",
        color=SECONDARY_BLUE,
        fontsize=9,
        va="bottom",
    )

    ax.set_title("Daily Job Posting Volume — Azerbaijan Labor Market")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Job Postings")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=4))
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, daily["count"].max() * 1.18)

    fig.tight_layout()
    save_chart(fig, "01_daily_posting_trend.png")


# ---------------------------------------------------------------------------
# Chart 2 — Weekly Job Posting Volume
# ---------------------------------------------------------------------------
def chart_02_weekly_volume(df: pd.DataFrame) -> None:
    df_main = df[df["date"] >= "2026-01-28"].copy()
    weekly = df_main.groupby("week").size().reset_index(name="count")
    weekly["week_label"] = weekly["week"].apply(
        lambda p: p.start_time.strftime("%b %d")
    )
    # Mark partial weeks
    partial = {"Jan 26": True, "Feb 23": True}  # W04 (3 days), W08 (4 days)
    weekly["partial"] = weekly["week_label"].map(partial).fillna(False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = [SECONDARY_BLUE if p else PRIMARY_BLUE for p in weekly["partial"]]
    bars = ax.bar(weekly["week_label"], weekly["count"], color=colors, width=0.55, edgecolor="white")

    for bar, row in zip(bars, weekly.itertuples()):
        label = f"{row.count}" + (" *" if row.partial else "")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=DARK_GRAY,
        )

    ax.set_title("Weekly Job Posting Volume — Jan–Feb 2026")
    ax.set_xlabel("Week Starting")
    ax.set_ylabel("Total Job Postings")
    ax.set_ylim(0, weekly["count"].max() * 1.18)

    # Legend for partial weeks
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=PRIMARY_BLUE, label="Full week"),
        Patch(facecolor=SECONDARY_BLUE, label="Partial week *"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9)
    ax.text(
        0.01, -0.12,
        "* Partial weeks: Jan 26 week (3 days captured), Feb 23 week (4 days captured)",
        transform=ax.transAxes, fontsize=8, color="gray", style="italic",
    )

    fig.tight_layout()
    save_chart(fig, "02_weekly_posting_volume.png")


# ---------------------------------------------------------------------------
# Chart 3 — Top 15 Hiring Companies
# ---------------------------------------------------------------------------
def chart_03_top_companies(df: pd.DataFrame) -> None:
    top = (
        df["company_display"]
        .value_counts()
        .head(15)
        .sort_values(ascending=True)  # ascending for barh (top = longest)
        .reset_index()
    )
    top.columns = ["company", "count"]

    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(top)))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.barh(top["company"], top["count"], color=colors, height=0.65)

    annotate_bars_h(ax, bars, fmt="{:.0f}", offset=0.5)

    ax.set_title("Top 15 Most Active Hiring Companies")
    ax.set_xlabel("Number of Job Postings")
    ax.set_xlim(0, top["count"].max() * 1.18)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    ax.text(
        0.99, 0.01,
        "Observation period: Jan 28 – Feb 27, 2026",
        transform=ax.transAxes, fontsize=8, color="gray",
        ha="right", va="bottom", style="italic",
    )

    fig.tight_layout()
    save_chart(fig, "03_top_companies.png")


# ---------------------------------------------------------------------------
# Chart 4 — Top 15 In-Demand Job Roles
# ---------------------------------------------------------------------------
def chart_04_top_roles(df: pd.DataFrame) -> None:
    top = (
        df["title"]
        .str.strip()
        .value_counts()
        .head(15)
        .sort_values(ascending=True)
        .reset_index()
    )
    top.columns = ["role", "count"]

    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(top)))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.barh(top["role"], top["count"], color=colors, height=0.65)

    annotate_bars_h(ax, bars, fmt="{:.0f}", offset=0.4)

    ax.set_title("Top 15 Most In-Demand Job Roles")
    ax.set_xlabel("Number of Job Postings")
    ax.set_xlim(0, top["count"].max() * 1.22)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    ax.text(
        0.99, 0.01,
        "Job titles shown in original Azerbaijani language",
        transform=ax.transAxes, fontsize=8, color="gray",
        ha="right", va="bottom", style="italic",
    )

    fig.tight_layout()
    save_chart(fig, "04_top_job_roles.png")


# ---------------------------------------------------------------------------
# Chart 5 — Salary Band Distribution
# ---------------------------------------------------------------------------
def chart_05_salary_bands(df: pd.DataFrame) -> None:
    df_sal = df[df["has_salary"]].copy()
    bins = [0, 500, 750, 1000, 1500, 2000, float("inf")]
    labels = ["Under 500", "500–749", "750–999", "1,000–1,499", "1,500–1,999", "2,000+"]
    df_sal["band"] = pd.cut(df_sal["salary_num"], bins=bins, labels=labels, right=False)
    band_counts = df_sal["band"].value_counts().reindex(labels).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bar_colors = [PRIMARY_BLUE if l == "500–749" else SECONDARY_BLUE for l in labels]
    bars = ax.bar(labels, band_counts.values, color=bar_colors, width=0.6, edgecolor="white")

    total = len(df_sal)
    for bar, count in zip(bars, band_counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{count}\n({pct:.0f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=DARK_GRAY,
        )

    ax.set_title("Salary Band Distribution — Disclosed Job Postings")
    ax.set_xlabel("Monthly Salary (AZN)")
    ax.set_ylabel("Number of Positions")
    ax.set_ylim(0, band_counts.max() * 1.22)

    ax.text(
        0.01, -0.13,
        f"Based on {total:,} positions that disclosed salary ({total/len(df)*100:.1f}% of all {len(df):,} postings). Currency: AZN (Azerbaijani Manat).",
        transform=ax.transAxes, fontsize=8, color="gray", style="italic",
    )

    fig.tight_layout()
    save_chart(fig, "05_salary_band_distribution.png")


# ---------------------------------------------------------------------------
# Chart 6 — Salary Disclosure Rate by Top Companies
# ---------------------------------------------------------------------------
def chart_06_salary_disclosure(df: pd.DataFrame) -> None:
    top15_names = df["company_display"].value_counts().head(15).index.tolist()
    sub = df[df["company_display"].isin(top15_names)].copy()

    summary = (
        sub.groupby("company_display")
        .agg(total=("has_salary", "count"), disclosed=("has_salary", "sum"))
        .reset_index()
    )
    summary["hidden"] = summary["total"] - summary["disclosed"]
    summary["disc_pct"] = summary["disclosed"] / summary["total"] * 100
    summary = summary.sort_values("disc_pct", ascending=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.barh(summary["company_display"], summary["disclosed"], color=GREEN, label="Salary Disclosed", height=0.6)
    ax.barh(
        summary["company_display"], summary["hidden"],
        left=summary["disclosed"], color=LIGHT_GRAY, label="Salary Hidden", height=0.6,
    )

    # Percentage labels
    for _, row in summary.iterrows():
        if row["disc_pct"] > 0:
            x_pos = row["disclosed"] / 2
            ax.text(
                x_pos, row.name if isinstance(row.name, (int, float)) else list(summary.index).index(row.name),
                f"{row['disc_pct']:.0f}%",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            )

    ax.set_title("Salary Transparency: Top 15 Hiring Companies")
    ax.set_xlabel("Number of Job Postings")
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    save_chart(fig, "06_salary_disclosure_by_company.png")


# ---------------------------------------------------------------------------
# Chart 7 — Day-of-Week Posting Pattern
# ---------------------------------------------------------------------------
def chart_07_day_of_week(df: pd.DataFrame) -> None:
    df_main = df[df["date"] >= "2026-01-28"].copy()
    DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    raw_counts = df_main["day_of_week"].value_counts().reindex(DOW_ORDER).fillna(0)
    # Number of each weekday in the date range — for per-day average
    n_each = df_main.groupby("day_of_week")["date"].nunique().reindex(DOW_ORDER).fillna(1)
    avg_counts = (raw_counts / n_each).round(1)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bar_colors = [
        LIGHT_GRAY if d in ("Saturday", "Sunday") else PRIMARY_BLUE
        for d in DOW_ORDER
    ]
    bars = ax.bar(DOW_ORDER, avg_counts.values, color=bar_colors, width=0.6, edgecolor="white")

    # Weekday average reference line
    weekday_avg = avg_counts[DOW_ORDER[:5]].mean()
    ax.axhline(weekday_avg, color=SECONDARY_BLUE, linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Weekday avg: {weekday_avg:.0f}")

    annotate_bars_v(ax, bars, fmt="{:.0f}", offset=1.5)

    ax.set_title("Average Job Postings by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Number of Postings (per day)")
    ax.set_ylim(0, avg_counts.max() * 1.22)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=PRIMARY_BLUE, label="Weekday"),
        Patch(facecolor=LIGHT_GRAY, label="Weekend"),
        plt.Line2D([0], [0], color=SECONDARY_BLUE, linestyle="--", linewidth=1.5,
                   label=f"Weekday avg: {weekday_avg:.0f}"),
    ]
    ax.legend(handles=legend_els, loc="upper right", fontsize=9)
    ax.text(
        0.01, -0.11,
        "Average calculated over the number of each weekday in the Jan 28 – Feb 27, 2026 observation window.",
        transform=ax.transAxes, fontsize=8, color="gray", style="italic",
    )

    fig.tight_layout()
    save_chart(fig, "07_day_of_week_pattern.png")


# ---------------------------------------------------------------------------
# Chart 8 — Average Salary by Company
# ---------------------------------------------------------------------------
def chart_08_avg_salary(df: pd.DataFrame) -> None:
    agg = (
        df[df["has_salary"]]
        .groupby("company_display")
        .agg(avg_salary=("salary_num", "mean"), n=("salary_num", "count"))
        .reset_index()
    )
    agg = agg[agg["n"] >= 5].sort_values("avg_salary", ascending=True).tail(15)

    market_avg = df[df["has_salary"]]["salary_num"].mean()

    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(agg)))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.barh(agg["company_display"], agg["avg_salary"], color=colors, height=0.65)

    # Value annotations
    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(
            bar.get_width() + 15,
            bar.get_y() + bar.get_height() / 2,
            f"{row['avg_salary']:.0f} AZN  (n={row['n']:.0f})",
            ha="left", va="center", fontsize=8.5, color=DARK_GRAY, fontweight="bold",
        )

    # Market average line
    ax.axvline(market_avg, color=DARK_GRAY, linestyle="--", linewidth=1.5,
               label=f"Market avg: {market_avg:.0f} AZN")

    ax.set_title("Average Monthly Salary by Company")
    ax.set_xlabel("Average Salary (AZN)")
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, agg["avg_salary"].max() * 1.3)

    ax.text(
        0.01, -0.11,
        "Only companies with 5+ salary disclosures included. Currency: AZN (Azerbaijani Manat).",
        transform=ax.transAxes, fontsize=8, color="gray", style="italic",
    )

    fig.tight_layout()
    save_chart(fig, "08_avg_salary_by_company.png")


# ---------------------------------------------------------------------------
# Chart 9 — Job Views by Salary Disclosure (Grouped Bar)
# ---------------------------------------------------------------------------
def chart_09_views_disclosure(df: pd.DataFrame) -> None:
    df_v = df[df["views"].notna()].copy()

    VIEW_BINS = [0, 200, 500, 1000, 2000, float("inf")]
    VIEW_LABELS = ["0–199", "200–499", "500–999", "1,000–1,999", "2,000+"]

    df_v["view_bucket"] = pd.cut(df_v["views"], bins=VIEW_BINS, labels=VIEW_LABELS, right=False)
    df_v["salary_group"] = df_v["has_salary"].map(
        {True: "Salary Disclosed", False: "Salary Hidden"}
    )

    grouped = (
        df_v.groupby(["view_bucket", "salary_group"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(VIEW_LABELS)
    )

    x = np.arange(len(VIEW_LABELS))
    width = 0.38

    fig, ax = plt.subplots(figsize=FIGSIZE)

    bars_d = ax.bar(x - width / 2, grouped.get("Salary Disclosed", 0),
                    width, color=GREEN, label="Salary Disclosed", edgecolor="white")
    bars_h = ax.bar(x + width / 2, grouped.get("Salary Hidden", 0),
                    width, color=LIGHT_GRAY, label="Salary Hidden", edgecolor="white")

    # Annotations
    for bar in list(bars_d) + list(bars_h):
        h = bar.get_height()
        if h >= 15:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 5,
                f"{int(h)}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=DARK_GRAY,
            )

    # Key insight box
    avg_disc = df_v[df_v["has_salary"]]["views"].mean()
    avg_hid = df_v[~df_v["has_salary"]]["views"].mean()
    uplift = (avg_disc / avg_hid - 1) * 100
    ax.text(
        0.98, 0.97,
        f"Avg views — Salary Disclosed: {avg_disc:.0f}\n"
        f"Avg views — Salary Hidden:    {avg_hid:.0f}\n"
        f"Uplift from transparency:       +{uplift:.0f}%",
        transform=ax.transAxes,
        fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=LIGHT_GRAY, linewidth=1),
        color=DARK_GRAY,
        family="monospace",
    )

    ax.set_title("Job Visibility by View Count — Does Salary Disclosure Attract More Candidates?")
    ax.set_xlabel("Number of Views per Posting")
    ax.set_ylabel("Number of Job Postings")
    ax.set_xticks(x)
    ax.set_xticklabels(VIEW_LABELS)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, grouped.values.max() * 1.22)

    fig.tight_layout()
    save_chart(fig, "09_views_by_salary_disclosure.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading data from: {DATA_PATH}")
    df = load_and_clean_data()
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"\nGenerating charts -> {CHARTS_DIR}\n")

    chart_01_daily_trend(df)
    chart_02_weekly_volume(df)
    chart_03_top_companies(df)
    chart_04_top_roles(df)
    chart_05_salary_bands(df)
    chart_06_salary_disclosure(df)
    chart_07_day_of_week(df)
    chart_08_avg_salary(df)
    chart_09_views_disclosure(df)

    print(f"\nDone. {len(list(CHARTS_DIR.glob('*.png')))} charts saved to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
