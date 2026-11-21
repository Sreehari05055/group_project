import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GROUPS = ("Normal Week", "Super Bowl", "Labour Day", "Thanksgiving", "Christmas")
COLOURS = ["#8c8c8c", "#4c72b0", "#55a868", "#dd8452", "#c44e52"]

def label_weeks(df):
    
    """ We Add a Group column: Normal Week, or the holiday the flagged week belongs to."""
    
    needed = {"Date", "Holiday_Flag", "Weekly_Sales"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing column: {', '.join(sorted(missing))}.")
    df = df.copy() 
    
    # we keep the original data unchanged
    # Here Dates are day-first (05-02-2010). Unreadable ones become NaT so we can catch them below.
    
    dates = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    if dates.isna().any():
        raise ValueError("A Date value could not be read. Dates must be day-first, such as 05-02-2010.")
    df["Date"] = dates

    # Holiday_Flag is only 1 or 0, so the month tells us which holiday it is
    # (2 is February, 9 is September, 11 is November, 12 is December).
    
    month_to_holiday = {2: "Super Bowl", 9: "Labour Day", 11: "Thanksgiving", 12: "Christmas"}
    df["Group"] = "Normal Week"
    flagged = df["Holiday_Flag"].eq(1)
    df.loc[flagged, "Group"] = df.loc[flagged, "Date"].dt.month.map(month_to_holiday)
    if df["Group"].isna().any():
        raise ValueError("A holiday week is not in February, September, November, or December.")
    return df


def group_means(df):
    
    """Mean weekly sales for each group, averaged over all stores and years."""
    # reindex puts the groups in our fixed order and gives NaN to any group with no rows
    
    means = df.groupby("Group")["Weekly_Sales"].mean().reindex(GROUPS)
    if means.isna().any():
        raise ValueError(f"No rows for: {', '.join(means.index[means.isna()])}.")
    return means


def plot_mean_bars(means, counts):
    
    """Bar chart of the group means, with a dashed line at the normal week mean."""
    
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    
    # We divide by 1 million so the axis is in millions of dollars.
    
    vals = means / 1e6
    bars = ax.bar(vals.index, vals.values, color=COLOURS)
    normal = vals["Normal Week"]
    ax.axhline(normal, color="black", linestyle="--", linewidth=0.8)

    gap = vals.max() * 0.02  # small space between the bar and its label
    for bar, name in zip(bars, vals.index):
        label = f"${vals[name]:.2f}M"
        if name != "Normal Week":
            label += f" ({(vals[name] / normal - 1) * 100:+.1f}%)"
        label += f"\nn = {counts[name]}"
        
        # We Place the text above the higher of the bar and the dashed line,
        # so that a bar below normal (Christmas) does not get the line through its label.
        
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), normal) + gap, label,
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, vals.max() * 1.2)
    ax.set_ylabel("Mean weekly sales ($ millions)")
    ax.set_title("Mean weekly sales: normal week vs holiday weeks")
    return fig


def plot_box(df):
    
    """Box plot of weekly sales per group, so the spread shows as well as the mean."""
    
    data = [df.loc[df["Group"].eq(g), "Weekly_Sales"] / 1e6 for g in GROUPS]  
    
    # millions, as in the bar chart
    
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    box = ax.boxplot(data, patch_artist=True, showmeans=True,
                     medianprops={"color": "black"},
                     meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
                     flierprops={"markersize": 3, "alpha": 0.4})
    
    # Here Names are set after drawing because the boxplot argument for them (labels or 
    #tick_labels) differs between matplotlib versions. Boxes sit at positions 1 to 5.
    
    ax.set_xticks(range(1, len(GROUPS) + 1))
    ax.set_xticklabels([f"{g}\n(n = {len(d)})" for g, d in zip(GROUPS, data)])
    for patch, colour in zip(box["boxes"], COLOURS):
        patch.set_facecolor(colour)
        patch.set_alpha(0.8)
    ax.legend(handles=[
        Line2D([], [], marker="D", color="black", markerfacecolor="white", linestyle="", label="Mean"),
        Line2D([], [], color="black", label="Median"),
    ], loc="upper right")
    ax.set_ylabel("Weekly sales ($ millions)")
    ax.set_title("Spread of weekly sales: normal week vs holiday weeks")
    return fig


def main():
    """Main method of objective 2.

    Compares mean weekly sales for a normal week against the Super Bowl, Labour
    Day, Thanksgiving, and Christmas, with a bar chart of the means and a box
    plot of the spread.
    """
    try:
        df = pd.read_csv("Walmart_Sales.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Walmart_Sales.csv was not found in the current folder.") from None
    except pd.errors.EmptyDataError:
        raise ValueError("Walmart_Sales.csv is empty.") from None

    df = label_weeks(df)
    means = group_means(df)
    counts = df["Group"].value_counts()

    summary = df.groupby("Group")["Weekly_Sales"].agg(["count", "mean", "median", "std"]).reindex(GROUPS)
    # Lift is how far each group's mean is above or below the normal week mean, in percent.
    summary["lift_vs_normal_%"] = (summary["mean"] / summary.loc["Normal Week", "mean"] - 1) * 100
    print(summary.round(1))

    plot_mean_bars(means, counts)
    plot_box(df)
    plt.show()


if __name__ == '__main__':
    main()
