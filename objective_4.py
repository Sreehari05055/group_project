import pandas as pd
import matplotlib.pyplot as plt

def categorize_temperature(df):
    """Split weeks into Cold (<45°F), Mild (45-70°F), and Hot (>70°F) temperature bands."""
    needed = {"Date", "Weekly_Sales", "Holiday_Flag", "Temperature"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing column: {', '.join(sorted(missing))}.")

    dates = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    if dates.isna().any():
        raise ValueError("A Date value could not be read. Dates must be day-first, such as 05-02-2010.")

    data = df.copy()
    data["Date"] = dates
    data["Temp_Band"] = pd.cut(
        data["Temperature"],
        bins=[-float("inf"), 45.0, 70.0, float("inf")],
        labels=["Cold (<45°F)", "Mild (45-70°F)", "Hot (>70°F)"],
        right=True
    )
    return data

def temperature_summary(df):
    """Calculate mean weekly sales and lift for holiday vs normal weeks in each temperature band."""
    needed = {"Weekly_Sales", "Holiday_Flag", "Temp_Band"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing column: {', '.join(sorted(missing))}.")

    grouped = df.groupby(["Temp_Band", "Holiday_Flag"], observed=True)["Weekly_Sales"].agg(
        Count="count", Mean="mean", Median="median"
    ).reset_index()

    piv = grouped.pivot(index="Temp_Band", columns="Holiday_Flag", values="Mean")
    lift = piv[1] - piv[0]
    lift_pct = (lift / piv[0]) * 100

    rows = []
    for band in ["Cold (<45°F)", "Mild (45-70°F)", "Hot (>70°F)"]:
        n_row = grouped[(grouped["Temp_Band"] == band) & (grouped["Holiday_Flag"] == 0)].iloc[0]
        h_row = grouped[(grouped["Temp_Band"] == band) & (grouped["Holiday_Flag"] == 1)].iloc[0]
        rows.append({
            "Temp_Band": band,
            "Normal_Count": int(n_row["Count"]),
            "Normal_Mean": n_row["Mean"],
            "Holiday_Count": int(h_row["Count"]),
            "Holiday_Mean": h_row["Mean"],
            "Gain": lift[band],
            "Gain_Pct": lift_pct[band]
        })
    return pd.DataFrame(rows)

def plot_holiday_temp_boxplot(df):
    """Box plot of weekly sales for holiday weeks across Cold, Mild, and Hot bands."""
    holidays = df[df["Holiday_Flag"].eq(1)]
    bands = ["Cold (<45°F)", "Mild (45-70°F)", "Hot (>70°F)"]
    sales_by_band = [holidays[holidays["Temp_Band"].eq(b)]["Weekly_Sales"] / 1000 for b in bands]

    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    box = ax.boxplot(sales_by_band, patch_artist=True, labels=bands,
                     medianprops=dict(color="black", linewidth=1.5),
                     flierprops=dict(marker="o", markersize=4, alpha=0.5))

    colors = ["#4c72b0", "#55a868", "#c44e52"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Holiday Weekly Sales Distribution by Temperature Band")
    ax.set_xlabel("Temperature Band")
    ax.set_ylabel("Weekly Sales ($ thousands)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    return fig

def plot_temperature_lift(summary_df):
    """Grouped bar chart comparing normal vs holiday mean sales in each temperature band."""
    bands = summary_df["Temp_Band"]
    fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
    x = range(len(bands))
    width = 0.35

    ax.bar([i - width/2 for i in x], summary_df["Normal_Mean"] / 1000, width,
           label="Normal Week", color="#4c72b0", alpha=0.85)
    ax.bar([i + width/2 for i in x], summary_df["Holiday_Mean"] / 1000, width,
           label="Holiday Week", color="#dd8452", alpha=0.85)

    ax.set_title("Mean Weekly Sales: Normal vs Holiday Weeks by Temperature Band")
    ax.set_xlabel("Temperature Band")
    ax.set_ylabel("Mean Sales ($ thousands)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bands)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, row in summary_df.iterrows():
        label = f"+{row['Gain_Pct']:.1f}%"
        ax.annotate(label, (i + width/2, row["Holiday_Mean"] / 1000),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    return fig

def main():
    """Main method of objective 4.

    Compares how holiday weeks sell in cold, mild, and hot weather against
    normal weeks in the same temperature band.
    """
    try:
        df = pd.read_csv("Walmart_Sales.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Walmart_Sales.csv was not found in the current folder.") from None
    except pd.errors.EmptyDataError:
        raise ValueError("Walmart_Sales.csv is empty.") from None

    data = categorize_temperature(df)
    summary = temperature_summary(data)
    print(summary[["Temp_Band", "Normal_Mean", "Holiday_Mean", "Gain", "Gain_Pct"]])

    plot_holiday_temp_boxplot(data)
    plot_temperature_lift(summary)
    plt.show()

if __name__ == "__main__":
    main()

