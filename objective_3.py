import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def filter_holidays(df):
    """Split the weeks into named holiday weeks and normal weeks.

    February is the Super Bowl, September is Labour Day, November is
    Thanksgiving, and December is the Christmas week.
    """
    needed = {"Date", "Holiday_Flag"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing column: {', '.join(sorted(missing))}.")
    dates = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    if dates.isna().any():
        raise ValueError("A Date value could not be read. Dates must be day-first, such as 05-02-2010.")
    df["Date"] = dates
    holiday = df["Holiday_Flag"].eq(1)
    holidays = df.loc[holiday].copy()

    # 2 is February, 9 is September, 11 is November, 12 is December.
    holidays["Holiday"] = holidays["Date"].dt.month.map(
        {2: "Super Bowl", 9: "Labour Day", 11: "Thanksgiving", 12: "Christmas"}
    )
    if holidays["Holiday"].isna().any():
        raise ValueError("A holiday week is not in February, September, November, or December.")
    return holidays, df.loc[~holiday].copy()

def holiday_gain(holidays, normal):
    """Gain for each store on each holiday.

    Gain is that store's average sales on the holiday minus the same store's
    average sales on normal weeks. A positive gain means the holiday week sold
    more than a normal week. A negative gain means it sold less.
    """
    holiday_avg = holidays.groupby(["Store", "Holiday"], as_index=False)["Weekly_Sales"].mean()
    normal_avg = normal.groupby("Store", as_index=False)["Weekly_Sales"].mean()
    holiday_avg = holiday_avg.rename(columns={"Weekly_Sales": "Holiday_Avg"})
    normal_avg = normal_avg.rename(columns={"Weekly_Sales": "Normal_Avg"})
    gain = holiday_avg.merge(normal_avg, on="Store")
    # Gain is the holiday week minus that store's normal week.
    gain["Gain"] = gain["Holiday_Avg"] - gain["Normal_Avg"]
    return gain

def plot_ranked_gains(gain):
    """Bar chart of store gains for each holiday.

    One panel each for the Super Bowl, Labour Day, Thanksgiving, and Christmas.
    Blue bars are the 10 stores with the highest gain. Red bars are the 10
    stores with the lowest gain. The other stores are left out.
    """
    
    holidays = ("Super Bowl", "Labour Day", "Thanksgiving", "Christmas")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
    for ax, holiday in zip(axes.ravel(), holidays):
        part = gain.loc[gain["Holiday"].eq(holiday)].sort_values("Gain")
        if part.empty:
            raise ValueError(f"No rows for {holiday}, so the chart cannot be drawn.")
        show = pd.concat([part.head(10), part.tail(10)])
        
        # Divide by 1000 so the axis is in thousands of dollars.
        ax.barh(show["Store"].astype(str), show["Gain"] / 1000, color=["#c44e52"] * 10 + ["#4c72b0"] * 10)
        
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(holiday)
        ax.set_xlabel("Gain ($ thousands)")
        ax.set_ylabel("Store")
        ax.tick_params(axis="y", labelsize=8)
    fig.legend(handles=[
        Patch(color="#4c72b0", label="Top 10"),
        Patch(color="#c44e52", label="Bottom 10"),
    ], loc="outside lower center", ncol=2)
    fig.suptitle("Top 10 and bottom 10 stores by holiday gain")
    return fig

def main():
    """Main method of objective 3.

    Finds which stores gain the most sales on Thanksgiving, Christmas, Labour Day,
    and the Super Bowl, and which barely move.
    """
    try:
        df = pd.read_csv("Walmart_Sales.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Walmart_Sales.csv was not found in the current folder.") from None
    except pd.errors.EmptyDataError:
        raise ValueError("Walmart_Sales.csv is empty.") from None
    needed = {"Store", "Weekly_Sales"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing column: {', '.join(sorted(missing))}.")

    holidays, normal = filter_holidays(df)
    gain = holiday_gain(holidays, normal)
    
    print(gain)
    plot_ranked_gains(gain)
    plt.show()

if __name__ == '__main__':
    main()