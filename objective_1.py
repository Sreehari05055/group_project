import pandas as pd
import matplotlib.pyplot as plt


def validate_dataframe(df):
    """Validate the columns and values needed by the analysis."""
    required = {"Date", "Store", "Weekly_Sales", "Holiday_Flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing column: {', '.join(sorted(missing))}.")
    if df.empty:
        raise ValueError("Walmart_Sales.csv contains no data.")
    if df[list(required)].isna().any().any():
        raise ValueError("Date, Store, Weekly_Sales, and Holiday_Flag cannot contain missing values.")

    try:
        pd.to_numeric(df["Weekly_Sales"])
    except (TypeError, ValueError):
        raise ValueError("Weekly_Sales must contain numeric values.") from None
    if (pd.to_numeric(df["Weekly_Sales"]) < 0).any():
        raise ValueError("Weekly_Sales cannot contain negative values.")

    holiday_flags = set(pd.to_numeric(df["Holiday_Flag"], errors="coerce").dropna().unique())
    if not holiday_flags.issubset({0, 1}):
        raise ValueError("Holiday_Flag must contain only 0 or 1.")
    if not (df["Holiday_Flag"] == 1).any():
        raise ValueError("The dataset contains no holiday weeks.")

    parsed_dates = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Date values must use day-first format such as 05-02-2010.")


def calculate_average_weekly_sales(df):
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y", errors="coerce")
    if data["Date"].isna().any():
        raise ValueError("A Date value could not be read.")

    average_weekly_sales = (
        data.groupby("Date", as_index=False)["Weekly_Sales"]
        .mean()
        .rename(columns={"Weekly_Sales": "Average_Weekly_Sales"})
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return average_weekly_sales


def calculate_holiday_surrounding_averages(df, average_weekly_sales):
    holiday_dates = set(
        pd.to_datetime(
            df.loc[df["Holiday_Flag"].eq(1), "Date"],
            format="%d-%m-%Y",
            errors="coerce",
        ).dropna()
    )

    weekly_dates = average_weekly_sales["Date"].tolist()
    positions = []
    for holiday_date in holiday_dates:
        try:
            holiday_position = weekly_dates.index(holiday_date)
        except ValueError:
            continue

        for offset in (-2, -1, 0, 1, 2):
            position = holiday_position + offset
            if 0 <= position < len(average_weekly_sales):
                positions.append({
                    "Relative_Week": offset,
                    "Average_Weekly_Sales": (
                        average_weekly_sales.iloc[position]["Average_Weekly_Sales"]
                    ),
                })

    surrounding = pd.DataFrame(positions)
    if surrounding.empty:
        raise ValueError("No holiday weeks were found.")

    return (
        surrounding.groupby("Relative_Week", as_index=False)["Average_Weekly_Sales"]
        .mean()
        .sort_values("Relative_Week")
        .reset_index(drop=True)
    )


def plot_holiday_surrounding_averages(surrounding, overall_average):
    """Plot average sales for two weeks before through two weeks after holidays."""
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(
        surrounding["Relative_Week"],
        surrounding["Average_Weekly_Sales"] / 1000,
        color="#4c72b0",
        linewidth=2,
        marker="o",
    )
    ax.axhline(
        overall_average / 1000,
        color="#55a868",
        linestyle="--",
        linewidth=1.5,
        label=f"Full-year average: ${overall_average / 1000:.1f}k",
    )
    ax.set_title("Average Weekly Sales Around Holiday Weeks")
    ax.set_xlabel("Relative Week (0 = holiday week)")
    ax.set_ylabel("Average Weekly Sales ($ thousands)")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.axvline(0, color="#c44e52", linestyle="--", linewidth=1.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    return fig


def main():
    try:
        df = pd.read_csv("Walmart_Sales.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Walmart_Sales.csv was not found in the current folder.") from None
    except pd.errors.EmptyDataError:
        raise ValueError("Walmart_Sales.csv is empty.") from None

    validate_dataframe(df)

    average_weekly_sales = calculate_average_weekly_sales(df)
    surrounding = calculate_holiday_surrounding_averages(df, average_weekly_sales)
    overall_average = average_weekly_sales["Average_Weekly_Sales"].mean()
    print(surrounding)
    plot_holiday_surrounding_averages(surrounding, overall_average)
    plt.show()


if __name__ == "__main__":
    main()
