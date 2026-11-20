import pandas as pd
import matplotlib.pyplot as plt

def sort_by_month(df):
    dates = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return dates.dt.to_period("M").astype(str).drop_duplicates().tolist()


def calculate_monthly_sales(df, months):
    monthly_sales = (
        df.assign(Month=pd.to_datetime(df["Date"], format="%d-%m-%Y").dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)["Weekly_Sales"]
        .mean()
        .rename(columns={"Weekly_Sales": "Average_Weekly_Sales"})
    )
    monthly_sales = monthly_sales[monthly_sales["Month"].isin(months)].reset_index(drop=True)

    monthly_sales[["Year", "Month"]] = monthly_sales["Month"].str.split("-", expand=True)
    return monthly_sales[["Month", "Year", "Average_Weekly_Sales"]]
def plot_yearly_sales(monthly_sales_df):
    for year, group in monthly_sales_df.groupby("Year"):
        sales = (
            group.assign(Month=group["Month"].astype(int))
            .set_index("Month")["Average_Weekly_Sales"]
            .reindex(range(1, 13))
        )
        yearly_avg = sales.mean()  # NaN months are skipped

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sales.index, sales.values, label="Monthly average")
        ax.axhline(
            yearly_avg,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"{year} average: {yearly_avg:,.0f}",
        )

        ax.set_xticks(range(1, 13))
        ax.set_title(f"Average Weekly Sales per Month - {year}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Weekly Sales")
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
        ax.legend()
        plt.show()


def main():
    df = pd.read_csv("Walmart_Sales.csv")
    months = sort_by_month(df)
    monthly_sales_df = calculate_monthly_sales(df, months)
    print(monthly_sales_df)
    plot_yearly_sales(monthly_sales_df)


if __name__ == "__main__":
    main()