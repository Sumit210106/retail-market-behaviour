import pandas as pd

def peak_sales_insights(df):

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Weekday"] = df["InvoiceDate"].dt.day_name()

    hour_sales = df.groupby("Hour")["Quantity"].sum()
    day_sales = df.groupby("Weekday")["Quantity"].sum()

    best_hour = int(hour_sales.idxmax())
    best_day = day_sales.idxmax()

    return {
        "best_hour": best_hour,
        "best_day": best_day,
        "hourly_sales": hour_sales.to_dict(),
        "weekday_sales": day_sales.to_dict()
    }
