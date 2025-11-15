from mlxtend.frequent_patterns import apriori, association_rules
from database import df  # import shared dataframe
import pandas as pd

def run_analysis(min_support=0.02, min_confidence=0.2):

    # Pivot dataset: InvoiceNo × Description
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity']\
                .sum().unstack().fillna(0)

    # Convert to 0/1
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Frequent itemsets
    frequent_items = apriori(basket, min_support=min_support, use_colnames=True)

    # Rules
    rules = association_rules(frequent_items, metric="confidence",
                              min_threshold=min_confidence)

    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    # frozenset → list (JSON friendly)
    rules['antecedents'] = rules['antecedents'].apply(list)
    rules['consequents'] = rules['consequents'].apply(list)

    return rules.to_dict(orient="records")


def sales_by_time():
    filtered = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()
    filtered['InvoiceDate'] = pd.to_datetime(filtered['InvoiceDate'])
    filtered['Hour'] = filtered['InvoiceDate'].dt.hour
    filtered['DayOfWeek'] = filtered['InvoiceDate'].dt.day_name()

    hourly_sales = filtered.groupby('Hour')['Quantity'].sum().to_dict()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_sales = filtered.groupby('DayOfWeek')['Quantity'].sum().reindex(day_order).fillna(0).to_dict()

    return {"hourly_sales": hourly_sales, "daily_sales": daily_sales}


def segment_customers_by_basket():
    filtered = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()
    filtered = filtered.dropna(subset=['CustomerID'])
    filtered['InvoiceDate'] = pd.to_datetime(filtered['InvoiceDate'], errors='coerce')
    filtered['Revenue'] = filtered['Quantity'] * filtered['UnitPrice']

    basket = filtered.groupby(['InvoiceNo', 'CustomerID']).agg({
        'Quantity': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    basket.rename(columns={'Quantity': 'BasketSize', 'Revenue': 'BasketValue'}, inplace=True)

    customer_stats = basket.groupby('CustomerID').agg({
        'BasketSize': 'mean',
        'BasketValue': 'mean',
        'InvoiceNo': 'count'
    }).rename(columns={'InvoiceNo': 'NumPurchases'}).reset_index()

    quantiles = customer_stats['BasketSize'].quantile([0.25, 0.5, 0.75])
    q25, q50, q75 = quantiles[0.25], quantiles[0.5], quantiles[0.75]

    def basket_segment(x):
        if x <= q25:
            return 'Small Basket'
        elif x <= q75:
            return 'Medium Basket'
        else:
            return 'Large Basket'

    customer_stats['Segment'] = customer_stats['BasketSize'].apply(basket_segment)

    segment_summary = (
        customer_stats.groupby('Segment')[['BasketSize', 'BasketValue', 'NumPurchases']]
        .mean()
        .round(2)
        .reset_index()
    )

    segment_counts = customer_stats['Segment'].value_counts(normalize=True).mul(100).round(2).to_dict()

    return {
        "segment_summary": segment_summary.to_dict(orient="records"),
        "segment_counts": segment_counts
    }
