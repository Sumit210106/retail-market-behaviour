from mlxtend.frequent_patterns import apriori, association_rules
from database import df  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    
    rules['antecedents'] = rules['antecedents'].apply(list)
    rules['consequents'] = rules['consequents'].apply(list)

    return rules.to_dict(orient="records")

def sales_by_time():

    # preprocess
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.dropna(subset=['CustomerID'])
    cleaned_df = cleaned_df[cleaned_df['Quantity'] > 0]
    cleaned_df = cleaned_df[cleaned_df['UnitPrice'] > 0]
    print(cleaned_df.head())

    # to date time
    cleaned_df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


    cleaned_df['Hour'] = cleaned_df['InvoiceDate'].dt.hour
    cleaned_df['DayOfWeek'] = cleaned_df['InvoiceDate'].dt.day_name()
    cleaned_df['Date'] = cleaned_df['InvoiceDate'].dt.date
    
    return 






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
