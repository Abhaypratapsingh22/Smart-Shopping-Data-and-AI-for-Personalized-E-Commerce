import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from database import get_all_customers, get_all_products, get_customer
import json

def load_data():
    """Load all data from the database"""
    customers_df = get_all_customers()
    products_df = get_all_products()
    return customers_df, products_df

def get_customer_details(customer_id):
    """Get detailed information about a customer"""
    customer = get_customer(customer_id)
    if customer.empty:
        return None
    return customer.iloc[0]

def plot_category_distribution(df, column='category', title='Category Distribution'):
    """Plot distribution of categories"""
    plt.figure(figsize=(10, 6))
    counts = df[column].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_customer_segments(df):
    """Plot distribution of customer segments"""
    plt.figure(figsize=(10, 6))
    counts = df['customer_segment'].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values)
    plt.title('Customer Segments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_price_distribution(df):
    """Plot distribution of product prices"""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=20, kde=True)
    plt.title('Product Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Count')
    return plt.gcf()

def plot_browsing_history_heatmap(customers_df):
    """Create a heatmap of browsing history by customer segment"""
    # Extract all unique categories
    all_categories = set()
    for history in customers_df['browsing_history']:
        all_categories.update(history)
    
    # Create a matrix of segment vs category
    segments = customers_df['customer_segment'].unique()
    heatmap_data = pd.DataFrame(0, index=segments, columns=sorted(all_categories))
    
    # Fill the matrix
    for _, customer in customers_df.iterrows():
        segment = customer['customer_segment']
        for category in customer['browsing_history']:
            heatmap_data.loc[segment, category] += 1
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=False, cmap='viridis')
    plt.title('Browsing History by Customer Segment')
    plt.xticks(rotation=90)
    plt.tight_layout()
    return plt.gcf()

def plot_recommendation_confidence(recommendations):
    """Plot confidence scores for recommendations"""
    plt.figure(figsize=(10, 6))
    
    # Extract product IDs and scores
    product_ids = [rec['product_id'] for rec in recommendations]
    scores = [rec['score'] for rec in recommendations]
    
    # Create bar chart
    ax = sns.barplot(x=product_ids, y=scores)
    plt.title('Recommendation Confidence Scores')
    plt.xlabel('Product ID')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_age_distribution(customers_df):
    """Plot distribution of customer ages"""
    plt.figure(figsize=(10, 6))
    sns.histplot(customers_df['age'], bins=15, kde=True)
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    return plt.gcf()

def plot_location_distribution(customers_df):
    """Plot distribution of customer locations"""
    plt.figure(figsize=(10, 6))
    counts = customers_df['location'].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values)
    plt.title('Customer Location Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_product_ratings(products_df):
    """Plot distribution of product ratings"""
    plt.figure(figsize=(10, 6))
    sns.histplot(products_df['product_rating'], bins=10, kde=True)
    plt.title('Product Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    return plt.gcf()

def plot_seasonal_distribution(products_df):
    """Plot product distribution by season"""
    plt.figure(figsize=(10, 6))
    season_counts = products_df.groupby(['season', 'category']).size().unstack(fill_value=0)
    season_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Product Distribution by Season and Category')
    plt.xlabel('Season')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def display_customer_profile(customer):
    """Display a customer profile in a readable format"""
    if customer is None:
        return
    
    cols = st.columns(2)
    
    with cols[0]:
        st.subheader("Basic Information")
        st.write(f"**Customer ID:** {customer['customer_id']}")
        st.write(f"**Age:** {customer['age']}")
        st.write(f"**Gender:** {customer['gender']}")
        st.write(f"**Location:** {customer['location']}")
        st.write(f"**Segment:** {customer['customer_segment']}")
    
    with cols[1]:
        st.subheader("Shopping Behavior")
        st.write(f"**Avg. Order Value:** ${customer['avg_order_value']:.2f}")
        st.write(f"**Season:** {customer['season']}")
        st.write(f"**Holiday Shopper:** {customer['holiday']}")
    
    st.subheader("Browsing History")
    st.write(", ".join(customer['browsing_history']))
    
    st.subheader("Purchase History")
    st.write(", ".join(customer['purchase_history']))

def display_recommendation(recommendation, index):
    """Display a product recommendation in a card-like format"""
    with st.container():
        st.subheader(f"#{index+1} Recommended Product")
        cols = st.columns(3)
        
        with cols[0]:
            st.write(f"**Product ID:** {recommendation['product_id']}")
            st.write(f"**Category:** {recommendation['category']}")
            st.write(f"**Subcategory:** {recommendation['subcategory']}")
        
        with cols[1]:
            st.write(f"**Price:** ${recommendation['price']:.2f}")
            
        with cols[2]:
            # Display confidence score with a progress bar
            st.write("**Confidence Score:**")
            # Ensure the score is between 0.0 and 1.0 for the progress bar
            normalized_score = min(max(recommendation['score'], 0.0), 1.0)
            st.progress(normalized_score)
            st.write(f"{recommendation['score']:.2f}")
        
        st.divider()
