from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
from database import initialize_database, get_all_customers, get_all_products, get_customer
from agents import CustomerAgent, ProductAgent, RecommendationAgent
from models import CustomerSegmentation, ProductRecommendation
import json

app = Flask(__name__)

# Create the assets directory for plot images
os.makedirs('static/images', exist_ok=True)

# Initialize the database
initialize_database()

# Create directories if they don't exist
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

# Initialize agents
customer_agent = CustomerAgent()
product_agent = ProductAgent()
recommendation_agent = RecommendationAgent()

# Initialize models
customer_segmentation = CustomerSegmentation()
product_recommendation = ProductRecommendation()

# Helper function to convert matplotlib figure to base64 encoded image
def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Get data for dashboard
    customers_df = get_all_customers()
    products_df = get_all_products()
    
    # Get key metrics
    total_customers = len(customers_df)
    total_products = len(products_df)
    avg_order_value = round(customers_df['avg_order_value'].mean(), 2)
    avg_product_rating = round(products_df['product_rating'].mean(), 1)
    
    # Generate customer segments chart
    segments = customers_df['customer_segment'].value_counts()
    plt.figure(figsize=(10, 6))
    segments.plot(kind='bar')
    plt.title('Customer Segments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    segments_chart = fig_to_base64(plt.gcf())
    plt.close()
    
    # Generate product categories chart
    categories = products_df['category'].value_counts()
    plt.figure(figsize=(10, 6))
    categories.plot(kind='bar')
    plt.title('Product Categories')
    plt.xticks(rotation=45)
    plt.tight_layout()
    categories_chart = fig_to_base64(plt.gcf())
    plt.close()
    
    # Generate age distribution chart
    plt.figure(figsize=(10, 6))
    sns.histplot(customers_df['age'], bins=15, kde=True)
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    age_chart = fig_to_base64(plt.gcf())
    plt.close()
    
    # Generate location distribution chart
    locations = customers_df['location'].value_counts()
    plt.figure(figsize=(10, 6))
    locations.plot(kind='bar')
    plt.title('Customer Location Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    location_chart = fig_to_base64(plt.gcf())
    plt.close()
    
    # Generate price distribution chart
    plt.figure(figsize=(10, 6))
    sns.histplot(products_df['price'], bins=20, kde=True)
    plt.title('Product Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Count')
    price_chart = fig_to_base64(plt.gcf())
    plt.close()
    
    # Generate rating distribution chart
    plt.figure(figsize=(10, 6))
    sns.histplot(products_df['product_rating'], bins=10, kde=True)
    plt.title('Product Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    rating_chart = fig_to_base64(plt.gcf())
    plt.close()
    
    # Get trending products
    trending_products = recommendation_agent.get_trending_products(n=5)
    trending_products_json = trending_products[['product_id', 'category', 'subcategory', 'price', 'product_rating', 'trending_score']].to_dict(orient='records')
    
    return render_template('dashboard.html', 
                          total_customers=total_customers,
                          total_products=total_products,
                          avg_order_value=avg_order_value,
                          avg_product_rating=avg_product_rating,
                          segments_chart=segments_chart,
                          categories_chart=categories_chart,
                          age_chart=age_chart,
                          location_chart=location_chart,
                          price_chart=price_chart,
                          rating_chart=rating_chart,
                          trending_products=trending_products_json)

@app.route('/customer-analysis')
def customer_analysis():
    # Get all customers for dropdown
    customers_df = get_all_customers()
    customer_ids = customers_df['customer_id'].tolist()
    
    return render_template('customer_analysis.html', customer_ids=customer_ids)

@app.route('/run-segmentation', methods=['POST'])
def run_segmentation():
    try:
        # Run customer segmentation
        customer_segmentation.refresh_data()
        customer_segmentation.prepare_features()
        n_clusters = 4
        customer_segmentation.train_model(n_clusters=n_clusters)
        
        # Generate cluster visualization
        fig = customer_segmentation.visualize_clusters()
        cluster_viz = fig_to_base64(fig)
        plt.close(fig)
        
        # Get cluster profiles
        cluster_profiles = customer_segmentation.get_cluster_profiles()
        
        return jsonify({
            'success': True,
            'cluster_viz': cluster_viz,
            'cluster_profiles': cluster_profiles
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get-customer-profile', methods=['POST'])
def get_customer_profile():
    customer_id = request.json.get('customer_id')
    if not customer_id:
        return jsonify({'success': False, 'error': 'No customer ID provided'})
    
    try:
        # Get customer profile
        customer = customer_agent.get_customer_profile(customer_id)
        if customer is None:
            return jsonify({'success': False, 'error': 'Customer not found'})
        
        # Convert customer data to dict for JSON
        customer_dict = {
            'customer_id': customer['customer_id'],
            'age': int(customer['age']),
            'gender': customer['gender'],
            'location': customer['location'],
            'customer_segment': customer['customer_segment'],
            'avg_order_value': float(customer['avg_order_value']),
            'season': customer['season'],
            'holiday': customer['holiday'],
            'browsing_history': customer['browsing_history'],
            'purchase_history': customer['purchase_history']
        }
        
        # Generate browsing history chart
        browsing_counts = pd.Series(customer['browsing_history']).value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        if not browsing_counts.empty:
            browsing_counts.plot(kind='bar', ax=ax)
        plt.title("Browsing Categories")
        plt.xticks(rotation=45)
        plt.tight_layout()
        browsing_chart = fig_to_base64(plt.gcf())
        plt.close(fig)
        
        # Generate purchase history chart
        purchase_counts = pd.Series(customer['purchase_history']).value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        if not purchase_counts.empty:
            purchase_counts.plot(kind='bar', ax=ax)
        plt.title("Purchase Categories")
        plt.xticks(rotation=45)
        plt.tight_layout()
        purchase_chart = fig_to_base64(plt.gcf())
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'customer': customer_dict,
            'browsing_chart': browsing_chart,
            'purchase_chart': purchase_chart
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/browsing-trends')
def browsing_trends():
    # Get browsing trends
    trends = customer_agent.get_browsing_trends()
    
    # Generate chart
    fig, ax = plt.subplots(figsize=(10, 6))
    trends.head(10).plot(kind='bar', ax=ax)
    plt.title("Top 10 Browsing Categories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart = fig_to_base64(plt.gcf())
    plt.close(fig)
    
    return jsonify({
        'success': True,
        'chart': chart
    })

@app.route('/purchase-trends')
def purchase_trends():
    # Get purchase trends
    trends = customer_agent.get_purchase_trends()
    
    # Generate chart
    fig, ax = plt.subplots(figsize=(10, 6))
    trends.head(10).plot(kind='bar', ax=ax)
    plt.title("Top 10 Purchased Products")
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart = fig_to_base64(plt.gcf())
    plt.close(fig)
    
    return jsonify({
        'success': True,
        'chart': chart
    })

@app.route('/product-analysis')
def product_analysis():
    # Get all products
    products_df = get_all_products()
    
    # Get categories for filter
    categories = sorted(products_df['category'].unique().tolist())
    
    return render_template('product_analysis.html', categories=categories)

@app.route('/get-products-by-category', methods=['POST'])
def get_products_by_category():
    category = request.json.get('category')
    sort_by = request.json.get('sort_by', 'Price (Low to High)')
    
    products_df = get_all_products()
    
    # Apply category filter
    if category and category != 'All':
        filtered_products = products_df[products_df['category'] == category]
    else:
        filtered_products = products_df
    
    # Apply sorting
    if sort_by == 'Price (Low to High)':
        filtered_products = filtered_products.sort_values('price')
    elif sort_by == 'Price (High to Low)':
        filtered_products = filtered_products.sort_values('price', ascending=False)
    elif sort_by == 'Rating (High to Low)':
        filtered_products = filtered_products.sort_values('product_rating', ascending=False)
    
    # Convert to dict for JSON
    products_list = filtered_products[['product_id', 'category', 'subcategory', 'price', 'product_rating']].to_dict(orient='records')
    
    return jsonify({
        'success': True,
        'products': products_list
    })

@app.route('/get-product-details', methods=['POST'])
def get_product_details():
    product_id = request.json.get('product_id')
    if not product_id:
        return jsonify({'success': False, 'error': 'No product ID provided'})
    
    try:
        # Get product details
        product_df = get_all_products()
        product = product_df[product_df['product_id'] == product_id]
        
        if product.empty:
            return jsonify({'success': False, 'error': 'Product not found'})
        
        product = product.iloc[0]
        
        # Convert to dict for JSON
        product_dict = {
            'product_id': product['product_id'],
            'category': product['category'],
            'subcategory': product['subcategory'],
            'brand': product['brand'],
            'price': float(product['price']),
            'product_rating': float(product['product_rating']),
            'sentiment_score': float(product['sentiment_score']),
            'season': product['season'],
            'holiday': product['holiday'],
            'geographical_location': product['geographical_location']
        }
        
        # Get similar products
        similar_products = product_recommendation.get_similar_products(product_id)
        similar_products_list = similar_products[['product_id', 'category', 'subcategory', 'price', 'product_rating']].to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'product': product_dict,
            'similar_products': similar_products_list
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/category-distribution')
def category_distribution():
    # Get category distribution
    categories = product_agent.get_product_categories()
    
    # Generate chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories.plot(kind='bar', ax=ax)
    plt.title("Product Categories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart = fig_to_base64(plt.gcf())
    plt.close(fig)
    
    return jsonify({
        'success': True,
        'chart': chart
    })

@app.route('/calculate-similarity', methods=['POST'])
def calculate_similarity():
    try:
        # Calculate product similarity
        similarity_matrix = product_recommendation.calculate_product_similarity()
        
        # Generate heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        similarity_subset = similarity_matrix[:20, :20]
        sns.heatmap(similarity_subset, annot=False, cmap='viridis', ax=ax)
        plt.title("Product Similarity Heatmap (Top 20 Products)")
        plt.tight_layout()
        heatmap = fig_to_base64(plt.gcf())
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'heatmap': heatmap
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/recommendations')
def recommendations():
    # Get all customers for dropdown
    customers_df = get_all_customers()
    customer_ids = customers_df['customer_id'].tolist()
    
    return render_template('recommendations.html', customer_ids=customer_ids)

@app.route('/generate-recommendations', methods=['POST'])
def generate_recommendations():
    customer_id = request.json.get('customer_id')
    recommendation_type = request.json.get('recommendation_type', 'Standard')
    
    if not customer_id:
        return jsonify({'success': False, 'error': 'No customer ID provided'})
    
    try:
        # Get customer profile
        customer = customer_agent.get_customer_profile(customer_id)
        
        if customer is None:
            return jsonify({'success': False, 'error': 'Customer not found'})
        
        # Generate recommendations
        if recommendation_type == 'Standard':
            recommendations = recommendation_agent.generate_recommendations(customer_id, top_n=5)
        else:
            recommendations = recommendation_agent.generate_content_based_recommendations(customer_id, top_n=5)
        
        # Convert recommendations to dict for JSON
        recommendations_list = []
        for i, rec in enumerate(recommendations):
            # Ensure score is between 0-1 for progress bars
            normalized_score = min(max(rec['score'], 0.0), 1.0)
            
            recommendations_list.append({
                'index': i,
                'product_id': rec['product_id'],
                'category': rec['category'],
                'subcategory': rec['subcategory'],
                'price': float(rec['price']),
                'score': float(rec['score']),
                'normalized_score': normalized_score
            })
        
        # Convert customer to dict for JSON
        customer_dict = {
            'customer_id': customer['customer_id'],
            'age': int(customer['age']),
            'gender': customer['gender'],
            'location': customer['location'],
            'customer_segment': customer['customer_segment']
        }
        
        return jsonify({
            'success': True,
            'customer': customer_dict,
            'recommendations': recommendations_list
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/agent-interaction')
def agent_interaction():
    return render_template('agent_interaction.html')

@app.route('/process-agent-query', methods=['POST'])
def process_agent_query():
    query = request.json.get('query')
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    try:
        # Simple query processing logic
        response = "Sorry, I couldn't understand your query. Please try to be more specific about what information you need."
        
        # Check for specific keywords
        query = query.lower()
        
        if 'trending' in query or 'popular' in query:
            trending_products = recommendation_agent.get_trending_products(n=5)
            products_list = trending_products[['product_id', 'category', 'subcategory', 'price', 'product_rating']].to_dict(orient='records')
            response = f"Here are the trending products:\n{json.dumps(products_list, indent=2)}"
        
        elif 'customer segment' in query or 'customer segments' in query:
            segments = customer_agent.segment_customers()
            response = f"Customer segment distribution:\n{segments.to_string()}"
        
        elif 'category' in query and 'distribution' in query:
            categories = product_agent.get_product_categories()
            response = f"Product category distribution:\n{categories.to_string()}"
        
        elif 'browsing trend' in query or 'browsing trends' in query:
            trends = customer_agent.get_browsing_trends()
            response = f"Top browsing trends:\n{trends.head(10).to_string()}"
        
        elif 'purchase trend' in query or 'purchase trends' in query:
            trends = customer_agent.get_purchase_trends()
            response = f"Top purchase trends:\n{trends.head(10).to_string()}"
        
        elif 'top rated' in query or 'highest rated' in query:
            top_products = product_agent.get_top_rated_products(n=5)
            products_list = top_products[['product_id', 'category', 'subcategory', 'price', 'product_rating']].to_dict(orient='records')
            response = f"Top rated products:\n{json.dumps(products_list, indent=2)}"
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
