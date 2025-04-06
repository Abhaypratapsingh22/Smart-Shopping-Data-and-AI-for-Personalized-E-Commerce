import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from database import get_all_customers, get_all_products, get_customer, save_recommendation
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class CustomerAgent:
    """Agent that handles customer profile analysis and segmentation"""
    
    def __init__(self):
        self.customers_df = get_all_customers()
        
    def refresh_data(self):
        """Refresh customer data from the database"""
        self.customers_df = get_all_customers()
    
    def segment_customers(self):
        """Segment customers based on various attributes"""
        segments = self.customers_df['customer_segment'].value_counts()
        return segments
    
    def get_customer_segments_by_location(self):
        """Get customer segments by location"""
        segment_location = self.customers_df.groupby(['location', 'customer_segment']).size().reset_index(name='count')
        return segment_location
    
    def get_customer_age_distribution(self):
        """Get customer age distribution"""
        age_groups = pd.cut(self.customers_df['age'], bins=[0, 18, 25, 35, 50, 100], labels=['<18', '18-25', '26-35', '36-50', '50+'])
        age_distribution = age_groups.value_counts().sort_index()
        return age_distribution
    
    def get_browsing_trends(self):
        """Analyze browsing trends across customers"""
        # Extract all categories from browsing history
        all_categories = []
        for history in self.customers_df['browsing_history']:
            all_categories.extend(history)
        
        category_counts = pd.Series(all_categories).value_counts()
        return category_counts
    
    def get_purchase_trends(self):
        """Analyze purchase trends across customers"""
        # Extract all products from purchase history
        all_purchases = []
        for history in self.customers_df['purchase_history']:
            all_purchases.extend(history)
        
        purchase_counts = pd.Series(all_purchases).value_counts()
        return purchase_counts
    
    def get_customer_profile(self, customer_id):
        """Get detailed profile for a customer"""
        customer_data = get_customer(customer_id)
        if customer_data.empty:
            return None
        
        return customer_data.iloc[0]
    
    def calculate_customer_vectors(self):
        """Calculate feature vectors for all customers"""
        # Create a feature vector for each customer based on browsing and purchase history
        customers_features = {}
        
        for _, customer in self.customers_df.iterrows():
            customer_id = customer['customer_id']
            
            # Combine browsing and purchase history with weights
            browsing = customer['browsing_history']
            purchases = customer['purchase_history']
            
            # Create a feature dictionary with category counts
            features = {}
            
            # Add browsing history (lower weight)
            for category in browsing:
                if category in features:
                    features[category] += 1
                else:
                    features[category] = 1
            
            # Add purchase history (higher weight)
            for product in purchases:
                if product in features:
                    features[product] += 3  # Higher weight for purchases
                else:
                    features[product] = 3
            
            # Add demographic features
            features[f"location_{customer['location']}"] = 1
            features[f"gender_{customer['gender']}"] = 1
            features[f"segment_{customer['customer_segment']}"] = 1
            features[f"season_{customer['season']}"] = 1
            
            customers_features[customer_id] = features
        
        return customers_features


class ProductAgent:
    """Agent that handles product analysis and relationships"""
    
    def __init__(self):
        self.products_df = get_all_products()
        
    def refresh_data(self):
        """Refresh product data from the database"""
        self.products_df = get_all_products()
    
    def get_product_categories(self):
        """Get product categories and their counts"""
        categories = self.products_df['category'].value_counts()
        return categories
    
    def get_subcategories_by_category(self, category):
        """Get subcategories for a specific category"""
        subcategories = self.products_df[self.products_df['category'] == category]['subcategory'].value_counts()
        return subcategories
    
    def get_price_distribution(self):
        """Get price distribution of products"""
        price_bins = pd.cut(self.products_df['price'], bins=5)
        price_distribution = price_bins.value_counts().sort_index()
        return price_distribution
    
    def get_top_rated_products(self, n=10):
        """Get top N rated products"""
        top_products = self.products_df.sort_values('product_rating', ascending=False).head(n)
        return top_products
    
    def get_products_by_season(self):
        """Get product distribution by season"""
        season_distribution = self.products_df.groupby(['season', 'category']).size().reset_index(name='count')
        return season_distribution
    
    def calculate_product_similarity(self):
        """Calculate similarity between products"""
        # Create a document for each product
        product_docs = []
        
        for _, product in self.products_df.iterrows():
            doc = f"{product['category']} {product['subcategory']} {product['brand']} {product['season']} "
            doc += f"{product['holiday']} {product['geographical_location']} "
            doc += ' '.join(product['similar_product_list'])
            product_docs.append(doc)
        
        # Convert to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(product_docs)
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        return cosine_sim
    
    def calculate_product_vectors(self):
        """Calculate feature vectors for all products"""
        # Create a feature vector for each product based on its attributes
        products_features = {}
        
        for _, product in self.products_df.iterrows():
            product_id = product['product_id']
            
            # Create a feature dictionary
            features = {}
            
            # Add category features
            features[f"category_{product['category']}"] = 1
            features[f"subcategory_{product['subcategory']}"] = 1
            features[f"brand_{product['brand']}"] = 1
            features[f"season_{product['season']}"] = 1
            features[f"location_{product['geographical_location']}"] = 1
            
            # Add numeric features
            features["price"] = product['price'] / 5000  # Normalize price
            features["rating"] = product['product_rating'] / 5  # Normalize rating
            features["sentiment"] = product['sentiment_score']  # Already normalized
            
            # Add similar products
            for similar_product in product['similar_product_list']:
                features[f"similar_{similar_product}"] = 1
            
            products_features[product_id] = features
        
        return products_features


class RecommendationAgent:
    """Agent that generates personalized recommendations"""
    
    def __init__(self):
        self.customer_agent = CustomerAgent()
        self.product_agent = ProductAgent()
        self.sia = SentimentIntensityAnalyzer()
        
    def refresh_data(self):
        """Refresh data from both customer and product agents"""
        self.customer_agent.refresh_data()
        self.product_agent.refresh_data()
    
    def calculate_customer_product_affinity(self, customer_id):
        """Calculate affinity between a customer and all products"""
        # Get customer profile
        customer = self.customer_agent.get_customer_profile(customer_id)
        if customer is None:
            return []
        
        # Get all products
        products = self.product_agent.products_df
        
        # Calculate affinity scores
        affinity_scores = []
        
        for _, product in products.iterrows():
            score = 0
            
            # Category match (browsing history)
            if product['category'] in customer['browsing_history']:
                score += 2
            
            # Subcategory match (purchase history)
            if product['subcategory'] in customer['purchase_history']:
                score += 3
            
            # Location match
            if customer['location'] == 'Bangalore' and product['geographical_location'] == 'India':
                score += 1
            elif customer['location'] == 'Delhi' and product['geographical_location'] == 'India':
                score += 1
            elif customer['location'] == 'Mumbai' and product['geographical_location'] == 'India':
                score += 1
            elif customer['location'] == 'Chennai' and product['geographical_location'] == 'India':
                score += 1
            elif customer['location'] == 'Kolkata' and product['geographical_location'] == 'India':
                score += 1
            
            # Season match
            if customer['season'] == product['season']:
                score += 1
            
            # Holiday match
            if customer['holiday'] == product['holiday']:
                score += 1
            
            # Price factor (based on average order value)
            price_ratio = product['price'] / customer['avg_order_value']
            if 0.7 <= price_ratio <= 1.3:  # Within 30% of average order value
                score += 1
            
            # Product rating factor
            if product['product_rating'] >= 4.0:
                score += 1
            
            # Normalize score (0-1)
            max_possible_score = 10  # Update this if you add more factors
            normalized_score = score / max_possible_score
            
            # Combine with recommendation probability from the dataset
            combined_score = 0.7 * normalized_score + 0.3 * product['recommendation_probability']
            
            affinity_scores.append({
                'product_id': product['product_id'],
                'category': product['category'],
                'subcategory': product['subcategory'],
                'price': product['price'],
                'score': combined_score
            })
        
        # Sort by score
        affinity_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return affinity_scores
    
    def generate_recommendations(self, customer_id, top_n=5):
        """Generate top N recommendations for a customer"""
        affinity_scores = self.calculate_customer_product_affinity(customer_id)
        
        # Store recommendations in database
        for recommendation in affinity_scores[:top_n]:
            save_recommendation(customer_id, recommendation['product_id'], recommendation['score'])
        
        return affinity_scores[:top_n]
    
    def generate_content_based_recommendations(self, customer_id, top_n=5):
        """Generate content-based recommendations using customer and product feature vectors"""
        # Get customer profile
        customer = self.customer_agent.get_customer_profile(customer_id)
        if customer is None:
            return []
        
        # Get products
        products = self.product_agent.products_df
        
        # Create features based on customer preferences
        customer_preferences = {
            "browse_categories": customer['browsing_history'],
            "purchase_products": customer['purchase_history'],
            "location": customer['location'],
            "avg_order_value": customer['avg_order_value'],
            "segment": customer['customer_segment'],
            "season": customer['season']
        }
        
        # Score products based on customer preferences
        recommendations = []
        
        for _, product in products.iterrows():
            score = 0
            
            # Category match from browsing history
            if product['category'] in customer_preferences["browse_categories"]:
                score += 2
            
            # Subcategory match from purchase history
            if any(item == product['subcategory'] for item in customer_preferences["purchase_products"]):
                score += 3
            
            # Price match (based on avg order value)
            if 0.5 * customer_preferences["avg_order_value"] <= product['price'] <= 1.5 * customer_preferences["avg_order_value"]:
                score += 1
            
            # Seasonal match
            if product['season'] == customer_preferences["season"]:
                score += 1
            
            # Rating bonus
            score += 0.5 * product['product_rating']
            
            # Sentiment score
            score += 2 * product['sentiment_score']
            
            # Similar product match
            similar_product_matches = sum(1 for item in product['similar_product_list'] if item in customer_preferences["purchase_products"])
            score += similar_product_matches
            
            # Calculate final score
            final_score = (score / 10) * 0.7 + product['recommendation_probability'] * 0.3
            
            recommendations.append({
                'product_id': product['product_id'],
                'category': product['category'],
                'subcategory': product['subcategory'],
                'price': product['price'],
                'score': final_score
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top N recommendations
        top_recommendations = recommendations[:top_n]
        
        # Store recommendations in database
        for rec in top_recommendations:
            save_recommendation(customer_id, rec['product_id'], rec['score'])
        
        return top_recommendations
    
    def analyze_recommendation_patterns(self):
        """Analyze patterns in recommendations across all customers"""
        # This would require generating recommendations for all customers
        # For now, let's return some sample statistics
        categories = self.product_agent.get_product_categories()
        avg_price = self.product_agent.products_df['price'].mean()
        avg_rating = self.product_agent.products_df['product_rating'].mean()
        
        return {
            'category_distribution': categories,
            'avg_recommended_price': avg_price,
            'avg_recommended_rating': avg_rating
        }
    
    def get_trending_products(self, n=5):
        """Get trending products based on ratings and sentiment"""
        products = self.product_agent.products_df
        
        # Calculate a trending score
        products['trending_score'] = (
            0.4 * products['product_rating'] + 
            0.3 * products['sentiment_score'] + 
            0.3 * products['recommendation_probability']
        )
        
        # Get top N trending products
        trending_products = products.sort_values('trending_score', ascending=False).head(n)
        
        return trending_products
