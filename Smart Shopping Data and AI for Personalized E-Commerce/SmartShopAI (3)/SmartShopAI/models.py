import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from database import get_all_customers, get_all_products

class CustomerSegmentation:
    """Model for segmenting customers based on their attributes"""
    
    def __init__(self):
        self.customers_df = get_all_customers()
        self.model = None
        self.vectors = None
    
    def refresh_data(self):
        """Refresh customer data from the database"""
        self.customers_df = get_all_customers()
    
    def prepare_features(self):
        """Prepare feature vectors for clustering"""
        # Convert categorical features to one-hot encoding
        location_dummies = pd.get_dummies(self.customers_df['location'], prefix='location')
        gender_dummies = pd.get_dummies(self.customers_df['gender'], prefix='gender')
        season_dummies = pd.get_dummies(self.customers_df['season'], prefix='season')
        
        # Process browsing and purchase history
        # Create a set of all unique categories and products
        all_categories = set()
        all_products = set()
        
        for categories in self.customers_df['browsing_history']:
            all_categories.update(categories)
        
        for products in self.customers_df['purchase_history']:
            all_products.update(products)
        
        # Create binary features for browsing history
        browsing_features = pd.DataFrame(index=self.customers_df.index)
        for category in all_categories:
            browsing_features[f'browse_{category}'] = self.customers_df['browsing_history'].apply(
                lambda x: 1 if category in x else 0
            )
        
        # Create binary features for purchase history
        purchase_features = pd.DataFrame(index=self.customers_df.index)
        for product in all_products:
            purchase_features[f'purchase_{product}'] = self.customers_df['purchase_history'].apply(
                lambda x: 1 if product in x else 0
            )
        
        # Normalize numerical features
        numerical_features = self.customers_df[['age', 'avg_order_value']].copy()
        numerical_features['age'] = numerical_features['age'] / numerical_features['age'].max()
        numerical_features['avg_order_value'] = numerical_features['avg_order_value'] / numerical_features['avg_order_value'].max()
        
        # Combine all features
        self.vectors = pd.concat([
            numerical_features,
            location_dummies,
            gender_dummies,
            season_dummies,
            browsing_features,
            purchase_features
        ], axis=1)
        
        return self.vectors
    
    def train_model(self, n_clusters=4):
        """Train a K-means clustering model on customer features"""
        if self.vectors is None:
            self.prepare_features()
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(self.vectors)
        
        # Assign clusters to customers
        self.customers_df['cluster'] = self.model.labels_
        
        return self.model
    
    def get_cluster_profiles(self):
        """Get characteristic profiles for each cluster"""
        if 'cluster' not in self.customers_df.columns:
            raise ValueError("Model must be trained before getting cluster profiles")
        
        cluster_profiles = []
        
        for cluster_id in range(len(self.model.cluster_centers_)):
            cluster_data = self.customers_df[self.customers_df['cluster'] == cluster_id]
            
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'avg_age': cluster_data['age'].mean(),
                'gender_distribution': cluster_data['gender'].value_counts(normalize=True).to_dict(),
                'top_locations': cluster_data['location'].value_counts().head(3).to_dict(),
                'avg_order_value': cluster_data['avg_order_value'].mean(),
                'top_browsing_categories': self._get_top_categories(cluster_data['browsing_history']),
                'top_purchased_products': self._get_top_categories(cluster_data['purchase_history'])
            }
            
            cluster_profiles.append(profile)
        
        return cluster_profiles
    
    def _get_top_categories(self, history_series, top_n=3):
        """Helper method to get top categories from browsing/purchase history"""
        all_items = []
        for history in history_series:
            all_items.extend(history)
        
        counts = pd.Series(all_items).value_counts().head(top_n).to_dict()
        return counts
    
    def visualize_clusters(self):
        """Create visualizations for customer clusters"""
        if 'cluster' not in self.customers_df.columns:
            raise ValueError("Model must be trained before visualization")
        
        # Visualization 1: Age vs Order Value by Cluster
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='age',
            y='avg_order_value',
            hue='cluster',
            palette='viridis',
            data=self.customers_df
        )
        plt.title('Customer Clusters: Age vs Average Order Value')
        plt.xlabel('Age')
        plt.ylabel('Average Order Value')
        
        return plt.gcf()


class ProductRecommendation:
    """Model for product recommendations"""
    
    def __init__(self):
        self.products_df = get_all_products()
        self.customers_df = get_all_customers()
        self.similarity_matrix = None
    
    def refresh_data(self):
        """Refresh product and customer data from the database"""
        self.products_df = get_all_products()
        self.customers_df = get_all_customers()
    
    def calculate_product_similarity(self):
        """Calculate similarity between products"""
        # Create text representation of products
        product_texts = []
        
        for _, product in self.products_df.iterrows():
            text = f"{product['category']} {product['subcategory']} {product['brand']} "
            text += f"{product['season']} {product['geographical_location']} "
            text += ' '.join(product['similar_product_list'])
            product_texts.append(text)
        
        # Calculate TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(product_texts)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return self.similarity_matrix
    
    def get_similar_products(self, product_id, n=5):
        """Get N most similar products to a given product"""
        if self.similarity_matrix is None:
            self.calculate_product_similarity()
        
        # Get product index
        product_idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        
        # Get similarity scores for this product
        similarity_scores = list(enumerate(self.similarity_matrix[product_idx]))
        
        # Sort based on similarity scores
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N similar products (excluding the product itself)
        similar_products_indices = [i[0] for i in similarity_scores[1:n+1]]
        
        return self.products_df.iloc[similar_products_indices]
    
    def recommend_for_customer(self, customer_id, n=5):
        """Generate recommendations for a customer"""
        # Get customer data
        customer = self.customers_df[self.customers_df['customer_id'] == customer_id].iloc[0]
        
        # Calculate scores for each product based on customer profile
        product_scores = []
        
        for idx, product in self.products_df.iterrows():
            score = 0
            
            # Category match from browsing history
            if product['category'] in customer['browsing_history']:
                score += 2
            
            # Subcategory match from purchase history
            if product['subcategory'] in customer['purchase_history']:
                score += 3
            
            # Price match (based on average order value)
            price_ratio = product['price'] / customer['avg_order_value']
            if 0.7 <= price_ratio <= 1.3:  # Within 30% of average order value
                score += 1
            
            # Season match
            if product['season'] == customer['season']:
                score += 1
            
            # Add product rating as a factor
            score += product['product_rating'] * 0.5
            
            # Add sentiment score as a factor
            score += product['sentiment_score'] * 2
            
            # Final score is a combination of calculated score and recommendation probability
            final_score = (score / 10) * 0.7 + product['recommendation_probability'] * 0.3
            
            product_scores.append((idx, final_score))
        
        # Sort products by score
        product_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        recommended_indices = [i[0] for i in product_scores[:n]]
        recommendations = self.products_df.iloc[recommended_indices].copy()
        
        # Add score to recommendations
        recommendations['score'] = [i[1] for i in product_scores[:n]]
        
        return recommendations
