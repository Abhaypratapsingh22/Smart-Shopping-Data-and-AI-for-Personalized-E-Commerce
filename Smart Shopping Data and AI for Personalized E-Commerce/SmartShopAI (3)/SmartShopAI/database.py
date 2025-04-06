import sqlite3
import pandas as pd
import os
import ast
import json

def create_connection():
    """Create a database connection to the SQLite database"""
    conn = None
    try:
        # Using a relative path to store in the project directory
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ecommerce.db')
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def close_connection(conn):
    """Close the database connection"""
    if conn:
        conn.close()

def setup_database():
    """Setup the database with the required tables"""
    conn = create_connection()
    if conn is not None:
        # Create customers table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            location TEXT,
            browsing_history TEXT,
            purchase_history TEXT,
            customer_segment TEXT,
            avg_order_value REAL,
            holiday TEXT,
            season TEXT
        )
        ''')
        
        # Create products table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            category TEXT,
            subcategory TEXT,
            price REAL,
            brand TEXT,
            avg_rating_similar REAL,
            product_rating REAL,
            sentiment_score REAL,
            holiday TEXT,
            season TEXT,
            geographical_location TEXT,
            similar_product_list TEXT,
            recommendation_probability REAL
        )
        ''')
        
        # Create recommendations table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            product_id TEXT,
            score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
        ''')
        
        close_connection(conn)
        return True
    else:
        return False

def import_customer_data(csv_path):
    """Import customer data from CSV file to the database"""
    try:
        df = pd.read_csv(csv_path)
        
        # Convert lists stored as strings back to actual lists
        df['Browsing_History'] = df['Browsing_History'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df['Purchase_History'] = df['Purchase_History'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Convert lists to JSON strings for storage
        df['Browsing_History'] = df['Browsing_History'].apply(json.dumps)
        df['Purchase_History'] = df['Purchase_History'].apply(json.dumps)
        
        conn = create_connection()
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM customers")
        
        # Insert data
        for _, row in df.iterrows():
            cursor.execute('''
            INSERT INTO customers (
                customer_id, age, gender, location, browsing_history, purchase_history, 
                customer_segment, avg_order_value, holiday, season
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['Customer_ID'], row['Age'], row['Gender'], row['Location'],
                row['Browsing_History'], row['Purchase_History'], row['Customer_Segment'],
                row['Avg_Order_Value'], row['Holiday'], row['Season']
            ))
        
        conn.commit()
        close_connection(conn)
        return True
    except Exception as e:
        print(f"Error importing customer data: {e}")
        return False

def import_product_data(csv_path):
    """Import product data from CSV file to the database"""
    try:
        df = pd.read_csv(csv_path)
        
        # Convert lists stored as strings back to actual lists
        df['Similar_Product_List'] = df['Similar_Product_List'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Convert lists to JSON strings for storage
        df['Similar_Product_List'] = df['Similar_Product_List'].apply(json.dumps)
        
        conn = create_connection()
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM products")
        
        # Insert data
        for _, row in df.iterrows():
            cursor.execute('''
            INSERT INTO products (
                product_id, category, subcategory, price, brand, avg_rating_similar,
                product_rating, sentiment_score, holiday, season, geographical_location,
                similar_product_list, recommendation_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['Product_ID'], row['Category'], row['Subcategory'], row['Price'],
                row['Brand'], row['Average_Rating_of_Similar_Products'], row['Product_Rating'],
                row['Customer_Review_Sentiment_Score'], row['Holiday'], row['Season'],
                row['Geographical_Location'], row['Similar_Product_List'], row['Probability_of_Recommendation']
            ))
        
        conn.commit()
        close_connection(conn)
        return True
    except Exception as e:
        print(f"Error importing product data: {e}")
        return False

def get_all_customers():
    """Get all customers from the database"""
    conn = create_connection()
    query = "SELECT * FROM customers"
    df = pd.read_sql_query(query, conn)
    close_connection(conn)
    
    # Convert JSON strings back to lists
    df['browsing_history'] = df['browsing_history'].apply(json.loads)
    df['purchase_history'] = df['purchase_history'].apply(json.loads)
    
    return df

def get_customer(customer_id):
    """Get a specific customer from the database"""
    conn = create_connection()
    query = f"SELECT * FROM customers WHERE customer_id = ?"
    df = pd.read_sql_query(query, conn, params=(customer_id,))
    close_connection(conn)
    
    if not df.empty:
        # Convert JSON strings back to lists
        df['browsing_history'] = df['browsing_history'].apply(json.loads)
        df['purchase_history'] = df['purchase_history'].apply(json.loads)
    
    return df

def get_all_products():
    """Get all products from the database"""
    conn = create_connection()
    query = "SELECT * FROM products"
    df = pd.read_sql_query(query, conn)
    close_connection(conn)
    
    # Convert JSON strings back to lists
    df['similar_product_list'] = df['similar_product_list'].apply(json.loads)
    
    return df

def get_product(product_id):
    """Get a specific product from the database"""
    conn = create_connection()
    query = f"SELECT * FROM products WHERE product_id = ?"
    df = pd.read_sql_query(query, conn, params=(product_id,))
    close_connection(conn)
    
    if not df.empty:
        # Convert JSON strings back to lists
        df['similar_product_list'] = df['similar_product_list'].apply(json.loads)
    
    return df

def get_products_by_category(category):
    """Get products by category from the database"""
    conn = create_connection()
    query = f"SELECT * FROM products WHERE category = ?"
    df = pd.read_sql_query(query, conn, params=(category,))
    close_connection(conn)
    
    # Convert JSON strings back to lists
    df['similar_product_list'] = df['similar_product_list'].apply(json.loads)
    
    return df

def save_recommendation(customer_id, product_id, score):
    """Save a recommendation to the database"""
    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO recommendations (customer_id, product_id, score)
    VALUES (?, ?, ?)
    ''', (customer_id, product_id, score))
    
    conn.commit()
    close_connection(conn)
    return True

def get_recommendations(customer_id):
    """Get recommendations for a specific customer"""
    conn = create_connection()
    query = """
    SELECT r.*, p.category, p.subcategory, p.price, p.product_rating 
    FROM recommendations r
    JOIN products p ON r.product_id = p.product_id
    WHERE r.customer_id = ?
    ORDER BY r.score DESC
    """
    df = pd.read_sql_query(query, conn, params=(customer_id,))
    close_connection(conn)
    return df

def initialize_database():
    """Initialize the database with the provided data"""
    setup_result = setup_database()
    if not setup_result:
        return False
    
    # Check if data files exist
    customer_csv = "attached_assets/customer_data_collection.csv"
    product_csv = "attached_assets/product_recommendation_data.csv"
    
    if not os.path.exists(customer_csv) or not os.path.exists(product_csv):
        return False
    
    customer_import = import_customer_data(customer_csv)
    product_import = import_product_data(product_csv)
    
    return customer_import and product_import
