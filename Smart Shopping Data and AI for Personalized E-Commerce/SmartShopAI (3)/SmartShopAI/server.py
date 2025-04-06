import http.server
import socketserver
import os

# Define the handler to use our templates directory
class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Map the root path to the templates directory
        if path == "/" or path == "":
            return os.path.join(os.getcwd(), "templates", "index.html")
        
        # Map .html requests to the templates directory
        if path.endswith('.html'):
            return os.path.join(os.getcwd(), "templates", os.path.basename(path))
        
        # For other requests (css, js, images), use the standard path
        return super().translate_path(path)

# Set up the server
PORT = 5000
Handler = MyHttpRequestHandler

# Ensure the static directory exists
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# Start the server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at http://0.0.0.0:{PORT}")
    httpd.serve_forever()