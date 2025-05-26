# run_local.py
import os
from werkzeug.serving import run_simple
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from app import app # Import the app from your app.py file

# Set an environment variable to indicate local development mode
# This will be checked in app.py to conditionally apply limits
os.environ['APP_ENV'] = 'development'

# This middleware creates a "virtual" subdirectory for your app.
# It tells Werkzeug: "Any request starting with /PulseDispersionAnalyzer
# should be handled by the Flask 'app'."
# Requests to any other path will result in a 404 Not Found.
application = DispatcherMiddleware(lambda e, s: s('404 NOT FOUND', [('Content-Type', 'text/plain')]), {
    '/PulseDispersionAnalyzer': app
})

if __name__ == '__main__':
    # Run the application with the dispatcher middleware
    print("Starting local development server for PulseDispersionAnalyzer...")
    print("APP_ENV is set to:", os.environ.get('APP_ENV'))
    print("Access at: http://localhost:5000/PulseDispersionAnalyzer/")
    run_simple('localhost', 5000, application, use_reloader=True, use_debugger=True)
