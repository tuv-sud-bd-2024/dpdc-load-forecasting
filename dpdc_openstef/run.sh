#!/bin/bash

echo "Starting DPDC OpenSTEF Application..."
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting server at http://localhost:8080"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
python main.py

