#!/bin/bash

# Start both Flask servers using virtual environment

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please create it first: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting Layer2 Filter API on port 5001..."
cd Layer2
python app_api.py &
LAYER2_PID=$!
cd ..

echo "Waiting for Layer2 to start..."
sleep 3

echo "Starting Main EEG Server on port 5000..."
python app.py &
MAIN_PID=$!

echo ""
echo "=========================================="
echo "Both servers are running!"
echo "Main Website: http://127.0.0.1:5000"
echo "Layer2 API: http://127.0.0.1:5001"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for Ctrl+C
trap "kill $LAYER2_PID $MAIN_PID; deactivate; exit" INT
wait
