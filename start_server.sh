#!/bin/bash

# Start both Flask servers

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting Layer2 Filter API on port 5001..."
cd Layer2
python3 app_api.py &
LAYER2_PID=$!
cd ..

echo "Waiting for Layer2 to start..."
sleep 3

echo "Starting Main EEG Server on port 5000..."
python3 app.py &
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
trap "kill $LAYER2_PID $MAIN_PID; exit" INT
wait
