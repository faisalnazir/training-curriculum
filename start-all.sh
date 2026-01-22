#!/bin/bash

# Script to start all three learning platforms

echo "Starting all three platforms..."
echo "================================"

# Function to kill background processes on exit
cleanup() {
    echo ""
    echo "Stopping all servers..."
    jobs -p | xargs -r kill 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Check required ports
PORTAL_PORT=3000
AIML_PORT=3001
QC_PORT=3002

# Start Portal
echo "🚀 Starting Portal on port $PORTAL_PORT..."
cd /Users/faisal/Dev/Training/portal
npm run dev &
PORTAL_PID=$!

# Wait a bit before starting next
sleep 2

# Start AIML-Learning
echo "🧠 Starting AI/ML Platform on port $AIML_PORT..."
cd /Users/faisal/Dev/Training/AIML-Learning/frontend
PORT=$AIML_PORT npm run dev &
AIML_PID=$!

# Wait a bit before starting next
sleep 2

# Start QC-Learning
echo "⚛️ Starting Quantum Platform on port $QC_PORT..."
cd /Users/faisal/Dev/Training/QC-Learning/quantum-computing-platform
PORT=$QC_PORT npm run dev &
QC_PID=$!

echo ""
echo "================================"
echo "All platforms started!"
echo "  📱 Portal:        http://localhost:$PORTAL_PORT"
echo "  🧠 AI/ML:         http://localhost:$AIML_PORT"
echo "  ⚛️ Quantum:       http://localhost:$QC_PORT"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "================================"

# Wait for all background jobs
wait
