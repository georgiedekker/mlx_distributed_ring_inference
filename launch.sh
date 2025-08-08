#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default action
ACTION=${1:-start}

# Function to stop servers
stop_servers() {
    echo -e "${YELLOW}⏹️  Stopping distributed inference servers...${NC}"
    
    # Kill using PID file if exists
    if [ -f .server.pid ]; then
        if kill $(cat .server.pid) 2>/dev/null; then
            echo -e "${GREEN}✓ Stopped main process (PID: $(cat .server.pid))${NC}"
        fi
        rm -f .server.pid
    fi
    
    # Kill local server
    if pkill -f "python.*server.py" 2>/dev/null; then
        echo -e "${GREEN}✓ Stopped local server${NC}"
    else
        echo "  No local server running"
    fi
    
    # Kill remote server on mini2
    if ssh mini2@192.168.5.2 "pkill -f 'python.*server.py'" 2>/dev/null; then
        echo -e "${GREEN}✓ Stopped mini2 server${NC}"
    else
        echo "  No mini2 server running"
    fi
    
    # Kill any process on port 8100
    if lsof -ti:8100 | xargs kill -9 2>/dev/null; then
        echo -e "${GREEN}✓ Released port 8100${NC}"
    fi
    
    sleep 1
}

# Function to start servers
start_servers() {
    echo -e "${GREEN}🚀 MLX Distributed Inference with Actual Forward Passes${NC}"
    echo "======================================================"
    
    # Clear old log files
    echo "Clearing old logs..."
    rm -f server.log
    ssh mini2@192.168.5.2 "rm -f /Users/Shared/mlx_distributed_ring_inference/server.log" 2>/dev/null
    
    # Sync the server to mini2
    echo "Syncing server.py to mini2..."
    if scp server.py mini2@192.168.5.2:/Users/Shared/mlx_distributed_ring_inference/; then
        echo -e "${GREEN}✓ Synced server.py to mini2${NC}"
    else
        echo -e "${RED}✗ Failed to sync to mini2${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}Launching distributed inference...${NC}"
    echo -e "${YELLOW}📝 Logging to server.log${NC}"
    echo -e "${YELLOW}📊 Monitor with: tail -f server.log${NC}"
    echo ""
    
    # Launch with mlx.launch and redirect to log file
    cd /Users/Shared/mlx_distributed_ring_inference
    mlx.launch --hostfile hosts.json --backend ring --verbose python3 server.py >> server.log 2>&1 &
    
    # Save PID for tracking
    echo $! > .server.pid
    
    # Wait a bit and check if server started
    sleep 3
    if ps -p $(cat .server.pid) > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Distributed inference launched (PID: $(cat .server.pid))${NC}"
        echo -e "${GREEN}✓ API endpoint: http://localhost:8100${NC}"
        echo ""
        echo "Use './launch.sh status' to check server status"
        echo "Use 'tail -f server.log' to monitor logs"
    else
        echo -e "${RED}✗ Failed to start distributed inference${NC}"
        echo "Check server.log for errors"
        exit 1
    fi
}

# Function to check status
check_status() {
    echo -e "${YELLOW}📊 Checking distributed inference status...${NC}"
    echo "======================================================"
    
    # Check local server
    if pgrep -f "python.*server.py" > /dev/null; then
        echo -e "${GREEN}✓ Local server is running${NC}"
        ps aux | grep "python.*server.py" | grep -v grep | head -1
    else
        echo -e "${RED}✗ Local server is not running${NC}"
    fi
    
    echo ""
    
    # Check mini2 server
    if ssh mini2@192.168.5.2 "pgrep -f 'python.*server.py'" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Mini2 server is running${NC}"
        ssh mini2@192.168.5.2 "ps aux | grep 'python.*server.py' | grep -v grep | head -1"
    else
        echo -e "${RED}✗ Mini2 server is not running${NC}"
    fi
    
    echo ""
    
    # Check API endpoint
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API endpoint is accessible at http://localhost:8100${NC}"
    else
        echo -e "${YELLOW}⚠ API endpoint not responding${NC}"
    fi
    
    echo ""
    
    # Show log file info
    if [ -f server.log ]; then
        echo -e "${YELLOW}📝 Log file: server.log${NC}"
        echo "Last 5 log entries:"
        tail -5 server.log
    fi
}

# Main logic
case "$ACTION" in
    start)
        stop_servers
        start_servers
        ;;
    stop)
        stop_servers
        echo -e "${GREEN}✓ All servers stopped${NC}"
        ;;
    restart)
        stop_servers
        echo ""
        start_servers
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "  start   - Stop any existing servers and start new ones"
        echo "  stop    - Stop all distributed inference servers"
        echo "  restart - Restart all servers"
        echo "  status  - Check the status of distributed servers"
        echo ""
        echo "Default action is 'start' if no argument provided"
        exit 1
        ;;
esac