#!/bin/bash

# This script manages the lifecycle of distributed inference servers for
# the DeepSeek model. It can start, stop, restart and report the
# status of the server processes.  When starting, it synchronises
# ``server.py`` to the remote machine defined in ``hosts.json`` and
# launches the server via ``mlx.launch`` using the ring backend. The
# API will be available on port 8100 of the local machine.

# Colour codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default action if no argument is provided
ACTION=${1:-start}

# Detected devices
DETECTED_DEVICES=()

detect_devices() {
    echo -e "${YELLOW}🔍 Detecting Thunderbolt devices...${NC}"
    DETECTED_DEVICES=("mini1")  # Always include local machine
    
    # Check mini2 - try SSH first, then ping
    if ssh -o ConnectTimeout=2 -o BatchMode=yes mini2@192.168.5.2 "exit" 2>/dev/null || ping -c 1 -W 1 192.168.5.2 > /dev/null 2>&1; then
        DETECTED_DEVICES+=("mini2")
        echo -e "${GREEN}✓ Found mini2 (192.168.5.2)${NC}"
    else
        echo "  mini2 not connected"
    fi
    
    # Check m4 - try SSH first, then ping
    if ssh -o ConnectTimeout=2 -o BatchMode=yes georgedekker@192.168.5.3 "exit" 2>/dev/null || ping -c 1 -W 1 192.168.5.3 > /dev/null 2>&1; then
        DETECTED_DEVICES+=("m4")
        echo -e "${GREEN}✓ Found m4 (192.168.5.3)${NC}"
    else
        echo "  m4 not connected"
    fi
    
    echo -e "${GREEN}✓ Detected ${#DETECTED_DEVICES[@]} devices: ${DETECTED_DEVICES[*]}${NC}"
    
    # Generate hosts.json based on detected devices
    generate_hosts_json
}

generate_hosts_json() {
    echo -e "${YELLOW}📝 Generating hosts.json for detected devices...${NC}"
    
    # Start the JSON array
    echo "[" > hosts.json
    
    # Always add mini1 (localhost)
    echo '    {"ssh": "localhost", "ips": ["192.168.5.1"]}' >> hosts.json
    
    # Add mini2 if detected
    if [[ " ${DETECTED_DEVICES[@]} " =~ " mini2 " ]]; then
        echo ',    {"ssh": "mini2@192.168.5.2", "ips": ["192.168.5.2"]}' >> hosts.json
    fi
    
    # Add m4 if detected
    if [[ " ${DETECTED_DEVICES[@]} " =~ " m4 " ]]; then
        echo ',    {"ssh": "georgedekker@192.168.5.3", "ips": ["192.168.5.3"]}' >> hosts.json
    fi
    
    # Close the JSON array
    echo "]" >> hosts.json
    
    echo -e "${GREEN}✓ Generated hosts.json with ${#DETECTED_DEVICES[@]} nodes${NC}"
}

stop_servers() {
    echo -e "${YELLOW}⏹️  Stopping distributed inference servers...${NC}"
    
    # Detect devices first if not already done
    if [ ${#DETECTED_DEVICES[@]} -eq 0 ]; then
        detect_devices > /dev/null 2>&1
    fi
    
    # Kill main process using PID file if present
    if [ -f .server.pid ]; then
        if kill $(cat .server.pid) 2>/dev/null; then
            echo -e "${GREEN}✓ Stopped main process (PID: $(cat .server.pid))${NC}"
        fi
        rm -f .server.pid
    fi
    
    # Kill any running server instances locally
    if pkill -f "python.*server.*\.py" 2>/dev/null; then
        echo -e "${GREEN}✓ Stopped local server${NC}"
    else
        echo "  No local server running"
    fi
    
    # Kill remote server on mini2 if connected
    if [[ " ${DETECTED_DEVICES[@]} " =~ " mini2 " ]]; then
        if ssh mini2@192.168.5.2 "pkill -f 'python.*server.*\.py'" 2>/dev/null; then
            echo -e "${GREEN}✓ Stopped mini2 server${NC}"
        else
            echo "  No mini2 server running"
        fi
    fi
    
    # Kill remote server on m4 if connected
    if [[ " ${DETECTED_DEVICES[@]} " =~ " m4 " ]]; then
        if ssh georgedekker@192.168.5.3 "pkill -f 'python.*server.*\.py'" 2>/dev/null; then
            echo -e "${GREEN}✓ Stopped m4 server${NC}"
        else
            echo "  No m4 server running"
        fi
    fi
    
    # Kill API process using PID file if present
    if [ -f .api.pid ]; then
        if kill $(cat .api.pid) 2>/dev/null; then
            echo -e "${GREEN}✓ Stopped API server (PID: $(cat .api.pid))${NC}"
        fi
        rm -f .api.pid
    fi
    
    # Kill any remaining API processes
    if pkill -f "python.*api\.py" 2>/dev/null; then
        echo -e "${GREEN}✓ Stopped remaining API processes${NC}"
    else
        echo "  No API server running"
    fi
    
    # Free port 8100 if still held
    if lsof -ti:8100 | xargs kill -9 2>/dev/null; then
        echo -e "${GREEN}✓ Released port 8100${NC}"
    fi
    sleep 1
}

start_servers() {
    echo -e "${GREEN}🚀 MLX Distributed Inference with DeepSeek${NC}"
    echo "======================================================"
    
    # Detect connected devices
    detect_devices
    echo ""
    
    # Remove old log files
    echo "Clearing old logs..."
    rm -f server.log api.log
    
    # Sync server.py to connected devices
    if [[ " ${DETECTED_DEVICES[@]} " =~ " mini2 " ]]; then
        ssh mini2@192.168.5.2 "rm -f /Users/Shared/mlx_distributed_ring_inference/server.log" 2>/dev/null
        echo "Syncing server.py to mini2..."
        if scp server.py mini2@192.168.5.2:/Users/Shared/mlx_distributed_ring_inference/; then
            echo -e "${GREEN}✓ Synced server.py to mini2${NC}"
        else
            echo -e "${RED}✗ Failed to sync to mini2${NC}"
            exit 1
        fi
    fi
    
    if [[ " ${DETECTED_DEVICES[@]} " =~ " m4 " ]]; then
        ssh georgedekker@192.168.5.3 "rm -f /Users/Shared/mlx_distributed_ring_inference/server.log" 2>/dev/null
        echo "Syncing server.py to m4..."
        if scp server.py georgedekker@192.168.5.3:/Users/Shared/mlx_distributed_ring_inference/; then
            echo -e "${GREEN}✓ Synced server.py to m4${NC}"
        else
            echo -e "${RED}✗ Failed to sync to m4${NC}"
            exit 1
        fi
    fi
    echo ""
    echo -e "${GREEN}Launching distributed inference...${NC}"
    echo -e "${YELLOW}📝 Logging to server.log${NC}"
    echo -e "${YELLOW}📊 Monitor with: tail -f server.log${NC}"
    echo ""
    # Change to working directory on local machine
    cd /Users/Shared/mlx_distributed_ring_inference
    # Launch the distributed server.  The --verbose flag prints MPI
    # communication diagnostics; remove it for quieter output.
    mlx.launch --hostfile hosts.json --backend ring --verbose python3 server.py >> server.log 2>&1 &
    # Save the PID so we can stop the server later
    echo $! > .server.pid
    # Check if the server started successfully
    sleep 3
    if ps -p $(cat .server.pid) > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Distributed inference launched (PID: $(cat .server.pid))${NC}"
        
        # Start the API server
        echo -e "${YELLOW}🌐 Starting API server...${NC}"
        python3 api.py >> api.log 2>&1 &
        echo $! > .api.pid
        
        # Wait for API to be ready
        sleep 2
        if ps -p $(cat .api.pid) > /dev/null 2>&1; then
            echo -e "${GREEN}✓ API server launched (PID: $(cat .api.pid))${NC}"
            echo -e "${GREEN}✓ API endpoint: http://localhost:8100${NC}"
        else
            echo -e "${RED}✗ Failed to start API server${NC}"
            echo "Check api.log for errors"
        fi
        
        echo ""
        echo "Use './launch.sh status' to check server status"
        echo "Use 'tail -f server.log' to monitor server logs"
        echo "Use 'tail -f api.log' to monitor API logs"
    else
        echo -e "${RED}✗ Failed to start distributed inference${NC}"
        echo "Check server.log for errors"
        exit 1
    fi
}

check_status() {
    echo -e "${YELLOW}📊 Checking distributed inference status...${NC}"
    echo "======================================================"
    
    # Detect devices first
    detect_devices
    echo ""
    
    # Check local server
    if pgrep -f "python.*server.*\.py" > /dev/null; then
        echo -e "${GREEN}✓ Local server is running${NC}"
        ps aux | grep "python.*server.*\.py" | grep -v grep | head -1
    else
        echo -e "${RED}✗ Local server is not running${NC}"
    fi
    echo ""
    
    # Check API server
    if pgrep -f "python.*api\.py" > /dev/null; then
        echo -e "${GREEN}✓ API server is running${NC}"
        ps aux | grep "python.*api\.py" | grep -v grep | head -1
    else
        echo -e "${RED}✗ API server is not running${NC}"
    fi
    echo ""
    
    # Check mini2 server if connected
    if [[ " ${DETECTED_DEVICES[@]} " =~ " mini2 " ]]; then
        if ssh mini2@192.168.5.2 "pgrep -f 'python.*server.*\.py'" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Mini2 server is running${NC}"
            ssh mini2@192.168.5.2 "ps aux | grep 'python.*server.*\.py' | grep -v grep | head -1"
        else
            echo -e "${RED}✗ Mini2 server is not running${NC}"
        fi
        echo ""
    fi
    
    # Check m4 server if connected
    if [[ " ${DETECTED_DEVICES[@]} " =~ " m4 " ]]; then
        if ssh georgedekker@192.168.5.3 "pgrep -f 'python.*server.*\.py'" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ M4 server is running${NC}"
            ssh georgedekker@192.168.5.3 "ps aux | grep 'python.*server.*\.py' | grep -v grep | head -1"
        else
            echo -e "${RED}✗ M4 server is not running${NC}"
        fi
    fi
    echo ""
    # Check the health endpoint
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API endpoint is accessible at http://localhost:8100${NC}"
    else
        echo -e "${YELLOW}⚠ API endpoint not responding${NC}"
    fi
    echo ""
    # Show information about the log file
    if [ -f server.log ]; then
        echo -e "${YELLOW}📝 Log file: server.log${NC}"
        echo "Last 5 log entries:"
        tail -5 server.log
    fi
}

# Main dispatcher
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
    detect)
        detect_devices
        echo ""
        echo "Current hosts.json:"
        cat hosts.json
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|detect}"
        echo ""
        echo "  start   - Stop any existing servers and start new ones"
        echo "  stop    - Stop all distributed inference servers"
        echo "  restart - Restart all servers"
        echo "  status  - Check the status of distributed servers"
        echo "  detect  - Detect connected Thunderbolt devices and update hosts.json"
        echo ""
        echo "Default action is 'start' if no argument provided"
        exit 1
        ;;
esac