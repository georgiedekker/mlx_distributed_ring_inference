#!/bin/bash

echo "🚀 MLX Distributed with Identical Paths"
echo "========================================"

# Kill any existing servers
pkill -f server.py 2>/dev/null
ssh mini2@192.168.5.2 "pkill -f server.py" 2>/dev/null

# Create hosts file with identical paths
cat > /Users/Shared/mlx_distributed_ring_inference/hosts.json << 'EOF'
[
    {"ssh": "localhost", "ips": ["192.168.5.1"]},
    {"ssh": "mini2@192.168.5.2", "ips": ["192.168.5.2"]}
]
EOF

# Copy hosts file to mini2 (same path)
scp -q /Users/Shared/mlx_distributed_ring_inference/hosts.json mini2@192.168.5.2:/Users/Shared/mlx_distributed_ring_inference/

echo "Launching unified distributed job from identical path..."
echo ""

# Change to the shared directory (same on both machines)
cd /Users/Shared/mlx_distributed_ring_inference

# Launch with mlx.launch - now the paths are identical
mlx.launch --hostfile hosts.json --backend ring --verbose python3 server.py