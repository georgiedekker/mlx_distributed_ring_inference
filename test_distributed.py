#!/usr/bin/env python3
"""
Simple test to verify MLX distributed communication
"""
import mlx.core as mx
import mlx.core.distributed as dist

def main():
    # Initialize distributed
    group = dist.init()
    rank = group.rank()
    size = group.size()
    
    print(f"🎯 Rank {rank}/{size} initialized successfully!")
    
    if size > 1:
        # Test communication with all_sum
        local_value = mx.array([float(rank)])
        print(f"Rank {rank}: Contributing value {rank} to all_sum")
        
        result = dist.all_sum(local_value, group=group)
        mx.eval(result)
        
        expected = sum(range(size))
        print(f"Rank {rank}: all_sum result = {result.item()} (expected {expected})")
        
        if abs(result.item() - expected) < 0.01:
            print(f"✅ Rank {rank}: Communication successful!")
        else:
            print(f"❌ Rank {rank}: Communication failed!")
    else:
        print(f"⚠️ Single device mode (rank {rank})")

if __name__ == "__main__":
    main()