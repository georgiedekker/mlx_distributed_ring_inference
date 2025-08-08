#!/usr/bin/env python3
"""
Distributed MoE Inference Server with Actual Forward Passes
This implements real distributed inference across Mac minis
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional
import uuid

import mlx.core as mx
import mlx.nn as nn
import mlx.core.distributed as dist
from dataclasses import dataclass
from transformers import AutoTokenizer

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import json

# Import local MoE model files
from shard import Shard
# from qwen_moe_mini import Model, ModelArgs  # Commented out - using MLX model
# from base import KVCache  # Commented out - using MLX model
from mlx_lm import load

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set GPU as default
mx.set_default_device(mx.gpu)

# Global state
model = None
config = None
distributed_group = None
rank = 0
world_size = 1
tokenizer = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "qwen-moe-mini"
    choices: List[Dict[str, Any]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown"""
    global rank, world_size, config
    
    if rank == 0 and config is not None and hasattr(config, 'shard'):
        start_layer = config.shard.start_layer
        end_layer = config.shard.end_layer
        
        logger.info(f"""
        ========================================
        Distributed MoE Server Ready!
        ========================================
        Rank: {rank}/{world_size}
        Model: Qwen-MoE-Mini
        Layers: {start_layer}-{end_layer}
        Distributed: {"✅ Yes" if world_size > 1 else "❌ No"}
        API: http://localhost:8100
        ========================================
        """)
    
    yield
    logger.info(f"[Rank {rank}] Shutting down...")

app = FastAPI(lifespan=lifespan)

def initialize_distributed():
    """Initialize distributed group"""
    global distributed_group, rank, world_size
    
    try:
        if dist.is_available():
            logger.info("MLX distributed is available, initializing...")
            
            # Initialize MLX distributed (auto-detects environment)
            distributed_group = dist.init()
            
            # Get actual rank and size from MLX
            rank = distributed_group.rank()
            world_size = distributed_group.size()
            
            logger.info(f"✅ MLX distributed initialized: rank {rank}/{world_size}")
            
            # Test communication if we have multiple ranks
            if world_size > 1:
                test = mx.array([float(rank)])
                result = dist.all_sum(test)
                mx.eval(result)
                
                expected_sum = sum(range(world_size))
                if abs(result.item() - expected_sum) < 0.01:
                    logger.info(f"🎉 All {world_size} devices connected and communicating!")
                else:
                    logger.warning(f"Communication test unexpected: got {result.item()}, expected {expected_sum}")
            
            return True
            
        else:
            logger.warning("MLX distributed not initialized, using single device")
            rank = 0
            world_size = 1
            distributed_group = None
        
        return True
    except Exception as e:
        logger.error(f"Distributed initialization failed: {e}")
        rank = 0
        world_size = 1
        distributed_group = None
        return False

def initialize_model():
    """Initialize the real Qwen2 MLX model with appropriate sharding"""
    global model, config, full_model
    
    # Initial synchronization barrier
    if world_size > 1:
        logger.info(f"[Rank {rank}] Synchronizing before model load...")
        sync_tensor = mx.array([1.0])
        sync_result = dist.all_sum(sync_tensor)
        mx.eval(sync_result)
        logger.info(f"[Rank {rank}] Synchronized, sum = {sync_result.item()}")
    
    logger.info(f"[Rank {rank}] Loading real Qwen2-0.5B MLX model...")
    
    # Load the full MLX Qwen model
    full_model, _ = load('Qwen/Qwen2-0.5B-Instruct-MLX')
    
    # Extract model configuration from the loaded model
    model_config = full_model.args
    logger.info(f"[Rank {rank}] Model config: {model_config}")
    
    # Calculate layer distribution for this rank
    num_layers = model_config.num_hidden_layers  # 24 layers
    if world_size > 1:
        layers_per_rank = num_layers // world_size
        start_layer = rank * layers_per_rank
        end_layer = start_layer + layers_per_rank - 1
        if rank == world_size - 1:
            end_layer = num_layers - 1
    else:
        start_layer = 0
        end_layer = num_layers - 1
    
    # Create shard configuration
    class Config:
        def __init__(self):
            self.hidden_size = model_config.hidden_size  # 896
            self.vocab_size = model_config.vocab_size    # 151936
            self.num_hidden_layers = model_config.num_hidden_layers  # 24
            self.shard = Shard(
                model_id="qwen2-0.5b-instruct-mlx",
                start_layer=start_layer,
                end_layer=end_layer,
                n_layers=num_layers
            )
    
    config = Config()
    
    logger.info(f"[Rank {rank}] Handling layers {start_layer}-{end_layer} (total: {num_layers})")
    
    # For distributed inference, we'll use the full model but only process relevant layers
    model = full_model
    
    # Check memory
    mem = mx.get_active_memory() / 1024**3
    logger.info(f"[Rank {rank}] Qwen2 model loaded, GPU memory: {mem:.2f} GB")

def initialize_tokenizer():
    """Initialize Qwen tokenizer"""
    global tokenizer
    try:
        logger.info("Loading Qwen tokenizer...")
        # Try to load Qwen2 tokenizer - this is the most common one
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
        logger.info(f"✓ Loaded Qwen tokenizer, vocab size: {tokenizer.vocab_size}")
        return True
    except Exception as e:
        logger.warning(f"Failed to load Qwen tokenizer: {e}")
        logger.info("Falling back to character-based tokenization")
        return False

def qwen_tokenize(text: str) -> mx.array:
    """Tokenize text using Qwen tokenizer"""
    global tokenizer
    try:
        if tokenizer is None:
            # Fallback to simple tokenization
            return simple_tokenize(text)
        
        # Use Qwen tokenizer
        encoded = tokenizer.encode(text, add_special_tokens=True)
        # Ensure minimum length for the model
        if len(encoded) < 10:
            encoded = encoded + [tokenizer.pad_token_id or 0] * (10 - len(encoded))
        return mx.array([encoded])  # Batch size 1
    except Exception as e:
        logger.warning(f"Qwen tokenization failed: {e}, falling back to simple")
        return simple_tokenize(text)

def qwen_detokenize(token_ids: List[int]) -> str:
    """Detokenize using Qwen tokenizer"""
    global tokenizer
    try:
        if tokenizer is None:
            return simple_detokenize(token_ids)
        
        # Filter out padding tokens and special tokens
        filtered_tokens = []
        for token_id in token_ids:
            if token_id == tokenizer.eos_token_id:
                break
            if token_id not in [tokenizer.pad_token_id, 0] and token_id < tokenizer.vocab_size:
                filtered_tokens.append(token_id)
        
        if not filtered_tokens:
            return f"[Empty response - tokens: {token_ids[:5]}...]"
        
        decoded = tokenizer.decode(filtered_tokens, skip_special_tokens=True)
        return decoded.strip() if decoded.strip() else f"[Blank - tokens: {token_ids[:5]}...]"
    except Exception as e:
        logger.warning(f"Qwen detokenization failed: {e}")
        return simple_detokenize(token_ids)

def simple_tokenize(text: str, vocab_size: int = 32000) -> mx.array:
    """Fallback character-based tokenization"""
    tokens = [ord(c) % vocab_size for c in text[:100]]
    while len(tokens) < 10:
        tokens.append(0)
    return mx.array([tokens])

def simple_detokenize(token_ids: List[int]) -> str:
    """Fallback character-based detokenization"""
    try:
        chars = []
        for token_id in token_ids:
            if token_id > 0 and token_id < 128:
                chars.append(chr(token_id))
            elif token_id == 2:
                break
            else:
                char_val = token_id % 95 + 32
                chars.append(chr(char_val))
        result = ''.join(chars)
        return result if result.strip() else f"[Generated {len(token_ids)} tokens: {token_ids[:10]}...]"
    except Exception as e:
        return f"[Token decoding error: {e}. Token IDs: {token_ids[:10]}...]"

def distributed_forward(input_ids: mx.array) -> mx.array:
    """
    Perform distributed forward pass through the model using collective operations
    Both ranks must call this function together
    """
    global model, distributed_group, rank, world_size, config
    
    logger.info(f"[Rank {rank}] Starting distributed_forward with input shape: {input_ids.shape}")
    batch_size, seq_len = input_ids.shape
    
    if world_size == 1:
        # Single device - just run the full model
        output = model(input_ids)
        return output
    
    # Distributed inference using collectives
    # Step 1: Rank 0 processes embeddings and first layers
    if rank == 0:
        logger.info(f"[Rank 0] Processing embeddings and layers {config.shard.start_layer}-{config.shard.end_layer}")
        
        # Get embeddings
        h = model.model.embed_tokens(input_ids)
        logger.info(f"[Rank 0] Embeddings shape: {h.shape}")
        
        # Process layers 0-7
        for i in range(config.shard.start_layer, config.shard.end_layer + 1):
            logger.debug(f"[Rank 0] Processing layer {i}")
            layer = model.model.layers[i]
            h = layer(h)
        
        logger.info(f"[Rank 0] Finished layers, hidden states shape: {h.shape}")
        hidden_to_share = h
    else:
        # Rank 1 creates dummy tensor to participate in all_gather
        logger.info(f"[Rank 1] Creating dummy hidden states for all_gather")
        hidden_to_share = mx.zeros((batch_size, seq_len, config.hidden_size), dtype=mx.float32)
    
    # Step 2: All-gather to share hidden states
    logger.info(f"[Rank {rank}] Participating in all_gather for hidden states")
    mx.eval(hidden_to_share)
    all_hidden = dist.all_gather(hidden_to_share, group=distributed_group)
    mx.eval(all_hidden)
    logger.info(f"[Rank {rank}] All_gather complete, received {len(all_hidden)} tensors")
    
    # Step 3: Rank 1 processes second half of layers
    if rank == 1:
        # Get hidden states from rank 0
        if isinstance(all_hidden, list):
            h = all_hidden[0]  # Get rank 0's hidden states
        else:
            # If concatenated, take first batch_size elements
            h = all_hidden[:batch_size]
        logger.info(f"[Rank 1] Processing layers {config.shard.start_layer}-{config.shard.end_layer}")
        
        # Process layers 8-15
        for i in range(config.shard.start_layer, config.shard.end_layer + 1):
            logger.debug(f"[Rank 1] Processing layer {i}")
            layer = model.model.layers[i]
            h = layer(h)
        
        # Apply final norm and lm_head
        h = model.model.norm(h)
        output = model.lm_head(h)
        logger.info(f"[Rank 1] Finished processing, output shape: {output.shape}")
        output_to_share = output
    else:
        # Rank 0 creates dummy output tensor
        logger.info(f"[Rank 0] Creating dummy output for all_gather")
        output_to_share = mx.zeros((batch_size, seq_len, config.vocab_size), dtype=mx.float32)
    
    # Step 4: All-gather to share final output
    logger.info(f"[Rank {rank}] Participating in all_gather for output")
    mx.eval(output_to_share)
    all_outputs = dist.all_gather(output_to_share, group=distributed_group)
    mx.eval(all_outputs)
    logger.info(f"[Rank {rank}] All_gather complete for outputs")
    
    # Step 5: Return the output from rank 1
    if isinstance(all_outputs, list):
        final_output = all_outputs[1]  # Get output from rank 1
    else:
        # If concatenated, slice to get rank 1's output
        final_output = all_outputs[batch_size:batch_size*2]
    return final_output

def worker_inference_participant():
    """
    Worker (rank > 0) participates in distributed inference
    Synchronizes with rank 0 to know when to start generation
    """
    global rank, world_size, distributed_group
    
    logger.info(f"[Rank {rank}] Starting worker inference participant")
    
    while True:
        try:
            # Wait for signal from rank 0 about whether to start inference
            # Use all_sum with a small tensor as a synchronization mechanism
            logger.debug(f"[Rank {rank}] Waiting for inference signal...")
            
            # Rank 0 will send 1 to start, -1 to shutdown
            signal = mx.zeros([1], dtype=mx.int32)
            signal = dist.all_sum(signal, group=distributed_group)
            mx.eval(signal)
            
            signal_value = int(signal[0].item())
            
            if signal_value == -1:
                logger.info(f"[Rank {rank}] Received shutdown signal")
                break
            elif signal_value > 0:  # Start inference
                logger.info(f"[Rank {rank}] Received inference signal")
                
                # Get input shape via all_gather
                shape_tensor = mx.zeros([2], dtype=mx.int32)
                all_shapes = dist.all_gather(shape_tensor, group=distributed_group)
                mx.eval(all_shapes)
                
                # Handle the all_gather result properly
                if isinstance(all_shapes, list):
                    shape_from_rank0 = all_shapes[0]
                else:
                    # If concatenated, take first 2 elements
                    shape_from_rank0 = all_shapes[:2]
                
                batch_size = int(shape_from_rank0[0].item())
                seq_len = int(shape_from_rank0[1].item())
                
                # Get max_tokens via all_sum
                max_tokens_tensor = mx.zeros([1], dtype=mx.int32)
                max_tokens_all = dist.all_sum(max_tokens_tensor, group=distributed_group)
                mx.eval(max_tokens_all)
                max_tokens = int(max_tokens_all[0].item())
                
                logger.info(f"[Rank {rank}] Starting generation: batch={batch_size}, seq={seq_len}, max_tokens={max_tokens}")
                
                # Create dummy input (actual input doesn't matter for rank > 0)
                dummy_input = mx.zeros((batch_size, seq_len), dtype=mx.int32)
                
                # Participate in token generation
                generated = generate_tokens(dummy_input, max_tokens, temperature=0.7)
                
                logger.info(f"[Rank {rank}] Generation complete")
            else:
                # No inference requested, wait a bit
                import time
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"[Rank {rank}] Error in worker participant: {e}")
            import traceback
            traceback.print_exc()
            break

def worker_loop_old_removed():
    """
    Simple worker loop for rank 1 - just waits for and processes requests
    """
    global model, config, distributed_group
    
    # Also log to a separate file for debugging
    with open("/tmp/rank1.log", "w") as f:
        f.write("[Rank 1] Starting worker loop\n")
        f.flush()
    
    logger.info(f"[Rank 1] Starting worker loop")
    
    while True:
        try:
            # First receive shape info
            logger.info(f"[Rank 1] Waiting for shape info...")
            shape_info = dist.recv(
                shape=(2,),
                dtype=mx.int32,
                src=0,
                group=distributed_group
            )
            mx.eval(shape_info)
            
            batch_size = shape_info[0].item()
            seq_len = shape_info[1].item()
            logger.info(f"[Rank 1] Received shape info: batch_size={batch_size}, seq_len={seq_len}")
            
            # Also log to file
            with open("/tmp/rank1.log", "a") as f:
                f.write(f"[Rank 1] Received shape: {batch_size}x{seq_len}\n")
                f.flush()
            
            # Now receive hidden states with the correct shape
            logger.info(f"[Rank 1] Waiting for hidden states...")
            h = dist.recv(
                shape=(batch_size, seq_len, config.hidden_size),
                dtype=mx.float32,
                src=0,
                group=distributed_group
            )
            mx.eval(h)
            logger.info(f"[Rank 1] Received hidden states shape: {h.shape}")
            
            # Process layers 8-15
            for i in range(config.shard.start_layer, config.shard.end_layer + 1):
                layer = model.model.layers[i]
                h = layer(h)
            
            # Apply final norm and lm_head
            h = model.model.norm(h)
            output = model.lm_head(h)
            
            logger.info(f"[Rank 1] Sending output back")
            mx.eval(output)
            dist.send(output, dst=0)
            
        except Exception as e:
            logger.error(f"[Rank 1] Error in worker loop: {e}")
            import traceback
            traceback.print_exc()
            break

def worker_inference_loop_old_removed():
    """
    Worker loop for rank > 0 to participate in distributed inference
    """
    global rank, world_size, distributed_group, config
    
    logger.info(f"[Rank {rank}] Starting worker inference loop")
    
    # Worker continuously waits for distributed_forward calls
    while True:
        try:
            # Use recv to wait for shape info from rank 0
            # This blocks until rank 0 sends something
            logger.info(f"[Rank {rank}] Waiting for shape info from rank 0...")
            
            # First receive shape info (batch_size, seq_len)
            shape_info = dist.recv(
                shape=(2,),
                dtype=mx.int32,
                src=0,
                group=distributed_group
            )
            mx.eval(shape_info)
            logger.info(f"[Rank {rank}] Received shape info: {shape_info.tolist()}")
            
            batch_size = shape_info[0].item()
            seq_len = shape_info[1].item()
            
            if batch_size == -1:  # Shutdown signal
                logger.info(f"[Rank {rank}] Received shutdown signal")
                break
                
            logger.info(f"[Rank {rank}] Received work request: batch_size={batch_size}, seq_len={seq_len}")
            
            # Now receive the hidden states
            h = dist.recv(
                shape=(batch_size, seq_len, config.hidden_size),
                dtype=mx.float32,
                src=0,
                group=distributed_group
            )
            mx.eval(h)
            logger.info(f"[Rank {rank}] Received hidden states shape: {h.shape}")
            
            # Process layers 8-15
            for i in range(config.shard.start_layer, config.shard.end_layer + 1):
                layer = model.model.layers[i]
                h = layer(h)
            
            # Apply final norm and lm_head
            h = model.model.norm(h)
            output = model.lm_head(h)
            
            logger.info(f"[Rank {rank}] Sending output shape: {output.shape} back to rank 0")
            mx.eval(output)
            dist.send(output, dst=0)
            logger.info(f"[Rank {rank}] Sent output, waiting for next request")
            
        except Exception as e:
            logger.error(f"[Rank {rank}] Worker error: {e}")
            import traceback
            traceback.print_exc()
            break

def generate_tokens(input_ids: mx.array, max_tokens: int, temperature: float = 0.7) -> List[int]:
    """
    Generate tokens using the distributed model
    Both ranks must call this function together
    """
    global rank, world_size, distributed_group
    
    logger.info(f"[Rank {rank}] Starting generate_tokens with max_tokens={max_tokens}")
    generated = []
    current_input = input_ids
    
    for step in range(max_tokens):
        logger.debug(f"[Rank {rank}] Generation step {step+1}/{max_tokens}")
        
        # Both ranks call distributed_forward
        logits = distributed_forward(current_input)
        
        # Rank 0 samples the next token
        if rank == 0:
            # Sample next token
            if temperature > 0:
                # Sample with temperature
                probs = mx.softmax(logits[0, -1, :] / temperature)
                next_token = mx.random.categorical(mx.log(probs))
            else:
                # Greedy sampling
                next_token = mx.argmax(logits[0, -1, :])
            
            next_token_id = int(next_token.item())
            next_token_array = mx.array([next_token_id], dtype=mx.int32)
        else:
            # Rank 1 creates dummy token for all_gather
            next_token_array = mx.zeros([1], dtype=mx.int32)
        
        # Use all_gather to broadcast the next token to all ranks
        logger.debug(f"[Rank {rank}] Broadcasting next token via all_gather")
        mx.eval(next_token_array)
        all_tokens = dist.all_gather(next_token_array, group=distributed_group)
        mx.eval(all_tokens)
        
        # Extract the token from rank 0
        if isinstance(all_tokens, list):
            next_token_id = int(all_tokens[0].item())
        else:
            # If concatenated, take the first element (from rank 0)
            next_token_id = int(all_tokens[0].item())
        generated.append(next_token_id)
        
        logger.debug(f"[Rank {rank}] Next token: {next_token_id}")
        
        # Check for EOS token (simplified - use 2 as EOS)
        if next_token_id == 2:
            logger.info(f"[Rank {rank}] EOS token detected, stopping generation")
            break
        
        # Both ranks update their input for next iteration
        next_token_arr = mx.array([[next_token_id]])
        current_input = mx.concatenate([current_input, next_token_arr], axis=1)
    
    logger.info(f"[Rank {rank}] Generated {len(generated)} tokens")
    return generated

@app.get("/")
async def root():
    """Health check and status"""
    mem = mx.get_active_memory() / 1024**3
    return {
        "status": "ready",
        "model": "qwen-moe-mini",
        "rank": f"{rank}/{world_size}",
        "layers": f"{config.shard.start_layer}-{config.shard.end_layer}" if config else "N/A",
        "gpu_memory_gb": round(mem, 2),
        "distributed": world_size > 1
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat endpoint with actual model inference"""
    
    if rank != 0:
        return JSONResponse({"error": "This is a worker node"}, status_code=400)
    
    try:
        # Extract the last message
        prompt = request.messages[-1].content if request.messages else "Hello"
        
        logger.info(f"[Rank 0] Received API request: {prompt[:50]}...")
        logger.info(f"[Rank 0] World size: {world_size}, distributed: {world_size > 1}")
        start_time = time.time()
        
        # Tokenize input using Qwen tokenizer
        input_ids = qwen_tokenize(prompt)
        logger.info(f"[Rank 0] Tokenized input shape: {input_ids.shape}")
        
        # Coordinate with other ranks for distributed generation
        logger.info(f"[Rank 0] Starting coordinated token generation...")
        
        # Signal all ranks to start inference
        signal = mx.array([1], dtype=mx.int32)  # 1 = start inference
        signal = dist.all_sum(signal, group=distributed_group)
        mx.eval(signal)
        
        # Share input shape with all ranks
        batch_size, seq_len = input_ids.shape
        shape_tensor = mx.array([batch_size, seq_len], dtype=mx.int32)
        all_shapes = dist.all_gather(shape_tensor, group=distributed_group)
        mx.eval(all_shapes)
        
        # Share max_tokens with all ranks
        max_tokens = min(request.max_tokens, 50)  # Limit for testing
        max_tokens_tensor = mx.array([max_tokens], dtype=mx.int32)
        max_tokens_all = dist.all_sum(max_tokens_tensor, group=distributed_group)
        mx.eval(max_tokens_all)
        
        # Now all ranks call generate_tokens together
        generated_tokens = generate_tokens(
            input_ids,
            max_tokens=max_tokens,
            temperature=request.temperature
        )
        logger.info(f"[Rank 0] Generated {len(generated_tokens)} tokens")
        
        elapsed = time.time() - start_time
        tokens_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0
        
        logger.info(f"Generated {len(generated_tokens)} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        # Convert tokens to text using Qwen tokenizer
        decoded_text = qwen_detokenize(generated_tokens)
        
        # If decoding fails or produces gibberish, provide debug info
        if len(decoded_text.strip()) < 3 or "Generated" in decoded_text or "Token" in decoded_text:
            response_text = f"[Distributed MoE Response] Tokens: {generated_tokens} -> \"{decoded_text}\""
        else:
            response_text = decoded_text
        
        # Format response
        response = ChatResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model="qwen-moe-mini",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Initialize distributed first to get rank
    initialize_distributed()
    
    # All ranks initialize their model portion
    initialize_model()
    
    # Initialize tokenizer (only needs to happen once, but both ranks can do it)
    initialize_tokenizer()
    
    # Only rank 0 runs the API server
    if rank == 0:
        logger.info("Starting API server on rank 0...")
        import asyncio
        uv_config = uvicorn.Config(app=app, host="0.0.0.0", port=8100, log_level="info")
        server = uvicorn.Server(uv_config)
        asyncio.run(server.serve())
    else:
        logger.info(f"Rank {rank} running as worker, waiting for inference requests...")
        # Rank 1 participates in inference when triggered by rank 0
        worker_inference_participant()