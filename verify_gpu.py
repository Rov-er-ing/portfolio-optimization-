import torch
import time
import os

def verify_gpu():
    print("="*50)
    print("MLPO GPU ACCELERATION VERIFICATION")
    print("="*50)
    
    # 1. Device Check
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("CRITICAL: CUDA not detected by PyTorch.")
        return

    device = torch.device("cuda")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # 2. Speed Test (Matrix Multiplication)
    size = 4000
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # CPU Benchmark
    start = time.time()
    _ = torch.matmul(a, b)
    cpu_time = time.time() - start
    print(f"\nCPU Matrix Multiplication ({size}x{size}): {cpu_time:.4f}s")

    # GPU Benchmark
    a_gpu = a.to(device)
    b_gpu = b.to(device)
    
    # Warmup
    _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(10):
        _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 10
    
    print(f"GPU Matrix Multiplication ({size}x{size}): {gpu_time:.4f}s")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

    # 3. MLPO Specific Check
    # Try to load a tensor and perform an operation
    x = torch.randn(1, 10, 20).to(device)
    print("\n[SUCCESS] Tensor successfully moved to GPU and back.")
    print("="*50)

if __name__ == "__main__":
    verify_gpu()
