# 2. Hardware Inventory

### 2.1 Machine: `lambda` (Threadripper)
| Property | Value |
|----------|-------|
| CPU | AMD Threadripper (high core count) |
| GPU | NVIDIA RTX 4090 (24 GB GDDR6X) |
| GPU Backend | CUDA 12.x |
| FP32 TFLOPS | 82.6 |
| FP16 TFLOPS | 165 (with tensor cores) |
| Role | **Primary trainer, student model** |
| Measured MFU | **21.9%** (350M, seq=1024, cuBLAS SIMD, no tensor cores) |
| Measured tok/s | **7,579** (350M, seq=1024, batch=4) |

### 2.2 Machine: `intel` (Mac Pro 2019 chassis, Linux)
| Property | Value |
|----------|-------|
| CPU | Intel Xeon W-3245 @ 3.20 GHz (16C/32T) |
| RAM | **~300 GB** |
| GPU | 2x AMD Radeon Pro W5700X (8 GB GDDR6 each) |
| GPU Backend | wgpu/Vulkan (ROCm unsupported for RDNA 1 / gfx1010) |
| FP32 TFLOPS | ~9 per card (~18 total) |
| Role | **Teacher inference (Qwen3-Coder-Next in CPU RAM), data pipeline, eval** |

### 2.3 Network
- SSH connectivity (`ssh intel`) with ControlMaster multiplexing (forjar FJ-252)
- LAN bandwidth assumed ≥1 Gbps

### 2.4 Key Insight: 300 GB RAM Enables CPU-Based Teacher Inference

The intel box's 300 GB RAM fundamentally changes the distillation architecture.
Qwen3-Coder-Next (80B params) fits entirely in CPU RAM:

| Model Format | Size in RAM | Fits in 300 GB? | Headroom |
|-------------|-------------|-----------------|----------|
| fp16 | ~160 GB | Yes | ~140 GB for KV cache + buffers |
| Q8 | ~80 GB | Easily | ~220 GB |
| Q4 | ~40 GB | Trivially | ~260 GB |

No quantization-induced quality loss needed. The teacher runs at full fp16
precision, producing the highest-quality soft targets for distillation.
