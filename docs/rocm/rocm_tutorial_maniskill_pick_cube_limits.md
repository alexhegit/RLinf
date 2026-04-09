# Root Cause Analysis: ManiSkill/SAPIEN ROCm Incompatibility

## Overview

This document explains why the `maniskill_sac_mlp` example cannot utilize AMD GPUs for physics simulation, despite PyTorch and the SAC neural network training working correctly on ROCm-enabled AMD hardware.

## Software Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     ManiSkill (Application Layer)                │
│  - Task definitions (PickCube, PushCube, etc.)                   │
│  - RL environment wrapper (Gymnasium interface)                  │
│  - Parallel environment management                               │
│  - Robot embodiments and sensor configurations                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SAPIEN (Simulation Layer)                    │
│  - Scene management (Entity-Component System)                    │
│  - Rendering (Vulkan, ray-tracing)                               │
│  - Camera systems and visual sensors                             │
│  - Bridge to physics engines                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              NVIDIA PhysX 5 (Physics Engine Layer)               │
│  - GPU-accelerated rigid body dynamics                           │
│  - Articulation and joint systems                                │
│  - Collision detection and response                              │
│  - CUDA kernels (proprietary NVIDIA implementation)              │
└─────────────────────────────────────────────────────────────────┘
```

### Component Relationships

| Component | Role | GPU Support |
|-----------|------|-------------|
| **ManiSkill** | High-level RL framework | Agnostic (depends on backend) |
| **SAPIEN** | Simulation middleware | Vulkan rendering works on AMD |
| **PhysX 5** | Physics computation | **CUDA-only** |

## The Core Problem

### Why PyTorch Works but PhysX Doesn't

#### PyTorch on ROCm (Works)

```
PyTorch (Python) 
    │
    ▼
torch.cuda.* Python API
    │
    ▼
ROCm/HIP Backend (Intercepted at Python binding layer)
    │
    ▼
AMD GPU (gfx1100)
```

- PyTorch's CUDA API calls go through a Python-to-C++ binding layer
- ROCm provides a compatibility layer that intercepts these calls
- PyTorch tensors and operations are translated to HIP at the framework level

#### PhysX 5 on ROCm (Fails)

```
PhysX 5 (C++ Binary)
    │
    ▼
CUDA Driver API (Direct binary calls in compiled .so/.dll)
    │
    ▼
NVIDIA GPU Only
```

- PhysX 5 is shipped as pre-compiled binaries with embedded CUDA machine code
- CUDA kernel launches are direct calls to `libcuda.so` driver APIs
- **ROCm cannot intercept these low-level binary calls**
- No source code available to recompile for HIP/ROCm

### Technical Comparison

| Aspect | PyTorch | PhysX 5 |
|--------|---------|---------|
| **Source Availability** | Open source | Proprietary (binary only) |
| **GPU Abstraction** | High-level API | Direct CUDA kernel calls |
| **ROCm Intercept Point** | Python/C++ binding layer | None (already compiled) |
| **Recompilation Possible** | Yes (via ROCm PyTorch builds) | No (NVIDIA closed source) |

## Error Manifestation

### What Happens When You Try GPU Backend

```python
# maniskill_pick_cube.yaml
init_params:
  sim_backend: "gpu"  # Tries to use CUDA
```

**Error Output:**
```
RuntimeError: failed to find device "cuda"
```

**Root Cause Path:**

1. SAPIEN calls `sapien.Device("cuda:0")`
2. PhysX 5 internally calls `cudaSetDevice(0)`
3. ROCm environment has no real CUDA driver
4. PhysX cannot find CUDA device → Error

### Why CPU Backend Works

```python
init_params:
  sim_backend: "cpu"  # Uses CPU physics
```

- PhysX CPU backend uses standard C++ code (no CUDA)
- Runs on x86_64 CPU cores
- Compatible with any hardware
- **Limitation**: Single environment only (no parallelization)

## Why ROCm's CUDA Compatibility Layer Fails Here

### ROCm's Design Limitations

ROCm provides CUDA compatibility at the **application framework level**, not the **driver binary level**:

```
┌─────────────────────────────────────────────────────────────────┐
│  ROCm Compatibility Scope                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ✓ Can Intercept (Framework Level)                               │
│    - PyTorch operations                                          │
│    - TensorFlow operations                                       │
│    - JAX operations (via XLA)                                    │
│    - High-level CUDA API calls from open-source libraries        │
│                                                                   │
│  ✗ Cannot Intercept (Binary Level)                               │
│    - Pre-compiled CUDA kernels in .so files                      │
│    - Direct cudaDriver API calls from closed-source binaries     │
│    - Inline PTX/SASS assembly in compiled code                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### The PhysX Binary Problem

PhysX 5 GPU implementation contains:
- Machine code with embedded CUDA PTX
- Direct `cuLaunchKernel` calls
- NVIDIA-specific memory management

These are **already compiled** and **cannot be translated** by ROCm at runtime.

## Potential Solutions

### 1. NVIDIA Adds ROCm Support (Low Probability)

- NVIDIA would need to compile PhysX 5 for HIP/ROCm
- Business incentive is low (sells more NVIDIA GPUs)
- Timeline: Unknown/unlikely

### 2. Use Alternative Physics Engines

| Engine | ROCm Support | Notes |
|--------|--------------|-------|
| **IsaacLab (Omniverse)** | ✅ ROCm 6.2+ | USD-based, PhysX 5 with abstraction layer |
| **MuJoCo + MJX** | ✅ Via JAX | JAX has ROCm backend via OpenXLA |
| **Brax** | ✅ Pure JAX | Fully differentiable, JAX-based |
| **SAPIEN CPU** | ✅ Always works | Slow, single-threaded physics |

### 3. Community Effort (Very Difficult)

- Rewriting PhysX 5 GPU kernels in HIP would require:
  - Reverse engineering NVIDIA's proprietary physics algorithms
  - Years of engineering effort
  - Ongoing maintenance as PhysX evolves
- **Not realistically feasible**

## Current Workaround

### What We Did

Modified `examples/embodiment/config/env/maniskill_pick_cube.yaml`:

```yaml
init_params:
  sim_backend: "cpu"  # Changed from "gpu"
```

### Trade-offs

| Aspect | GPU Backend (CUDA) | CPU Backend (Current) |
|--------|-------------------|----------------------|
| **Physics Device** | NVIDIA GPU | CPU |
| **Parallel Envs** | 1000s | 1 only |
| **Simulation Speed** | 30,000+ FPS | ~100-500 FPS |
| **NN Training** | GPU (fast) | GPU (fast) |
| **Data Transfer** | GPU→GPU (fast) | CPU→GPU (bottleneck) |

### Performance Impact

- **Neural network training**: Still on AMD GPU (fast)
- **Physics simulation**: CPU-bound (slow)
- **Overall throughput**: Limited by CPU physics, not GPU compute

## CPU Multi-Environment Limitation

### Why CPU Cannot Run Multiple Parallel Environments

While CPU **theoretically** supports multi-environment training, ManiSkill's current implementation imposes a **single-environment restriction** when using `sim_backend="cpu"`.

#### GPU Backend: Single-Process Multi-Env (Native Batch API)

```
┌─────────────────────────────────────────┐
│  Single Process                         │
│  ┌─────────┬─────────┬─────────┐       │
│  │ Env 0   │ Env 1   │ Env 2   │ ...   │  ← Parallel on GPU
│  │ (CUDA)  │ (CUDA)  │ (CUDA)  │       │
│  └─────────┴─────────┴─────────┘       │
│         ↓ Parallel computation           │
│         Shared GPU memory                │
└─────────────────────────────────────────┘

Configuration: sim_backend="gpu", num_envs=1000  ✅ Supported
Throughput: 30,000+ FPS
```

**How it works:**
- PhysX GPU provides **native batch API** for parallel environments
- All 1000 environments run in **single CUDA context**
- GPU kernels process all environments in parallel
- Zero-copy data transfer between environments

#### CPU Backend: Multi-Process Required (Not Implemented)

```
┌─────────────────────────────────────────┐
│  AsyncVectorEnv (Multi-Process)         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Process │ │ Process │ │ Process │   │
│  │  Env 0  │ │  Env 1  │ │  Env 2  │   │  ← Independent Python processes
│  │  (CPU)  │ │  (CPU)  │ │  (CPU)  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘   │
│       └───────────┼───────────┘         │
│                   ↓ IPC overhead         │
│              Pipe/Queue communication    │
└─────────────────────────────────────────┘

Configuration: sim_backend="cpu", num_envs=4  ❌ Not Supported by ManiSkill
Workaround: Requires AsyncVectorEnv wrapper (not integrated)
```

**How it SHOULD work (but doesn't):**
- Each environment runs in **separate Python process**
- Processes communicate via **IPC (Inter-Process Communication)**
- Gymnasium provides `AsyncVectorEnv` for this purpose
- Significant overhead from process spawning and IPC

### Why ManiSkill Doesn't Support CPU Multi-Env

| Aspect | GPU Backend | CPU Backend |
|--------|-------------|-------------|
| **Implementation** | Native batch API in PhysX | Not implemented in ManiSkill |
| **API Design** | `ManiSkillEnv(num_envs=N)` | Same API, but N>1 throws error |
| **Parallelization** | Single process, GPU kernels | Requires multi-process wrapper |
| **RLinf Integration** | Direct integration | Requires architecture changes |

**Technical Reasons:**

1. **SAPIEN Design Decision**: CPU backend was designed for single-environment debugging, not production training
2. **Missing AsyncVectorEnv Integration**: ManiSkill doesn't automatically wrap CPU environments
3. **Ray Worker Assumptions**: RLinf's Ray-based workers assume environments can be serialized and distributed

### Attempting Multi-Env on CPU (Error)

```python
# maniskill_pick_cube.yaml
env:
  train:
    total_num_envs: 4  # Try to use 4 environments
    init_params:
      sim_backend: "cpu"
```

**Result:**
```
RuntimeError: Cannot set the sim backend to 'cpu' and have multiple environments.
            If you want to do CPU sim backends and have environment vectorization 
            you must use multi-processing across CPUs.
            This can be done via the gymnasium's AsyncVectorEnv API
```

### Theoretical Workaround (Not Integrated with RLinf)

```python
# Manual AsyncVectorEnv (works standalone, NOT with RLinf)
from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym

def make_env():
    return gym.make("PickCube-v1", sim_backend="cpu")

# Create 4 parallel CPU environments
envs = AsyncVectorEnv([make_env] * 4)

# This works for Gymnasium, but RLinf expects a different API
```

**Why this doesn't work with RLinf:**
- RLinf's `ManiSkillEnv` wrapper expects to manage environments internally
- Ray workers serialize environment configs, not environment objects
- `AsyncVectorEnv` requires process fork/spawn which conflicts with Ray's architecture

### Performance Comparison

| Metric | GPU Backend | CPU Backend (Single) | CPU (Theoretical Multi) |
|--------|-------------|---------------------|------------------------|
| **Environments** | 1000 | 1 | 4-16 (limited by cores) |
| **Throughput** | 30,000 FPS | ~100 FPS | ~400-800 FPS (estimate) |
| **Memory** | GPU VRAM | System RAM | High RAM + IPC overhead |
| **CPU Usage** | Low | 1 core | 100% (all cores) |
| **Training Time** | Hours | Days | Days (still slow) |

### Bottom Line

**CPU multi-environment training is theoretically possible but:**
1. **Not implemented** in ManiSkill/SAPIEN
2. **Requires significant engineering** to integrate with RLinf
3. **Performance still poor** compared to GPU (even with 16 CPU envs)
4. **Not a practical solution** for production RL training

**Recommendation**: For any serious training, use NVIDIA GPU or alternative simulators (IsaacLab, MJX) that support ROCm.

## Recommendations

### For AMD GPU Users

1. **Short-term**: Use CPU backend for development/testing
2. **Medium-term**: Migrate to IsaacLab (NVIDIA Omniverse has ROCm support)
3. **Long-term**: Wait for NVIDIA PhysX ROCm support (if ever)

### For Production RL Training

- **State-based RL**: CPU backend may be acceptable for small-scale experiments
- **Vision-based RL**: Impossible without GPU rendering (disabled in CPU mode)
- **Large-scale training**: Requires NVIDIA GPUs or alternative simulators

## References

- SAPIEN GitHub: https://github.com/haosulab/SAPIEN
- ManiSkill GitHub: https://github.com/haosulab/ManiSkill
- PhysX Documentation: https://developer.nvidia.com/physx-sdk
- ROCm GPU Support: https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html
- Related Issue: https://github.com/haosulab/SAPIEN/issues/209

## Conclusion

The inability to use AMD GPUs for ManiSkill physics simulation is **not a bug in ManiSkill or SAPIEN**, but rather an **architecture limitation** stemming from NVIDIA PhysX 5's proprietary CUDA-only GPU implementation. ROCm's CUDA compatibility layer operates at the framework level and cannot intercept pre-compiled binary CUDA calls from closed-source physics engines.

The only viable paths forward are:
1. NVIDIA adding ROCm support to PhysX (unlikely)
2. Migrating to alternative simulators with ROCm support (IsaacLab, MJX)
3. Using NVIDIA hardware for full GPU-accelerated simulation

---

**Document Version:** 1.0  
**Last Updated:** 2026-04-09  
**Related Tutorial:** `rocm_tutorial_maniskill_pick_cube.md`
