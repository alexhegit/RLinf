ROCm ManiSkill SAC-MLP
======================

This document shows how to run ``examples/embodiment/config/maniskill_sac_mlp.yaml`` on AMD GPUs with ROCm.

The key constraint is that ManiSkill / SAPIEN physics still depends on CUDA for the GPU backend, so on ROCm the recommended workaround is to keep the repo's tracked config unchanged and override the physics backend to CPU at launch time.

This page covers three validated setups:

1. **Radeon PRO W7900** — single-container, works with Hydra overrides only.
2. **MI300X single-container** — requires llvmpipe Vulkan ICD isolation to prevent SAPIEN from probing the amdgpu driver.
3. **MI300X split deployment** — two-container fallback when the single-container workaround is unavailable.

Common Setup
------------

Docker image
~~~~~~~~~~~~

All three setups use the same base image:

.. code-block:: bash

   rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1

Software versions
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Version
   * - ROCm
     - 7.2.1
   * - PyTorch
     - 2.9.1+rocm7.2.1
   * - Python
     - 3.10
   * - RLinf
     - 0.2.0+

Why keep the repo config unchanged
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared environment config ``examples/embodiment/config/env/maniskill_pick_cube.yaml`` should stay at its default GPU backend so standard CUDA workflows continue to work.

For ROCm, use a Hydra runtime override instead of editing the tracked YAML:

.. code-block:: bash

   env.train.init_params.sim_backend=cpu \
   env.eval.init_params.sim_backend=cpu

The default config remains:

.. code-block:: yaml

   init_params:
     id: "PickCube-v1"
     num_envs: null
     obs_mode: "state"
     control_mode: None
     sim_backend: "gpu"
     sim_config:
       sim_freq: 100
       control_freq: 20

Install dependencies
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # In the container
   cd /workspace/RLinf
   pip install -q -e . --no-deps
   pip install -q 'setuptools<75.9'
   pip install -q ray[default]>=2.47.0 hydra-core numpy datasets torchdata scipy \
       accelerate debugpy einops nvitop pybind11 ninja pytest gsutil ruff \
       pre-commit uv huggingface_hub icmplib 'wandb<0.25.1' \
       'swanlab>=0.6.11' tensorboard transformers==4.57.6 peft timm \
       imageio[ffmpeg] gymnasium gym easydict cloudpickle draccus rich \
       mani-skill sapien==3.0.1

.. note::

   Avoid ``pip install -e .`` without ``--no-deps`` in this container. In the tested setup that replaced the image's ROCm PyTorch stack with a non-ROCm torch build, which then broke ``torchvision`` imports.

Verify GPU visibility
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -c "
   import torch
   print(f'PyTorch: {torch.__version__}')
   print(f'ROCm available: {torch.cuda.is_available()}')
   print(f'Device: {torch.cuda.get_device_name(0)}')
   "

Expected output on W7900:

::

   PyTorch: 2.9.1+rocm7.2.1
   ROCm available: True
   Device: AMD Radeon PRO W7900

Expected output on MI300X:

::

   PyTorch: 2.9.1+rocm7.2.1
   ROCm available: True
   Device: AMD Instinct MI300X VF

---

Setup 1: Radeon PRO W7900 (Single Container)
--------------------------------------------

Hardware
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Specification
   * - GPU
     - AMD Radeon PRO W7900 (gfx1100, RDNA 3)
   * - VRAM
     - 48 GB GDDR6
   * - Host OS
     - Ubuntu 22.04.5 LTS

Launch container
~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker run -d --name rlinf-rocm \
       --network=host \
       --device=/dev/kfd \
       --device=/dev/dri \
       --group-add=video \
       --ipc=host \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --shm-size 16G \
       -v <YOUR_RLINF_PATH>:/workspace/RLinf \
       -w /workspace/RLinf \
       rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1 \
       sleep infinity

Run training
~~~~~~~~~~~~

.. code-block:: bash

   docker exec rlinf-rocm bash -c "
   export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
   export REPO_PATH=/workspace/RLinf
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export PYTHONPATH=/workspace/RLinf:\$PYTHONPATH
   export MS_ASSET_DIR=/tmp/.maniskill

   cd /workspace/RLinf/examples/embodiment

   python3 train_embodied_agent.py \
       --config-path ./config/ \
       --config-name maniskill_sac_mlp \
       env.train.init_params.sim_backend=cpu \
       env.eval.init_params.sim_backend=cpu \
       env.train.total_num_envs=1 \
       env.eval.total_num_envs=1 \
       runner.max_epochs=10 \
       runner.val_check_interval=5 \
       actor.micro_batch_size=256 \
       actor.global_batch_size=256 \
       env.train.max_steps_per_rollout_epoch=50
   "

W7900 results
~~~~~~~~~~~~~

- Physics simulation ran on **CPU** because the GPU backend requires CUDA.
- Actor, critic, rollout inference, and tensor computation ran on the **AMD GPU through ROCm**.
- Training completed successfully for a short validation run.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Metric
     - Value
   * - Global steps
     - 3 / 3 completed
   * - Total wall time
     - ~31 seconds
   * - Physics backend
     - CPU
   * - Neural network backend
     - AMD GPU via ROCm

---

Setup 2: MI300X Single Container (Recommended)
----------------------------------------------

Hardware
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Specification
   * - GPU
     - AMD Instinct MI300X (gfx942)
   * - CPU
     - Intel Xeon Platinum 8568Y+ @ 2.0 GHz
   * - Host Memory
     - ~236 GB
   * - GPU Memory
     - 192 GB HBM3

Why llvmpipe isolation is needed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On MI300X, exposing ``/dev/dri`` is required for ROCm GPU visibility, but it also exposes the amdgpu DRM device to Mesa/Vulkan. SAPIEN initializes a Vulkan context even when ``render_backend=none`` is set, and the RADV driver segfaults on gfx942 because MI300X has no display pipeline.

The fix is to install Mesa's llvmpipe ICD and point ``VK_ICD_FILENAMES`` at it. This makes Vulkan enumerate only the CPU software device, preventing RADV from ever loading.

Launch container
~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker run -it --rm --name rlinf-mi300x-single \
       --device /dev/kfd --device /dev/dri \
       --security-opt seccomp=unconfined \
       --group-add video \
       -v <YOUR_RLINF_PATH>:/workspace/RLinf \
       rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1 \
       bash

.. note::

   The volume mount ``-v <YOUR_RLINF_PATH>:/workspace/RLinf`` is **machine-specific**. Adjust the host path to point to your own RLinf clone.

Install extra system packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The llvmpipe workaround requires Mesa Vulkan drivers:

.. code-block:: bash

   apt-get update -qq && apt-get install -y -qq \
       libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
       wget unzip cmake libglfw3-dev libglew-dev libosmesa6-dev libegl1 \
       mesa-vulkan-drivers vulkan-tools

Run training
~~~~~~~~~~~~

.. code-block:: bash

   export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
   export REPO_PATH=/workspace/RLinf
   export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl

   # llvmpipe isolation: force SAPIEN/Vulkan to use CPU software rendering
   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
   export LIBGL_ALWAYS_SOFTWARE=1
   export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

   # Gloo backend fix for single-node container
   export GLOO_SOCKET_IFNAME=lo
   export MASTER_ADDR=127.0.0.1

   export MS_ASSET_DIR=/tmp/.maniskill

   cd /workspace/RLinf/examples/embodiment

   python3 train_embodied_agent.py \
       --config-path ./config/ \
       --config-name maniskill_sac_mlp \
       env.train.init_params.sim_backend=cpu \
       env.eval.init_params.sim_backend=cpu \
       +env.train.init_params.render_backend=none \
       +env.eval.init_params.render_backend=none \
       env.train.total_num_envs=1 \
       env.eval.total_num_envs=1 \
       env.train.max_steps_per_rollout_epoch=50 \
       env.eval.max_steps_per_rollout_epoch=50 \
       runner.max_epochs=2000 \
       runner.val_check_interval=100 \
       runner.save_interval=-1 \
       actor.micro_batch_size=256 \
       actor.global_batch_size=256 \
       env.eval.video_cfg.save_video=False

MI300X convergence results
~~~~~~~~~~~~~~~~~~~~~~~~~~

A full 2,000-step convergence run was performed on MI300X using the single-container llvmpipe workaround.

**Final metrics (step 2,000 / 2,000):**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Split
     - Reward
     - Success (end)
     - Success (once)
   * - Train
     - 0.8367
     - **1.0**
     - **1.0**
   * - Eval
     - 0.8360
     - **1.0**
     - **1.0**

**Convergence timeline:**

- ``success_once`` reached 1.0 as early as step 423 in some rollouts.
- From approximately step 1,900 onward, ``success_once`` stabilized at 1.0 consistently.
- Total wall time: 43 min 49 s.
- Average step time: ~1.31 s.

**Key training parameters:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Value
   * - Algorithm
     - SAC (Soft Actor-Critic)
   * - Update epochs
     - 32
   * - Gamma
     - 0.8
   * - Tau (soft update)
     - 0.01
   * - Actor LR
     - 3.0e-4
   * - Critic LR
     - 3.0e-4
   * - Micro batch size
     - 256
   * - Global batch size
     - 256
   * - Replay buffer cache size
     - 10,000 trajectories

---

Setup 3: MI300X Split Deployment (Fallback)
-------------------------------------------

If the single-container workaround does not work in your environment (e.g. a different Mesa or driver version), use a two-container split placement.

Architecture
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Role
     - Container
     - Key Properties
   * - actor + rollout + head
     - GPU container
     - Exposed ``/dev/kfd`` and ``/dev/dri``; ROCm visible; Ray node rank ``0``
   * - env worker
     - CPU-only container
     - No ``/dev/dri``; ManiSkill with ``sim_backend=cpu`` and ``render_backend=none``; Ray node rank ``1``

Use the dedicated split config:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-path /workspace/RLinf/examples/embodiment/config \
       --config-name maniskill_sac_mlp_mi300x_split \
       env.train.init_params.sim_backend=cpu \
       env.eval.init_params.sim_backend=cpu \
       +env.train.init_params.render_backend=none \
       +env.eval.init_params.render_backend=none

The config places actor + rollout on the ``mi300x_gpu`` node group and env on the ``mi300x_env`` node group (``ignore_hardware: True``).

---

Known Limitations
-----------------

1. ``sim_backend=cpu`` currently implies a **single-environment** ManiSkill run in this workflow.
2. Simulation throughput is much lower than CUDA-based GPU physics.
3. This setup is suitable for framework verification, debugging, and small-scale validation runs, but not a replacement for full CUDA physics throughput.

Troubleshooting
---------------

Transformers import error
~~~~~~~~~~~~~~~~~~~~~~~~~

::

   ImportError: cannot import name 'AutoModelForVision2Seq'

Fix:

.. code-block:: bash

   pip install transformers==4.57.6

Multi-environment error on CPU backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   RuntimeError: Cannot set the sim backend to 'cpu' and have multiple environments

Fix:

Keep the ROCm workaround paired with:

.. code-block:: bash

   env.train.total_num_envs=1 \
   env.eval.total_num_envs=1

SAPIEN segfault during env initialization (MI300X)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   radv/amdgpu: The CS has been rejected, see dmesg for more information (-22).
   Segmentation fault

Fix:

Ensure ``mesa-vulkan-drivers`` is installed and the llvmpipe ICD is active:

.. code-block:: bash

   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
   export LIBGL_ALWAYS_SOFTWARE=1
   export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

Gloo network error
~~~~~~~~~~~~~~~~~~

::

   RuntimeError: Unable to find interface for: [0.0.0.0]

Fix:

Set ``GLOO_SOCKET_IFNAME`` to a local loopback or host network interface:

.. code-block:: bash

   export GLOO_SOCKET_IFNAME=lo
   export MASTER_ADDR=127.0.0.1

References
----------

- `RLinf repository <https://github.com/RLinf/RLinf>`_
- `ManiSkill documentation <https://maniskill.readthedocs.io/>`_
- `ROCm PyTorch Docker images <https://hub.docker.com/r/rocm/pytorch>`_
- `AMD GPU / OS support matrix <https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html>`_
