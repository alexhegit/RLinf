ROCm Tutorial for ManiSkill SAC-MLP
===================================

This document shows how to run ``examples/embodiment/config/maniskill_sac_mlp.yaml`` on an AMD GPU with ROCm.
The key constraint is that ManiSkill / SAPIEN physics still depends on CUDA for the GPU backend, so on ROCm the recommended workaround is to keep the repo's tracked config unchanged and override the physics backend to CPU at launch time.

This page focuses on the end-to-end ROCm workflow for this example.

Tested Platform
---------------

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
   * - Docker
     - 28.0.1

Software
~~~~~~~~

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
     - 0.2.0
   * - Docker image
     - ``rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1``

Why this tutorial keeps the repo config unchanged
-------------------------------------------------

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

Environment Setup
-----------------

1. Clone RLinf
~~~~~~~~~~~~~~

.. code-block:: bash

   cd /DATA/Repo
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Launch the ROCm container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
       -v /DATA/Repo/RLinf:/workspace/RLinf \
       -v /home/amd/dockerx:/dockerx \
       -w /workspace/RLinf \
       rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1 \
       sleep infinity

3. Install system dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker exec rlinf-rocm bash -c "
   export DEBIAN_FRONTEND=noninteractive
   apt-get update -qq && apt-get install -y -qq \
       libgl1 libglib2.0-0 libsm6 libxext6 \
       libxrender-dev libgomp1 wget unzip cmake \
       libglfw3-dev libglew-dev libosmesa6-dev libegl1
   "

4. Install Python dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker exec rlinf-rocm bash -c "
   cd /workspace/RLinf
   pip install -q -e .
   pip install -q sapien==3.0.1 gymnasium gym transformers==4.57.6 \
       peft timm imageio[ffmpeg] easydict cloudpickle draccus rich
   pip install -q mani-skill
   "

Verification
------------

Verify that PyTorch sees the AMD GPU inside the container:

.. code-block:: bash

   docker exec rlinf-rocm python3 -c "
   import torch
   print(f'PyTorch: {torch.__version__}')
   print(f'ROCm available: {torch.cuda.is_available()}')
   print(f'Device: {torch.cuda.get_device_name(0)}')
   "

Expected output:

::

   PyTorch: 2.9.1+rocm7.2.1
   ROCm available: True
   Device: AMD Radeon PRO W7900

Run Training
------------

Set the common environment variables:

.. code-block:: bash

   docker exec rlinf-rocm bash -c "
   export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
   export REPO_PATH=/workspace/RLinf
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export PYTHONPATH=/workspace/RLinf:\$PYTHONPATH
   export MS_ASSET_DIR=/dockerx/.maniskill
   "

Then launch the ManiSkill SAC-MLP example with ROCm-specific overrides:

.. code-block:: bash

   docker exec rlinf-rocm bash -c "
   export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
   export REPO_PATH=/workspace/RLinf
   export PYTHONPATH=/workspace/RLinf:\$PYTHONPATH
   export MS_ASSET_DIR=/dockerx/.maniskill

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
       env.train.max_steps_per_rollout_epoch=1
   "

Results Summary
---------------

In the tested setup:

- Physics simulation ran on **CPU** because the GPU backend requires CUDA.
- Actor, critic, rollout inference, and tensor computation still ran on the **AMD GPU through ROCm**.
- Training completed successfully for a short validation run of the ``maniskill_sac_mlp`` example.

Observed metrics from a 3-step smoke test:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Metric
     - Value
   * - Global steps
     - 3 / 3 completed
   * - Total wall time
     - About 31 seconds
   * - Physics backend
     - CPU
   * - Neural network backend
     - AMD GPU via ROCm
   * - Numerical stability
     - No NaN / Inf observed in the test run

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

References
----------

- `RLinf repository <https://github.com/RLinf/RLinf>`_
- `ManiSkill documentation <https://maniskill.readthedocs.io/>`_
- `ROCm PyTorch Docker images <https://hub.docker.com/r/rocm/pytorch>`_
- `AMD GPU / OS support matrix <https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html>`_
