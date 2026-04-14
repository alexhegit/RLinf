MI300X Tutorial for ManiSkill SAC-MLP
=====================================

This document records the current validation status of ``examples/embodiment/config/maniskill_sac_mlp.yaml`` on a single AMD Instinct MI300X VF GPU.

Unlike the :doc:`rocm_maniskill_sac_mlp` workflow that was validated on a Radeon PRO W7900, the same single-container setup does **not** currently work end-to-end on this MI300X machine. The main blocker is the interaction between ManiSkill / SAPIEN renderer initialization and the container's ``/dev/dri`` exposure.

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
     - AMD Instinct MI300X VF
   * - Architecture
     - gfx942
   * - Host setup
     - Single-GPU ROCm machine

Software
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Version
   * - ROCm
     - 7.2.1
   * - Docker image
     - ``rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1``
   * - Python
     - 3.10
   * - PyTorch
     - 2.9.1+rocm7.2.1

Key difference from the W7900 workflow
--------------------------------------

On this MI300X VF machine, there are two conflicting requirements:

1. Exposing ``/dev/dri`` to the container is needed for ROCm GPU visibility.
2. Exposing ``/dev/dri`` causes ManiSkill / SAPIEN initialization to fail or segfault in the tested state-only PickCube workflow, even when CPU physics and ``render_backend=none`` are used.

That means the W7900-style single-container tutorial cannot be reused unchanged here.

Container setup used for validation
-----------------------------------

The base container command was:

.. code-block:: bash

   docker run -d --name rlinf-mi300x \
       --network=host \
       --device=/dev/kfd \
       --group-add=video \
       --ipc=host \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --shm-size 16G \
       -v /DATA/Repo/RLinf:/workspace/RLinf \
       -w /workspace/RLinf \
       rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1 \
       sleep infinity

For the validation runs, the following dependency pattern was the most reliable:

.. code-block:: bash

   # Keep the image's built-in ROCm torch stack
   pip install -e . --no-deps

   # SAPIEN still expects pkg_resources on this path
   pip install "setuptools<75.9"

   # Additional runtime dependencies used in the smoke test
   pip install ray[default]>=2.47.0 hydra-core numpy datasets torchdata scipy \
       accelerate debugpy einops nvitop pybind11 ninja pytest gsutil ruff \
       pre-commit uv huggingface_hub icmplib "wandb<0.25.1" \
       "swanlab>=0.6.11" tensorboard transformers==4.57.6 peft timm \
       imageio[ffmpeg] gymnasium gym easydict cloudpickle draccus rich \
       mani-skill sapien==3.0.1

.. note::

   Avoid ``pip install -e .`` without ``--no-deps`` in this container. In the tested setup that replaced the image's ROCm PyTorch stack with a non-ROCm torch build, which then broke ``torchvision`` imports.

What was verified successfully
------------------------------

1. ROCm / PyTorch device detection inside the container worked:

   .. code-block:: bash

      python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

   Expected output in the tested setup:

   ::

      2.9.1+rocm7.2.1...
      True
      AMD Instinct MI300X VF

2. A state-only ManiSkill ``PickCube-v1`` environment could run with CPU physics **only when the container did not expose ``/dev/dri``**:

   .. code-block:: python

      import gymnasium as gym
      import mani_skill.envs

      env = gym.make(
          "PickCube-v1",
          obs_mode="state",
          control_mode=None,
          sim_backend="cpu",
          render_backend="none",
          num_envs=1,
      )
      obs, info = env.reset(seed=0)
      obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

   In that configuration, ``reset`` and ``step`` both succeeded.

What failed in the MI300X validation
------------------------------------

Single-container run with ``/dev/dri``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the container exposed ``/dev/dri`` so that ROCm GPU training remained available, ManiSkill / SAPIEN initialization failed during environment creation.

Observed symptom:

::

   radv/amdgpu: The CS has been rejected, see dmesg for more information (-22).
   Segmentation fault

This happened even with:

- ``sim_backend=cpu``
- ``render_backend=none``
- state-only observations
- a single environment

Container without ``/dev/dri``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``/dev/dri`` was removed from the container, the environment could initialize and step, but PyTorch no longer saw the GPU:

::

   torch.cuda.is_available() == False

So this workaround is sufficient for environment-level validation only, not for end-to-end ROCm training validation.

Training script smoke test status
---------------------------------

The RLinf training script was partially validated:

- ``train_embodied_agent.py`` launched successfully with Hydra overrides for CPU physics.
- The config was composed correctly with:

  .. code-block:: bash

     env.train.init_params.sim_backend=cpu \
     env.eval.init_params.sim_backend=cpu \
     +env.train.init_params.render_backend=none \
     +env.eval.init_params.render_backend=none \
     env.train.total_num_envs=1 \
     env.eval.total_num_envs=1

- However, the run could not complete in a single container because:

  - with ``/dev/dri``: ManiSkill / SAPIEN segfaulted during env initialization
  - without ``/dev/dri``: Ray / rollout workers saw no GPU resources

Current conclusion
------------------

At the moment, the following statements are supported by direct validation on this MI300X VF machine:

- **Environment-only validation** is possible with CPU physics and ``render_backend=none`` when the container does not expose ``/dev/dri``.
- **A single-machine, two-container split smoke test** completed successfully when the environment worker ran in a CPU-only container and actor / rollout ran in a GPU-visible ROCm container.

The following statement is still **not yet validated**:

- **End-to-end single-container ManiSkill SAC-MLP training on MI300X VF**
- **Long-running training stability for the split deployment**

Validated split deployment
--------------------------

The split workaround was validated with a two-node Ray cluster on the same machine:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Role
     - Container
     - Key properties
   * - actor + rollout + head
     - GPU container
     - Exposed ``/dev/kfd`` and ``/dev/dri``; ROCm training visible; Ray node rank ``0``
   * - env worker
     - CPU-only container
     - No ``/dev/dri`` exposure; ManiSkill ``PickCube-v1`` with ``sim_backend=cpu`` and ``render_backend=none``; Ray node rank ``1``

The repository now includes a dedicated split placement example:

.. code-block:: text

   examples/embodiment/config/maniskill_sac_mlp_mi300x_split.yaml

The smoke test used that config together with the same runtime ManiSkill overrides:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-path /workspace/RLinf/examples/embodiment/config \
       --config-name maniskill_sac_mlp_mi300x_split \
       runner.max_epochs=1 \
       runner.val_check_interval=-1 \
       env.train.init_params.sim_backend=cpu \
       env.eval.init_params.sim_backend=cpu \
       +env.train.init_params.render_backend=none \
       +env.eval.init_params.render_backend=none

In the successful run:

- the ``EnvGroup`` worker was launched on the CPU-only container,
- actor / rollout workers stayed on the GPU container,
- the training loop completed ``Global Step: 1/1`` without the previous SAPIEN segmentation fault.

Recommended next directions
---------------------------

If MI300X support for this example is needed beyond the smoke test, the next experiments should focus on:

1. extending the split run beyond a one-step smoke test and checking long-run stability,
2. packaging the two-container Ray bootstrap into a cleaner tutorial workflow,
3. testing a host-native setup instead of Docker to see whether the split remains necessary,
4. investigating whether a different SAPIEN / ManiSkill renderer path is required for MI300X VF,
5. checking whether the MI300X VF virtualization setup imposes additional DRM / Vulkan restrictions compared with the W7900 workstation.

References
----------

- :doc:`rocm_maniskill_sac_mlp`
- `RLinf repository <https://github.com/RLinf/RLinf>`_
- `ManiSkill documentation <https://maniskill.readthedocs.io/>`_
