# GPU availability and current training device

The current ten training processes use the CPU. The Apple MPS backend is built
and available in PyTorch 2.15.0; GPU availability is not the reason
for CPU-only execution. Native trainers construct the networks/tensors on CPU,
exchange NumPy observations/actions/hidden features every control, and perform
physics, sensing, rewards and evaluation with the current CPU simulator.
The [read-only measurement](gpu-training-probe.json) uses a captured circle seed11
checkpoint at update 247, without modifying active jobs.

| Measurement | CPU | MPS GPU | Scope |
| --- | ---: | ---: | --- |
| GRU64 forward, batch40 robots | 0.0583 ms | 0.3272 ms | Inputs and hidden already resident; synchronized timing |
| GPU forward with host input/action/hidden exchanges | Not separately measured | 3.8576 ms | Includes copies and host synchronization required by the current control boundary |
| Packed recurrent actor forward/backward,64 steps x80 streams | 22.64 ms | 44.63 ms | Diagnostic squared-output loss, not a full PPO update |

The actual native CPU rollout took 1.134s and its
three-epoch PPO update took 0.702s in this probe.
Rollout accounts for 61.7% of the measured total.
These measurements occurred under the existing concurrent training load and are
one bounded workload sample, not an idle-machine benchmark or a universal backend
ranking. The GPU tests actually execute on MPS. Forward parity errors are
2.38e-07 for means and
8.94e-08 for hidden state.

A naive whole-critic move to MPS raises TypeError because the existing PopArt
statistics use FP64, unsupported by this MPS device. A future accelerator path
could retain those statistics on CPU and transfer appropriate FP32 values; the
current trainer has no such path. Its CPU NumPy exchanges and recurrent chunk
loops also need deliberate device/batching handling.

The measured small actor workloads favor CPU. This does not measure a complete
GPU PPO implementation, larger batches or an accelerated simulator, and therefore
does not establish their end-to-end speed. Packed CPU and MPS training should be
compared with identical PPO objectives, recurrent resets, optimizer state and
complete rollout/transfer timing before changing a long-running experiment.
Current jobs continue unchanged; no device migration or training candidate is
created by this diagnostic.
