"""NPU synchronization prototypes for cooperative reduction experiments."""

import triton
import triton.language as tl
from triton.language.extra.cann import extension


@triton.jit
def x_grid_barrier(sem):
    """Synchronize all participating Vector cores using event 0.

    Keep the upstream call signature; sem is neither read nor written.
    Unlike upstream's independent y/z groups, this synchronizes the entire
    participating Vector set. All participants must reach every invocation
    in the same order, including calls in loops and across output groups.

    The caller must use a supported Vector execution mode and mapping with
    all participants able to reach the barrier. Event 0 must be available
    for this protocol. This prototype is validated on A5 SIMD only.
    """
    extension.sync_block_all("all_vector", 0)


@triton.jit
def x_grid_barrier_debug(sem):
    """The all-Vector prototype with the user's two local barriers.

    This has the same participation and event constraints as x_grid_barrier.
    The local barriers do not restore upstream's independent y/z groups.
    """
    tl.debug_barrier()
    extension.sync_block_all("all_vector", 0)
    tl.debug_barrier()
