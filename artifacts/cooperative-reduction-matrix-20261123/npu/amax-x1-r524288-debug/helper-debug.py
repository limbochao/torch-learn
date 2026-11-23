@triton.jit
def x_grid_barrier_debug(sem):
    """The all-Vector prototype with the user's two local barriers.

    This has the same participation and event constraints as x_grid_barrier.
    The local barriers do not restore upstream's independent y/z groups.
    """
    tl.debug_barrier()
    extension.sync_block_all("all_vector", 0)
    tl.debug_barrier()
