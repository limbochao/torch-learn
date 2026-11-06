def horizontal_partition(nodes, triton_scheduling, kernel_map, node_info_map, custom_algorithm=False):
    original_groups = []
    try:
        if custom_algorithm:
            for node in nodes:
                device, sizes = node.group
                if any((isinstance(size, NumelList) for size in sizes)):
                    original_groups.append((node, node.group))
                    node.group = (device, tuple((sympy.sympify(size.numels()) if isinstance(size, NumelList) else size for size in sizes)))
        partitions = ComboKernel.horizontal_partition(nodes, triton_scheduling, kernel_map, node_info_map, custom_algorithm)
    finally:
        for node, group in original_groups:
            node.group = group
    compatible_partitions = []
    for partition in partitions:
        groups = {}
        for node in partition:
            kernel = kernel_map[node]
            key = (kernel.npu_kernel_type, kernel.inside_reduction, kernel.persistent_reduction, tuple(((axis.name, axis.is_split_axis, axis.is_tiling_axis, axis.is_no_loop_axis, axis.is_reduction) for axis in kernel.sorted_axis)))
            groups.setdefault(key, []).append(node)
        compatible_partitions.extend(groups.values())
    return compatible_partitions
