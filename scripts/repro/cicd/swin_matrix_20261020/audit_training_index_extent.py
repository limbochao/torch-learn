"""Audit kernel 47's captured in_ptr0 address formula without launching the kernel."""

import json


def main():
    channels = 512
    input_numel = 1568 * channels
    maximum = -1
    invalid_elements = 0
    first_invalid = None
    for batch in range(8):
        for row in range(14):
            for window_column in range(2):
                for column in range(7):
                    base = (
                        512 * column + 3584 * row + 25088 * window_column
                        + 50176 * (row // 7) + 100352 * batch
                    )
                    maximum = max(maximum, base + channels - 1)
                    invalid = max(0, base + channels - max(base, input_numel))
                    invalid_elements += invalid
                    if invalid and first_invalid is None:
                        first_invalid = {
                            'r3': batch, 'r2': row, 'r4': window_column, 'r5': column,
                            'x0': max(0, input_numel - base), 'element_offset': max(base, input_numel),
                        }
    print(json.dumps({
        'input_shape': [1568, 512], 'input_stride': [512, 1], 'input_numel': input_numel,
        'maximum_element_offset': maximum, 'out_of_bounds_element_count': invalid_elements,
        'first_out_of_bounds_index': first_invalid,
        'note': 'CPU address audit only; this does not launch the generated kernel or compare model accuracy.',
    }, indent=2))


if __name__ == '__main__':
    main()
