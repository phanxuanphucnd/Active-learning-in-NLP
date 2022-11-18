import time


def measure_time(func, has_return_value=True):
    start_time = time.perf_counter()
    return_values = func()
    end_time = time.perf_counter()

    if has_return_value:
        return (end_time-start_time), return_values
    else:
        return end_time-start_time
