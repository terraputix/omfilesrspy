import os
import statistics
import time
from functools import wraps

import psutil


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


# Decorator to measure memory usage
def measure_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        before_mem = get_memory_usage()
        result = func(*args, **kwargs)
        after_mem = get_memory_usage()
        return result, after_mem - before_mem

    return wrapper


# Decorator to measure execution time
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        cpu_start_time = time.process_time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        cpu_elapsed_time = time.process_time() - cpu_start_time
        return result, elapsed_time, cpu_elapsed_time

    return wrapper


def run_benchmark(func, iterations=5):
    times = []
    cpu_times = []
    for _ in range(iterations):
        result, elapsed, cpu_elapsed = func()
        times.append(elapsed)
        cpu_times.append(cpu_elapsed)

    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times),
        "min": min(times),
        "max": max(times),
        "cpu_mean": statistics.mean(cpu_times),
        "cpu_std": statistics.stdev(cpu_times),
    }
