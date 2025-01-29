import numpy as np


def generate_test_data(array_size, noise_level=5, amplitude=20, offset=20):
    """Generate sinusoidal test data with noise.

    Args:
        array_size: Tuple specifying dimensions of output array
        noise_level: Standard deviation of Gaussian noise to add
        amplitude: Amplitude of sine wave
        offset: Vertical offset of sine wave

    Returns:
        numpy.ndarray: Array containing sinusoidal data with noise
    """
    x = np.linspace(0, 2 * np.pi * (array_size[1] / 100), array_size[1])
    sinusoidal_data = amplitude * np.sin(x) + offset
    noise = np.random.normal(0, noise_level, array_size)
    return (sinusoidal_data + noise).astype(np.float32)
