import numpy as np

def sample_weather_trajectory(total_laps, base_rain, sigma=0.05):
    noise = np.random.normal(0, sigma, total_laps)
    rain_series = base_rain + noise
    rain_series = np.clip(rain_series, 0, None)

    return rain_series