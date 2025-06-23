import numpy as np
from typing import Optional

class NoiseComponent:
    """Base class for all noise components"""
    def __init__(self, amplitude: float):
        self.amplitude = amplitude
        self._last_value = 0.0
    
    def generate(self, time_step: float = 1.0) -> float:
        """Generate noise for our 1-second time step"""
        raise NotImplementedError("Subclasses must implement generate()")

class WhiteNoise(NoiseComponent):
    """Gaussian white noise implementation, not time dependent"""
    def __init__(self, amplitude: float):
        super().__init__(amplitude)
        self.rng = np.random.RandomState()
    
    def generate(self, time_step: float = 1.0) -> float:
        return self.rng.normal(0, self.amplitude)

class PoissonNoise(NoiseComponent):
    """Poisson noise implementation for shot noise in photon detection"""
    def __init__(self, amplitude: float):
        super().__init__(amplitude)
        self.rng = np.random.RandomState()
    
    def generate(self, time_step: float = 1.0) -> float:
        # Convert amplitude to mean number of photons
        # The amplitude represents the standard deviation at reference CO2 level
        mean_photons = (self.amplitude ** 2)  # Since variance = mean for Poisson
        # Generate Poisson random number
        photons = self.rng.poisson(mean_photons)
        # Convert back to noise amplitude
        return (photons - mean_photons) / np.sqrt(mean_photons)

class FlickerNoise(NoiseComponent):
    """1/f noise implementation using a simple approximation, time dependent"""
    def __init__(self, amplitude: float, frequency: float = 0.1):
        super().__init__(amplitude)
        self.frequency = frequency
        self.rng = np.random.RandomState()
    
    def generate(self, time_step: float = 1.0) -> float:
        alpha = np.exp(-self.frequency)
        self._last_value = alpha * self._last_value + self.rng.normal(0, self.amplitude * np.sqrt(1 - alpha))
        return self._last_value

class RandomWalkNoise(NoiseComponent):
    """Random walk with occasional resets to simulate sensor drift and recalibration"""
    def __init__(self, amplitude: float, frequency: float = 0.01):
        super().__init__(amplitude)
        self.frequency = frequency
        # Create a separate random number generator for this instance
        self.rng = np.random.RandomState()
    
    def generate(self, time_step: float = 1.0) -> float:
        if self.rng.random() < self.frequency:
            self._last_value = 0.0
        self._last_value += self.rng.normal(0, self.amplitude)
        return self._last_value

class PPCO2TrueValue:
    """Models the true CO2 level variations in the environment"""
    def __init__(self):
        # True value variation parameters
        self.base_variation = 0.0005  # 0.05% base variation
        self.ventilation_effect = 0.02  # Effect of ventilation changes
        self.pressure_effect = 0.05     # Effect of pressure variations
        
        # Mean reversion parameters
        self.mean_reversion_rate = 0.017  # Rate at which the system returns to nominal
        self.volatility = 0.0003        # Volatility of the process
        
        # Initialize true value state variables
        self._last_true_value = 0.0
        self._ventilation_state = 0.0
        self._pressure_state = 0.0
    
    def generate(self, time_step: float = 1.0, base_co2: float = 0.5) -> float:
        """
        Generate realistic variation in true CO2 level using a mean-reverting process
        
        Args:
            time_step: Time step in seconds
            base_co2: Base CO2 level in mmHg
            
        Returns:
            float: True CO2 level with variations (mmHg)
        """
        # Mean-reverting process for base variations (implemented as an Ohrnstein-Uhlenbeck process)
        # dx = -θ(x-μ)dt + σdW
        # where θ is mean reversion rate, μ is mean (0), σ is volatility
        mean_reversion_term = -self.mean_reversion_rate * self._last_true_value * time_step
        random_term = self.volatility * np.sqrt(time_step) * np.random.normal(0, 1)
        self._last_true_value += mean_reversion_term + random_term
        
        # Ventilation effect (slow changes)
        if np.random.random() < 0.01:  # 1% chance of ventilation change
            self._ventilation_state = np.random.normal(0, self.ventilation_effect)
        
        # Pressure effect (very slow changes)
        if np.random.random() < 0.001:  # 0.1% chance of pressure change
            self._pressure_state = np.random.normal(0, self.pressure_effect)
        
        return float(base_co2 + self._last_true_value + self._ventilation_state + self._pressure_state)

class PPCO2Sensor:
    """Models the noise characteristics of a NDIR ppCO2 sensor"""
    def __init__(self, temperature: float = 298.15):
        # Base noise amplitudes at 0.5 mmHg CO2
        self.base_emitter_noise = 0.0004  # 0.1% of full scale
        self.base_shot_noise = 0.00025    # Quantum noise floor
        self.base_thermal_noise = 0.0004  # Johnson-Nyquist noise
        self.base_ambient_noise = 0.0003  # Environmental effects
        
        # Initialize noise components with base amplitudes
        self.emitter_noise = RandomWalkNoise(amplitude=self.base_emitter_noise, frequency=0.001)
        self.shot_noise = PoissonNoise(amplitude=self.base_shot_noise)
        self.thermal_noise = WhiteNoise(amplitude=self.base_thermal_noise)
        self.ambient_noise = FlickerNoise(amplitude=self.base_ambient_noise, frequency=0.1)
        
        # Temperature scaling factors (per degree C deviation from reference temperature)
        self.temp_scaling = {
            'emitter': 0.0007,   # 0.07% per degree C - IR source with active temperature control
            'thermal': 0.0006,   # 0.06% per degree C - temperature-stabilized detector (will be sqrt scaled)
            'shot': 0.00015,     # 0.015% per degree C - quantum-limited noise floor
            'ambient': 0.0004    # 0.04% per degree C - controlled space station environment
        }
        # Reference temperature (25°C = 77°F)
        self.ref_temp_kelvin = 298.15
    
    def _scale_noise(self, base_noise: float, co2_level: float, is_shot_noise: bool = False) -> float:
        """Scale noise based on CO2 level with non-linear effects at extremes"""
        if is_shot_noise:
            # For Poisson noise, we scale the mean (which is the square of the amplitude)
            return base_noise * np.sqrt(co2_level / 0.5)
        
        # Simplified non-linear scaling
        if co2_level < 0.5:
            scale = (co2_level / 0.5) * (1.0 + 0.2 * (0.5 - co2_level))
        elif co2_level > 2.0:
            scale = (co2_level / 0.5) * (1.0 - 0.1 * (co2_level - 2.0))
        else:
            scale = co2_level / 0.5
        return base_noise * scale
    
    def _scale_temperature(self, temp_f: float) -> float:
        """Scale noise based on temperature"""
        temp_k = (temp_f - 32.0) * 5.0/9.0 + 273.15
        return (temp_k - self.ref_temp_kelvin) / self.ref_temp_kelvin
    
    def generate(self, time_step: float = 1.0, temperature: float = 77.0, co2_level: float = 0.5) -> float:
        """
        Generate sensor noise for the given CO2 level
        
        Args:
            time_step: Time step in seconds
            temperature: Temperature in Fahrenheit
            co2_level: Current CO2 level in mmHg
            
        Returns:
            float: Sensor noise (mmHg)
        """
        # Generate sensor noise components
        emitter_noise = float(self.emitter_noise.generate() * self._scale_noise(1.0, co2_level))
        shot_noise = float(self.shot_noise.generate() * self._scale_noise(1.0, co2_level, is_shot_noise=True))
        thermal_noise = float(self.thermal_noise.generate())
        ambient_noise = float(self.ambient_noise.generate() * self._scale_noise(1.0, co2_level))
        
        # Calculate temperature scaling
        temp_scale = self._scale_temperature(temperature)
        
        # Combine noise components with appropriate temperature scaling
        return float(
            emitter_noise * (1.0 + temp_scale * self.temp_scaling['emitter']) +
            shot_noise * (1.0 + temp_scale * self.temp_scaling['shot']) +
            thermal_noise * (1.0 + np.sqrt(1.0 + temp_scale) * self.temp_scaling['thermal']) +  # Square root scaling for thermal
            ambient_noise * (1.0 + temp_scale * self.temp_scaling['ambient'])
        )

class PPO2TrueValue:
    """Models the true O2 level variations in the environment"""
    def __init__(self):
        # True value variation parameters
        self.base_variation = 0.001  # 0.1% base variation (kept for now)
        self.ventilation_effect = 0.001  # Effect of ventilation changes (reduced)
        self.pressure_effect = 0.002     # Effect of pressure variations (reduced)

        # Mean reversion parameters
        self.mean_reversion_rate = 0.017  # Rate at which the system returns to nominal (same as CO2)
        self.volatility = 0.00002        # Volatility of the process (reduced)

        # Initialize true value state variables
        self._last_true_value = 0.0
        self._ventilation_state = 0.0
        self._pressure_state = 0.0

    def generate(self, time_step: float = 1.0, base_o2: float = 163.81) -> float:
        """
        Generate realistic variation in true O2 level using a mean-reverting process
        Args:
            time_step: Time step in seconds
            base_o2: Base O2 level in mmHg
        Returns:
            float: True O2 level with variations (mmHg)
        """
        # Mean-reverting process for base variations (Ornstein-Uhlenbeck process)
        mean_reversion_term = -self.mean_reversion_rate * self._last_true_value * time_step
        random_term = self.volatility * np.sqrt(time_step) * np.random.normal(0, 1)
        self._last_true_value += mean_reversion_term + random_term

        # Ventilation effect (slow changes)
        if np.random.random() < 0.01:  # 1% chance of ventilation change
            self._ventilation_state = np.random.normal(0, self.ventilation_effect)

        # Pressure effect (very slow changes)
        if np.random.random() < 0.001:  # 0.1% chance of pressure change
            self._pressure_state = np.random.normal(0, self.pressure_effect)

        return float(base_o2 + self._last_true_value + self._ventilation_state + self._pressure_state)

class PPO2Sensor:
    """Models the noise characteristics of an electrochemical ppO2 sensor, including humidity and temperature effects"""
    def __init__(self, calibration_temp_c: float = 23.0, nominal_ppO2: float = 163.81, nominal_humidity: float = 50.0):
        # Base noise amplitudes at nominal conditions (all reduced)
        self.base_drift_noise = 0.01      # mmHg, random walk (reduced)
        self.base_random_noise = 0.01    # mmHg, white noise (reduced)
        self.base_thermal_noise = 0.01   # mmHg, temperature noise (reduced)
        self.base_ambient_noise = 0.01   # mmHg, flicker noise (reduced)
        self.humidity_sensitivity = 0.001  # mmHg per %RH deviation

        # Calibration and nominal values
        self.calibration_temp_c = calibration_temp_c
        self.nominal_ppO2 = nominal_ppO2
        self.nominal_humidity = nominal_humidity

        # Temperature scaling factors (per degree C deviation from calibration)
        self.temp_scaling = {
            'drift': 0.002,    # per deg C
            'random': 0.001,   # per deg C
            'thermal': 0.003,  # per deg C
            'ambient': 0.001   # per deg C
        }

        # State for drift noise
        self._last_drift = 0.0
        self.rng = np.random.RandomState()
        self._last_ambient = 0.0

    def _scale_noise(self, base_noise: float, ppO2_level: float) -> float:
        """Scale noise based on ppO2 level (linear scaling)"""
        return base_noise * (ppO2_level / self.nominal_ppO2)

    def _scale_temperature(self, temp_c: float, noise_type: str) -> float:
        """Scale noise based on temperature deviation from calibration"""
        delta_t = temp_c - self.calibration_temp_c
        return 1.0 + self.temp_scaling[noise_type] * delta_t

    def _humidity_bias(self, current_humidity: float) -> float:
        """Bias or noise term proportional to deviation from nominal humidity"""
        return self.humidity_sensitivity * (current_humidity - self.nominal_humidity)

    def generate(self, time_step: float = 1.0, temperature: float = 73.4, ppO2_level: float = 163.81, humidity: float = 50.0) -> float:
        """
        Generate sensor noise for the given O2 level
        Args:
            time_step: Time step in seconds
            temperature: Temperature in Fahrenheit (converted to Celsius)
            ppO2_level: Current O2 level in mmHg
            humidity: Current relative humidity in %RH
        Returns:
            float: Sensor noise (mmHg)
        """
        # Convert temperature to Celsius if given in Fahrenheit
        if temperature > 60:  # crude check, assume F if > 60
            temp_c = (temperature - 32.0) * 5.0/9.0
        else:
            temp_c = temperature

        # Drift noise (random walk)
        drift_scale = self._scale_noise(self.base_drift_noise, ppO2_level) * self._scale_temperature(temp_c, 'drift')
        self._last_drift += self.rng.normal(0, drift_scale)
        drift_noise = self._last_drift

        # Random/white noise
        random_noise = self.rng.normal(0, self._scale_noise(self.base_random_noise, ppO2_level) * self._scale_temperature(temp_c, 'random'))

        # Thermal noise
        thermal_noise = self.rng.normal(0, self._scale_noise(self.base_thermal_noise, ppO2_level) * self._scale_temperature(temp_c, 'thermal'))

        # Ambient/flicker noise (simple 1/f approximation)
        alpha = np.exp(-0.1)
        ambient_scale = self._scale_noise(self.base_ambient_noise, ppO2_level) * self._scale_temperature(temp_c, 'ambient')
        self._last_ambient = alpha * self._last_ambient + self.rng.normal(0, ambient_scale * np.sqrt(1 - alpha))
        ambient_noise = self._last_ambient

        # Humidity bias
        humidity_bias = self._humidity_bias(humidity)

        # Combine all noise components
        return float(drift_noise + random_noise + thermal_noise + ambient_noise + humidity_bias)

class HumidityTrueValue:
    """Models the true humidity variations in the environment using an Ornstein-Uhlenbeck process with rare event-driven changes."""
    def __init__(self, nominal_humidity: float = 52.0, mean_reversion: float = 0.017, volatility: float = 0.002):
        self.nominal_humidity = nominal_humidity  # %RH, NASA-STD-3001 optimal
        self.mean_reversion = mean_reversion      # Matched to ppO2/ppCO2 (0.017)
        self.volatility = volatility              # Lowered for realistic 1s changes
        self._last_value = nominal_humidity
        self._event_state = 0.0

    def generate(self, time_step: float = 1.0) -> float:
        # Ornstein-Uhlenbeck process
        mean_reversion_term = self.mean_reversion * (self.nominal_humidity - self._last_value) * time_step
        random_term = self.volatility * np.sqrt(time_step) * np.random.normal(0, 1)
        self._last_value += mean_reversion_term + random_term
        # Rare event-driven changes (e.g., crew activity, equipment cycling)
        if np.random.random() < 0.0001:  # ~once every 2.7 hours
            self._event_state = np.random.normal(0, 0.05)  # Small event, up to ±0.05% RH
        self._event_state *= 0.98
        return float(self._last_value + self._event_state)

class HumiditySensor:
    """Models the noise characteristics of a capacitive humidity sensor with drift, flicker, and white noise."""
    def __init__(self, base_noise: float = 0.02, drift_noise: float = 0.00005, flicker_noise: float = 0.01):
        self.base_noise = base_noise      # White noise, %RH (set for ISS/space-rated sensor)
        self.drift_noise = drift_noise    # Random walk drift, %RH
        self.flicker_noise = flicker_noise  # Flicker noise, %RH
        self.rng = np.random.RandomState()
        self._last_drift = 0.0
        self._last_flicker = 0.0

    def generate(self, true_humidity: float) -> float:
        # Drift (random walk)
        self._last_drift += self.rng.normal(0, self.drift_noise)
        # Flicker noise (1/f, low frequency)
        alpha = np.exp(-0.05)  # Lower frequency than white noise
        self._last_flicker = alpha * self._last_flicker + self.rng.normal(0, self.flicker_noise * np.sqrt(1 - alpha))
        # White noise
        white_noise = self.rng.normal(0, self.base_noise)
        # Combine all noise components
        return float(true_humidity + self._last_drift + self._last_flicker + white_noise)