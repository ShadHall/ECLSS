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

class DriftNoise(NoiseComponent):
    """Slow drift component, time dependent"""
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
        self.mixing_time = 120.0      # Time constant for air mixing (seconds)
        self.ventilation_effect = 0.02  # Effect of ventilation changes
        self.pressure_effect = 0.05     # Effect of pressure variations
        
        # Mean reversion parameters
        self.mean_reversion_rate = 0.01  # Rate at which the system returns to nominal
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
        self.emitter_noise = DriftNoise(amplitude=self.base_emitter_noise, frequency=0.001)
        self.shot_noise = WhiteNoise(amplitude=self.base_shot_noise)
        self.thermal_noise = WhiteNoise(amplitude=self.base_thermal_noise)
        self.ambient_noise = FlickerNoise(amplitude=self.base_ambient_noise, frequency=0.1)
        
        # Temperature scaling factors (per degree C deviation from reference temperature)
        self.temp_scaling = {
            'emitter': 0.0007,   # 0.07% per degree C - IR source with active temperature control
            'thermal': 0.0006,   # 0.06% per degree C - temperature-stabilized detector
            'shot': 0.00015,     # 0.015% per degree C - quantum-limited noise floor
            'ambient': 0.0004    # 0.04% per degree C - controlled space station environment
        }
        # Reference temperature (25°C = 77°F)
        self.ref_temp_kelvin = 298.15
    
    def _scale_noise(self, base_noise: float, co2_level: float, is_shot_noise: bool = False) -> float:
        """Scale noise based on CO2 level with non-linear effects at extremes"""
        if is_shot_noise:
            return base_noise * np.sqrt(co2_level / 2.5)
        
        # Non-linear scaling with wider ranges for -2 to 8 mmHg operation
        if co2_level < 0.0:
            # Enhanced sensitivity for negative values
            scale = (co2_level / 2.5) * (1.0 + 0.3 * (0.0 - co2_level))
        elif co2_level < 2.5:
            # Linear scaling below reference
            scale = co2_level / 2.5
        elif co2_level < 5.0:
            # Linear scaling above reference
            scale = co2_level / 2.5
        else:
            # Reduced scaling for very high values to prevent excessive noise
            scale = (co2_level / 2.5) * (1.0 - 0.15 * (co2_level - 5.0))
        return base_noise * scale
    
    def _scale_temperature(self, temp_f: float) -> float:
        """Scale noise based on temperature"""
        return ((temp_f - 32.0) * 5.0/9.0 + 273.15 - self.ref_temp_kelvin) / 273.15
    
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
        
        # Combine noise components
        return float(
            emitter_noise * (1.0 + temp_scale * self.temp_scaling['emitter']) +
            shot_noise * (1.0 + temp_scale * self.temp_scaling['shot']) +
            thermal_noise * (1.0 + temp_scale * self.temp_scaling['thermal']) +
            ambient_noise * (1.0 + temp_scale * self.temp_scaling['ambient'])
        ) 