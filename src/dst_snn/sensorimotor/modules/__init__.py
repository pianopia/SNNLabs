"""Built-in synthetic sensorimotor modules."""

from .mock_actuator import MockActuator
from .synthetic_sensor import SyntheticSensor
from .webcam_sensor import WebcamSensor, opencv_available

__all__ = ["MockActuator", "SyntheticSensor", "WebcamSensor", "opencv_available"]
