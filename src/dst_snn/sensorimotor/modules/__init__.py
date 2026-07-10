"""Built-in synthetic sensorimotor modules."""

from .mock_actuator import MockActuator
from .serial_bridge import MockSerialPort, SerialMotorBridge, SerialTactileSensor
from .synthetic_sensor import SyntheticSensor
from .webcam_sensor import WebcamSensor, opencv_available

__all__ = [
    "MockActuator",
    "MockSerialPort",
    "SerialMotorBridge",
    "SerialTactileSensor",
    "SyntheticSensor",
    "WebcamSensor",
    "opencv_available",
]
