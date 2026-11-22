"""
Serial Communication Controller for Fog Detection Alert
Uses pySerial to send alerts (e.g. '*1') to connected microcontrollers/hardware.
"""

import os
import logging
import threading
import time
from typing import Tuple, Optional, List, Dict, Any

try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    serial = None
    PYSERIAL_AVAILABLE = False

logger = logging.getLogger(__name__)


def list_available_ports() -> List[Dict[str, str]]:
    """
    Returns a list of available serial port descriptions.
    """
    if not PYSERIAL_AVAILABLE:
        return []
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append({
            "device": p.device,
            "description": p.description or "N/A",
            "hwid": p.hwid or "N/A",
        })
    return ports


def detect_candidate_port() -> Optional[str]:
    """
    Attempts to identify an appropriate hardware serial port (USB / Arduino / UART).
    Checks Django settings SERIAL_PORT, environment variable, or scans available ports.
    """
    # 1. Django settings override (highest priority for project config)
    try:
        from django.conf import settings
        cfg_port = getattr(settings, "SERIAL_PORT", None)
        if cfg_port:
            return str(cfg_port).strip()
    except Exception:
        pass

    # 2. Environment variable override
    env_port = os.environ.get("SERIAL_PORT")
    if env_port:
        return env_port.strip()

    if not PYSERIAL_AVAILABLE:
        return None

    # 3. Scan ports for USB/Modem/Serial hardware
    comports = serial.tools.list_ports.comports()
    keywords = ("usb", "usbmodem", "usbserial", "ch340", "cp210", "ftdi", "arduino", "ttyusb", "ttyacm")

    for p in comports:
        search_target = f"{p.device} {p.description} {p.hwid}".lower()
        if any(kw in search_target for kw in keywords):
            return p.device

    return None


class SerialController:
    """
    Thread-safe controller managing persistent serial connection and sending alerts.
    """
    _instance = None
    _singleton_lock = threading.Lock()

    def __init__(self, port: Optional[str] = None, baudrate: int = 9600, timeout: float = 1.0):
        self.custom_port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection: Optional[serial.Serial] = None
        self.active_port: Optional[str] = None
        self.lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SerialController":
        with cls._singleton_lock:
            if cls._instance is None:
                baud = 9600
                try:
                    from django.conf import settings
                    cfg_baud = getattr(settings, "SERIAL_BAUDRATE", None)
                    if cfg_baud is not None:
                        baud = int(cfg_baud)
                except Exception:
                    pass
                if "SERIAL_BAUDRATE" in os.environ:
                    try:
                        baud = int(os.environ["SERIAL_BAUDRATE"])
                    except ValueError:
                        pass
                cls._instance = cls(baudrate=baud)
            return cls._instance

    def _ensure_connected(self) -> Tuple[bool, str]:
        """
        Connects or verifies the existing connection.
        """
        if not PYSERIAL_AVAILABLE:
            return False, "pySerial library is not installed."

        target_port = self.custom_port or detect_candidate_port()
        if not target_port:
            return False, "No serial port configured or detected. Please set SERIAL_PORT in settings.py."

        # Dynamically read baud rate from Django settings if set
        target_baud = self.baudrate
        try:
            from django.conf import settings
            cfg_baud = getattr(settings, "SERIAL_BAUDRATE", None)
            if cfg_baud is not None:
                target_baud = int(cfg_baud)
        except Exception:
            pass
        if "SERIAL_BAUDRATE" in os.environ:
            try:
                target_baud = int(os.environ["SERIAL_BAUDRATE"])
            except ValueError:
                pass
        self.baudrate = target_baud

        # If already connected, verify port and baud rate haven't changed
        if self.connection and self.connection.is_open:
            if self.active_port == target_port and getattr(self.connection, "baudrate", None) == target_baud:
                return True, f"Already connected to {target_port} at {target_baud} baud"
            else:
                self.close()

        try:
            self.connection = serial.Serial(
                port=target_port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            self.active_port = target_port
            # Brief delay for Arduino bootloader if freshly opened
            time.sleep(0.1)
            logger.info("Opened serial connection on %s at %d baud", target_port, self.baudrate)
            return True, f"Connected to {target_port} at {self.baudrate} baud"
        except Exception as e:
            self.connection = None
            self.active_port = None
            logger.warning("Failed to open serial port %s: %s", target_port, e)
            return False, f"Failed to open port {target_port}: {e}"

    def send_command(self, command: str = "*1") -> Tuple[bool, str]:
        """
        Sends the specified command string over serial.
        Returns (success: bool, detail_message: str).
        """
        with self.lock:
            ok, msg = self._ensure_connected()
            if not ok:
                return False, msg

            try:
                cmd_bytes = command.encode("utf-8")
                self.connection.write(cmd_bytes)
                self.connection.flush()
                logger.info("Sent serial command: %s to %s", command, self.active_port)
                return True, f"Sent '{command}' to {self.active_port}"
            except Exception as e:
                logger.error("Error writing to serial port %s: %s", self.active_port, e)
                # Invalidate connection so next call tries to reconnect
                try:
                    if self.connection:
                        self.connection.close()
                except Exception:
                    pass
                self.connection = None
                return False, f"Failed to send '{command}': {e}"

    def close(self):
        """
        Safely closes the serial connection if open.
        """
        with self.lock:
            if self.connection:
                try:
                    self.connection.close()
                except Exception:
                    pass
                self.connection = None
                self.active_port = None


def send_fog_alert(command: str = "*1", port: Optional[str] = None, baudrate: Optional[int] = None) -> Tuple[bool, str]:
    """
    Public convenience helper to send serial command when fog is detected.
    """
    controller = SerialController.get_instance()
    if port:
        controller.custom_port = port
    if baudrate:
        controller.baudrate = baudrate
    return controller.send_command(command)


def get_serial_status() -> Dict[str, Any]:
    """
    Returns diagnostics about the serial connection and available ports.
    """
    controller = SerialController.get_instance()
    ports = list_available_ports()
    target_port = controller.custom_port or detect_candidate_port()
    is_open = bool(controller.connection and controller.connection.is_open)
    return {
        "pyserial_available": PYSERIAL_AVAILABLE,
        "active_port": controller.active_port,
        "target_port": target_port,
        "is_connected": is_open,
        "baudrate": controller.baudrate,
        "available_ports": ports,
    }

