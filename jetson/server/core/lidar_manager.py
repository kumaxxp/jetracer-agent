"""STL-27L LiDAR driver manager.

Reads scan data from the STL-27L via UART and provides the latest
complete 360-degree scan to the sensor streaming pipeline.

Based on the verified D800_STL-27L_test implementation.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from typing import TYPE_CHECKING

import serial

from shared.protocol.sensor_types import LidarPoint, LidarScan

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# --- STL-27L protocol constants ---
HEADER = 0x54
VERLEN = 0x2C
POINTS_PER_PACK = 12
PACKET_SIZE = 47  # Header(1)+VerLen(1)+Speed(2)+StartAngle(2)+Data(12*3)+EndAngle(2)+Timestamp(2)+CRC(1)

# CRC8 lookup table (from STL-27L Development Manual)
CRC_TABLE = [
    0x00, 0x4D, 0x9A, 0xD7, 0x79, 0x34, 0xE3, 0xAE,
    0xF2, 0xBF, 0x68, 0x25, 0x8B, 0xC6, 0x11, 0x5C,
    0xA9, 0xE4, 0x33, 0x7E, 0xD0, 0x9D, 0x4A, 0x07,
    0x5B, 0x16, 0xC1, 0x8C, 0x22, 0x6F, 0xB8, 0xF5,
    0x1F, 0x52, 0x85, 0xC8, 0x66, 0x2B, 0xFC, 0xB1,
    0xED, 0xA0, 0x77, 0x3A, 0x94, 0xD9, 0x0E, 0x43,
    0xB6, 0xFB, 0x2C, 0x61, 0xCF, 0x82, 0x55, 0x18,
    0x44, 0x09, 0xDE, 0x93, 0x3D, 0x70, 0xA7, 0xEA,
    0x3E, 0x73, 0xA4, 0xE9, 0x47, 0x0A, 0xDD, 0x90,
    0xCC, 0x81, 0x56, 0x1B, 0xB5, 0xF8, 0x2F, 0x62,
    0x97, 0xDA, 0x0D, 0x40, 0xEE, 0xA3, 0x74, 0x39,
    0x65, 0x28, 0xFF, 0xB2, 0x1C, 0x51, 0x86, 0xCB,
    0x21, 0x6C, 0xBB, 0xF6, 0x58, 0x15, 0xC2, 0x8F,
    0xD3, 0x9E, 0x49, 0x04, 0xAA, 0xE7, 0x30, 0x7D,
    0x88, 0xC5, 0x12, 0x5F, 0xF1, 0xBC, 0x6B, 0x26,
    0x7A, 0x37, 0xE0, 0xAD, 0x03, 0x4E, 0x99, 0xD4,
    0x7C, 0x31, 0xE6, 0xAB, 0x05, 0x48, 0x9F, 0xD2,
    0x8E, 0xC3, 0x14, 0x59, 0xF7, 0xBA, 0x6D, 0x20,
    0xD5, 0x98, 0x4F, 0x02, 0xAC, 0xE1, 0x36, 0x7B,
    0x27, 0x6A, 0xBD, 0xF0, 0x5E, 0x13, 0xC4, 0x89,
    0x63, 0x2E, 0xF9, 0xB4, 0x1A, 0x57, 0x80, 0xCD,
    0x91, 0xDC, 0x0B, 0x46, 0xE8, 0xA5, 0x72, 0x3F,
    0xCA, 0x87, 0x50, 0x1D, 0xB3, 0xFE, 0x29, 0x64,
    0x38, 0x75, 0xA2, 0xEF, 0x41, 0x0C, 0xDB, 0x96,
    0x42, 0x0F, 0xD8, 0x95, 0x3B, 0x76, 0xA1, 0xEC,
    0xB0, 0xFD, 0x2A, 0x67, 0xC9, 0x84, 0x53, 0x1E,
    0xEB, 0xA6, 0x71, 0x3C, 0x92, 0xDF, 0x08, 0x45,
    0x19, 0x54, 0x83, 0xCE, 0x60, 0x2D, 0xFA, 0xB7,
    0x5D, 0x10, 0xC7, 0x8A, 0x24, 0x69, 0xBE, 0xF3,
    0xAF, 0xE2, 0x35, 0x78, 0xD6, 0x9B, 0x4C, 0x01,
    0xF4, 0xB9, 0x6E, 0x23, 0x8D, 0xC0, 0x17, 0x5A,
    0x06, 0x4B, 0x9C, 0xD1, 0x7F, 0x32, 0xE5, 0xA8,
]


def calc_crc8(data: bytes) -> int:
    """Calculate CRC8 checksum per STL-27L protocol."""
    crc = 0
    for b in data:
        crc = CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


def parse_packet(packet: bytes) -> tuple[float, list[tuple[float, int, int]]] | None:
    """Parse a single STL-27L packet.

    Returns (speed_deg_s, [(angle_deg, distance_mm, intensity), ...])
    or None on validation failure.
    """
    if len(packet) != PACKET_SIZE:
        return None
    if packet[0] != HEADER or packet[1] != VERLEN:
        return None

    if calc_crc8(packet[:-1]) != packet[-1]:
        return None

    speed = struct.unpack_from("<H", packet, 2)[0]
    start_angle = struct.unpack_from("<H", packet, 4)[0]
    end_angle = struct.unpack_from("<H", packet, 42)[0]

    diff = end_angle - start_angle
    if diff < 0:
        diff += 36000

    step = diff / (POINTS_PER_PACK - 1) if POINTS_PER_PACK > 1 else 0

    points = []
    for i in range(POINTS_PER_PACK):
        offset = 6 + i * 3
        distance = struct.unpack_from("<H", packet, offset)[0]
        intensity = packet[offset + 2]
        angle_raw = start_angle + step * i
        angle_deg = (angle_raw / 100.0) % 360.0
        points.append((angle_deg, distance, intensity))

    return speed, points


class LidarManager:
    """Manages the STL-27L LiDAR sensor.

    Runs a background thread that continuously reads UART packets,
    assembles complete 360-degree scans, and stores the latest scan
    for consumption by the sensor streaming pipeline.
    """

    def __init__(self, port: str, baudrate: int = 921600):
        self._port = port
        self._baudrate = baudrate
        self._lock = threading.Lock()
        self._latest_scan: LidarScan | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        # Scan assembly state
        self._scan_buf: list[tuple[float, int, int]] = []
        self._prev_angle = 0.0
        self._speed_deg_s = 0.0

        # Stats
        self._packets = 0
        self._crc_errors = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Start the LiDAR reader thread. Returns True if started."""
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop the LiDAR reader thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("LiDAR stopped")

    def get_latest_scan(self) -> LidarScan | None:
        """Return the latest complete scan, or None if not yet available."""
        with self._lock:
            return self._latest_scan

    def get_stats(self) -> dict:
        """Return packet/error stats."""
        return {
            "packets": self._packets,
            "crc_errors": self._crc_errors,
            "running": self._running,
        }

    def _read_loop(self):
        """Background thread: read packets from UART and assemble scans."""
        try:
            ser = serial.Serial(
                self._port,
                baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS,
                stopbits=serial.STOPBITS_ONE,
                parity=serial.PARITY_NONE,
                timeout=1.0,
            )
        except serial.SerialException as e:
            logger.error(f"LiDAR serial open failed: {e}")
            self._running = False
            return

        logger.info(f"LiDAR connected: {self._port} @ {self._baudrate} baud")
        buf = bytearray()

        while self._running:
            waiting = ser.in_waiting
            data = ser.read(min(waiting, 1024) if waiting else 1)
            if not data:
                continue
            buf.extend(data)

            while len(buf) >= PACKET_SIZE:
                idx = buf.find(HEADER)
                if idx < 0:
                    buf.clear()
                    break
                if idx > 0:
                    del buf[:idx]
                if len(buf) < PACKET_SIZE:
                    break

                if buf[1] != VERLEN:
                    del buf[:1]
                    continue

                packet = bytes(buf[:PACKET_SIZE])
                del buf[:PACKET_SIZE]

                result = parse_packet(packet)
                if result is None:
                    self._crc_errors += 1
                    continue

                speed, points = result
                self._packets += 1
                self._speed_deg_s = speed

                for angle, dist, intensity in points:
                    # Detect 360-degree wrap → complete scan
                    if angle < self._prev_angle and self._prev_angle > 300 and angle < 60:
                        self._finalize_scan()
                    self._prev_angle = angle
                    if dist > 0:
                        self._scan_buf.append((angle, dist, intensity))

        ser.close()

    def _finalize_scan(self):
        """Convert accumulated points to a LidarScan and store it."""
        if not self._scan_buf:
            return

        points = [
            LidarPoint(angle=a, distance=float(d), quality=q)
            for a, d, q in self._scan_buf
        ]

        rpm = self._speed_deg_s / 6.0  # deg/s → RPM

        scan = LidarScan(
            points=points,
            rpm=rpm,
            scan_count=len(points),
        )

        with self._lock:
            self._latest_scan = scan

        self._scan_buf = []
