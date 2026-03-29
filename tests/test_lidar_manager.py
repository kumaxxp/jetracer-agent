"""Unit tests for the STL-27L LiDAR driver.

Tests packet parsing, CRC validation, and scan assembly logic
using the Development Manual 3.3 example data.
"""

import struct
import threading

import pytest

from jetson.server.core.lidar_manager import (
    HEADER,
    VERLEN,
    PACKET_SIZE,
    POINTS_PER_PACK,
    calc_crc8,
    parse_packet,
    LidarManager,
)

# --- Manual 3.3 Example packet ---
EXAMPLE_PACKET = bytes.fromhex(
    "542C6808AB7EE000E4DC00E2D900E5D500E3D300E4D000E9CD"
    "00E4CA00E2C700E9C500E5C200E5C000E5BE823A1A50"
)


class TestProtocolConstants:
    def test_header(self):
        assert HEADER == 0x54

    def test_verlen(self):
        assert VERLEN == 0x2C

    def test_packet_size(self):
        assert PACKET_SIZE == 47

    def test_points_per_pack(self):
        assert POINTS_PER_PACK == 12


class TestCRC8:
    def test_empty_data(self):
        assert calc_crc8(b"") == 0

    def test_example_packet_crc(self):
        assert calc_crc8(EXAMPLE_PACKET[:-1]) == 0x50

    def test_single_byte_range(self):
        result = calc_crc8(b"\x54")
        assert 0 <= result <= 255


class TestParsePacket:
    def test_example_parses(self):
        assert parse_packet(EXAMPLE_PACKET) is not None

    def test_example_speed(self):
        speed, _ = parse_packet(EXAMPLE_PACKET)
        assert speed == 2152

    def test_example_start_angle(self):
        _, points = parse_packet(EXAMPLE_PACKET)
        assert abs(points[0][0] - 324.27) < 0.1

    def test_example_end_angle(self):
        _, points = parse_packet(EXAMPLE_PACKET)
        assert abs(points[-1][0] - 334.70) < 0.1

    def test_example_point1_distance(self):
        _, points = parse_packet(EXAMPLE_PACKET)
        assert points[0][1] == 224

    def test_example_point1_intensity(self):
        _, points = parse_packet(EXAMPLE_PACKET)
        assert points[0][2] == 228

    def test_returns_12_points(self):
        _, points = parse_packet(EXAMPLE_PACKET)
        assert len(points) == 12

    def test_angles_monotonically_increasing(self):
        _, points = parse_packet(EXAMPLE_PACKET)
        angles = [p[0] for p in points]
        for i in range(1, len(angles)):
            assert angles[i] > angles[i - 1]

    def test_wrong_length(self):
        assert parse_packet(b"\x54" * 10) is None

    def test_wrong_header(self):
        bad = bytearray(EXAMPLE_PACKET)
        bad[0] = 0x00
        assert parse_packet(bytes(bad)) is None

    def test_wrong_verlen(self):
        bad = bytearray(EXAMPLE_PACKET)
        bad[1] = 0x00
        assert parse_packet(bytes(bad)) is None

    def test_corrupted_crc(self):
        bad = bytearray(EXAMPLE_PACKET)
        bad[-1] ^= 0xFF
        assert parse_packet(bytes(bad)) is None

    def test_angle_wrap_around(self):
        pkt = bytearray(EXAMPLE_PACKET)
        struct.pack_into("<H", pkt, 4, 35900)   # start: 359.00 deg
        struct.pack_into("<H", pkt, 42, 100)     # end: 1.00 deg
        pkt[-1] = calc_crc8(bytes(pkt[:-1]))

        result = parse_packet(bytes(pkt))
        assert result is not None
        _, points = result
        assert abs(points[0][0] - 359.00) < 0.1


class TestLidarManagerScanAssembly:
    """Test scan assembly logic without real hardware."""

    def test_initial_state(self):
        mgr = LidarManager(port="/dev/null")
        assert mgr.get_latest_scan() is None
        assert not mgr.is_running

    def test_finalize_scan_creates_lidar_scan(self):
        mgr = LidarManager(port="/dev/null")
        # Simulate accumulated points
        mgr._scan_buf = [(i * 10.0, 1000 + i, 200) for i in range(36)]
        mgr._speed_deg_s = 3600.0

        mgr._finalize_scan()

        scan = mgr.get_latest_scan()
        assert scan is not None
        assert len(scan.points) == 36
        assert scan.rpm == 600.0  # 3600 deg/s / 6
        assert scan.scan_count == 36

    def test_finalize_scan_converts_to_lidar_points(self):
        mgr = LidarManager(port="/dev/null")
        mgr._scan_buf = [(45.5, 2000, 150)]
        mgr._speed_deg_s = 3600.0

        mgr._finalize_scan()

        scan = mgr.get_latest_scan()
        p = scan.points[0]
        assert p.angle == 45.5
        assert p.distance == 2000.0
        assert p.quality == 150

    def test_finalize_scan_clears_buffer(self):
        mgr = LidarManager(port="/dev/null")
        mgr._scan_buf = [(0.0, 1000, 200)]
        mgr._speed_deg_s = 3600.0

        mgr._finalize_scan()
        assert mgr._scan_buf == []

    def test_finalize_empty_buffer_no_scan(self):
        mgr = LidarManager(port="/dev/null")
        mgr._finalize_scan()
        assert mgr.get_latest_scan() is None

    def test_get_stats(self):
        mgr = LidarManager(port="/dev/null")
        stats = mgr.get_stats()
        assert stats["packets"] == 0
        assert stats["crc_errors"] == 0
        assert stats["running"] is False

    def test_speed_to_rpm_conversion(self):
        mgr = LidarManager(port="/dev/null")
        mgr._scan_buf = [(0.0, 1000, 200)]

        # 10 Hz = 3600 deg/s = 600 RPM
        mgr._speed_deg_s = 3600.0
        mgr._finalize_scan()
        assert mgr.get_latest_scan().rpm == 600.0

        # 6 Hz = 2160 deg/s = 360 RPM
        mgr._scan_buf = [(0.0, 1000, 200)]
        mgr._speed_deg_s = 2160.0
        mgr._finalize_scan()
        assert mgr.get_latest_scan().rpm == 360.0
