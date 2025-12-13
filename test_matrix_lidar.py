#!/usr/bin/env python3
"""DFRobot Matrix LiDAR (SEN0628) テストスクリプト

DFRobotプロトコルを使用してセンサーと通信テスト
"""

import smbus2 as smbus
import time

BUS_NUM = 7
ADDR = 0x33

# DFRobotプロトコル定数
CMD_SETMODE = 1
CMD_ALL_DATA = 2
CMD_FIXED_POINT = 3

MATRIX_4X4 = 16
MATRIX_8X8 = 64

STATUS_SUCCESS = 0x53
STATUS_FAILED = 0x63

def send_packet(bus, pkt):
    """パケット送信 (0x55レジスタへ)"""
    try:
        bus.write_i2c_block_data(ADDR, 0x55, pkt)
        return True
    except Exception as e:
        print(f"Send error: {e}")
        return False

def recv_bytes(bus, count):
    """バイト受信"""
    result = []
    for _ in range(count):
        try:
            result.append(bus.read_byte(ADDR))
        except:
            result.append(0xFF)
    return result

def recv_packet(bus, expected_cmd, timeout=2.0):
    """レスポンスパケット受信"""
    start = time.time()
    
    while time.time() - start < timeout:
        status = recv_bytes(bus, 1)[0]
        
        if status == 0xFF:
            time.sleep(0.07)
            continue
        
        if status in [STATUS_SUCCESS, STATUS_FAILED]:
            cmd = recv_bytes(bus, 1)[0]
            len_bytes = recv_bytes(bus, 2)
            length = len_bytes[0] | (len_bytes[1] << 8)
            
            print(f"  Status: 0x{status:02X} ({'SUCCESS' if status == STATUS_SUCCESS else 'FAILED'})")
            print(f"  Command: {cmd}")
            print(f"  Data length: {length}")
            
            if length > 0 and length < 256:
                data = recv_bytes(bus, length)
                return status, cmd, data
            return status, cmd, []
        
        time.sleep(0.07)
    
    print("  Timeout!")
    return None, None, []

def test_set_mode(bus, mode):
    """モード設定テスト"""
    mode_name = "8x8" if mode == MATRIX_8X8 else "4x4"
    print(f"\n=== Setting {mode_name} mode ===")
    
    # パケット構築: [length_h, length_l, cmd, 0, 0, 0, mode]
    pkt = [0x00, 0x05, CMD_SETMODE, 0x00, 0x00, 0x00, mode]
    
    print(f"Sending: {[hex(b) for b in pkt]}")
    if send_packet(bus, pkt):
        time.sleep(0.1)
        status, cmd, data = recv_packet(bus, CMD_SETMODE)
        if status == STATUS_SUCCESS:
            print(f"Mode set to {mode_name} successfully!")
            time.sleep(0.5)  # モード切替待ち
            return True
    return False

def test_get_all_data(bus):
    """全データ取得テスト"""
    print(f"\n=== Getting all distance data ===")
    
    # パケット構築: [length_h, length_l, cmd]
    pkt = [0x00, 0x01, CMD_ALL_DATA]
    
    print(f"Sending: {[hex(b) for b in pkt]}")
    if send_packet(bus, pkt):
        time.sleep(0.1)
        status, cmd, data = recv_packet(bus, CMD_ALL_DATA)
        if status == STATUS_SUCCESS and data:
            print(f"Received {len(data)} bytes of data")
            return data
    return []

def test_get_fixed_point(bus, x, y):
    """指定座標の距離取得テスト"""
    print(f"\n=== Getting distance at ({x}, {y}) ===")
    
    # パケット構築: [length_h, length_l, cmd, x, y]
    pkt = [0x00, 0x03, CMD_FIXED_POINT, x, y]
    
    print(f"Sending: {[hex(b) for b in pkt]}")
    if send_packet(bus, pkt):
        time.sleep(0.1)
        status, cmd, data = recv_packet(bus, CMD_FIXED_POINT)
        if status == STATUS_SUCCESS and len(data) >= 2:
            distance = data[0] | (data[1] << 8)
            print(f"Distance at ({x}, {y}): {distance} mm")
            return distance
    return -1

def parse_8x8_data(raw_data):
    """8x8データをパース"""
    if len(raw_data) < 128:
        print(f"Warning: Expected 128 bytes, got {len(raw_data)}")
        return None
    
    grid = [[0] * 8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            idx = (i * 8 + j) * 2
            grid[i][j] = raw_data[idx] | (raw_data[idx + 1] << 8)
    return grid

def print_grid(grid):
    """グリッドを表示"""
    print("\nDistance Grid (mm):")
    print("    X0    X1    X2    X3    X4    X5    X6    X7")
    for i, row in enumerate(grid):
        print(f"Y{i}: ", end="")
        for val in row:
            if val == 0 or val >= 4000:
                print(" ---- ", end="")
            else:
                print(f"{val:5d} ", end="")
        print()

def main():
    print("=" * 60)
    print("DFRobot Matrix LiDAR (SEN0628) Test")
    print("=" * 60)
    print(f"Bus: {BUS_NUM}, Address: 0x{ADDR:02X}")
    
    try:
        bus = smbus.SMBus(BUS_NUM)
        
        # デバイス存在確認
        print("\n=== Checking device ===")
        try:
            bus.read_byte(ADDR)
            print(f"Device found at 0x{ADDR:02X}")
        except Exception as e:
            print(f"Device not found: {e}")
            return
        
        # 8x8モード設定
        if not test_set_mode(bus, MATRIX_8X8):
            print("Failed to set mode, trying to read anyway...")
        
        # 全データ取得
        raw_data = test_get_all_data(bus)
        if raw_data:
            grid = parse_8x8_data(raw_data)
            if grid:
                print_grid(grid)
                
                # 統計
                all_vals = [v for row in grid for v in row if 0 < v < 4000]
                if all_vals:
                    print(f"\nStatistics:")
                    print(f"  Min: {min(all_vals)} mm")
                    print(f"  Max: {max(all_vals)} mm")
                    print(f"  Avg: {sum(all_vals)/len(all_vals):.1f} mm")
                    print(f"  Valid points: {len(all_vals)}/64")
        
        # 個別ポイント取得テスト
        test_get_fixed_point(bus, 0, 0)
        test_get_fixed_point(bus, 3, 3)
        test_get_fixed_point(bus, 7, 7)
        
        bus.close()
        print("\n=== Test complete ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
