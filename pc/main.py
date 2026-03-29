"""PC-side entry point for JetRacer autonomous driving.

Connects to the Jetson server, receives sensor data,
and provides a control interface.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from pc.jetson_client.client import JetsonClient
from pc.jetson_client.sensor_receiver import SensorReceiver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load driving configuration."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "driving_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


async def run_pc(config: dict):
    """Main PC application loop."""
    jetson_cfg = config.get("jetson", {})

    client = JetsonClient(
        jetson_host=jetson_cfg.get("host", "jetson"),
        http_port=jetson_cfg.get("http_port", 8000),
    )
    receiver = SensorReceiver()
    client.on_message(receiver.handle_message)

    # Connect to Jetson
    try:
        await client.connect()
        logger.info("Connected to Jetson server")
    except Exception as e:
        logger.error(f"Failed to connect to Jetson: {e}")
        logger.info("Running in offline mode (no Jetson connection)")
        return

    # Main monitoring loop
    try:
        while True:
            await asyncio.sleep(2.0)

            stats = receiver.get_stats()
            status = receiver.get_vehicle_status()

            logger.info(
                f"[Monitor] connected={stats['connected']} "
                f"frames={stats['frame_count']} "
                f"scans={stats['scan_count']} "
                f"cameras={stats['active_cameras']} "
                f"vehicle={'OK' if status.connected else 'N/A'} "
                f"steering={status.steering:.2f} throttle={status.throttle:.2f}"
            )
    except asyncio.CancelledError:
        pass
    finally:
        await client.send_emergency_stop(reason="pc_shutdown")
        await client.disconnect()
        logger.info("PC shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="JetRacer PC Controller")
    parser.add_argument("--config", type=str, help="Path to driving_config.yaml")
    parser.add_argument("--jetson-host", type=str, help="Jetson hostname/IP")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.jetson_host:
        config.setdefault("jetson", {})["host"] = args.jetson_host

    loop = asyncio.new_event_loop()

    def shutdown():
        for task in asyncio.all_tasks(loop):
            task.cancel()

    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    try:
        loop.run_until_complete(run_pc(config))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
