from __future__ import annotations
import time
import argparse
import logging
import signal
import sys
from typing import Optional, List
from datetime import datetime, timezone
import httpx

from app.simulator.generator import SyntheticSensorGenerator
from app.simulator.fault_injector import FaultInjector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FusionHeal-Simulator")


class TelemetrySimulatorRunner:
    """Continuously runs synthetic industrial telemetry generation and transmits to the ingestion API."""

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000/api/telemetry",
        interval_seconds: float = 1.0,
        assets: Optional[List[str]] = None,
        sensors: Optional[List[str]] = None,
        fault_scenario: str = "normal",
        fault_after_seconds: int = 5,
        random_seed: int = 42,
    ):
        self.api_url = api_url
        self.interval_seconds = interval_seconds
        self.assets = assets or ["boiler", "pump"]
        self.sensors = sensors or ["boiler_01:temperature_01", "pump_01:vibration_01"]
        self.fault_scenario = fault_scenario
        self.fault_after_seconds = fault_after_seconds
        self.generator = SyntheticSensorGenerator(random_seed=random_seed)
        self.injector = FaultInjector()
        self.running = False

    def stop(self):
        """Signal the simulator to stop gracefully."""
        self.running = False
        logger.info("Stopping telemetry simulator gracefully...")

    def run(self, duration_seconds: Optional[int] = None) -> int:
        """Execute the simulator loop.

        Args:
            duration_seconds: Max seconds to run; if None, runs indefinitely until interrupted.

        Returns:
            Total count of readings generated and transmitted.
        """
        self.running = True
        start_time = time.time()
        readings_sent = 0

        logger.info(
            f"Starting FusionHeal Telemetry Simulator -> Target: {self.api_url} "
            f"(Interval: {self.interval_seconds}s, Assets: {self.assets}, Fault: '{self.fault_scenario}')"
        )

        with httpx.Client(timeout=5.0) as client:
            while self.running:
                now_utc = datetime.now(timezone.utc)
                elapsed = time.time() - start_time

                # Check duration limit
                if duration_seconds is not None and elapsed >= duration_seconds:
                    logger.info(f"Reached configured duration of {duration_seconds} seconds.")
                    break

                # Determine active scenario for this tick
                active_scenario = "normal"
                if elapsed >= self.fault_after_seconds:
                    active_scenario = self.fault_scenario

                readings_to_send = []

                # 1. Boiler Telemetry (Asset A: Industrial Boiler - Temperature in C)
                if "boiler" in self.assets and "boiler_01:temperature_01" in self.sensors:
                    boiler_rd, clean_temp = self.generator.generate_boiler_temperature(timestamp=now_utc)
                    mod_boiler, _ = self.injector.inject_fault(boiler_rd, clean_temp, active_scenario)
                    if mod_boiler is not None:
                        readings_to_send.append(mod_boiler.model_dump(mode="json"))

                # 2. Pump Telemetry (Asset B: Industrial Pump - Vibration in m/s2)
                if "pump" in self.assets and "pump_01:vibration_01" in self.sensors:
                    pump_rd, clean_vib = self.generator.generate_pump_vibration(timestamp=now_utc)
                    mod_pump, _ = self.injector.inject_fault(pump_rd, clean_vib, active_scenario)
                    if mod_pump is not None:
                        readings_to_send.append(mod_pump.model_dump(mode="json"))

                # Transmit batch to ingestion API
                if readings_to_send:
                    try:
                        resp = client.post(self.api_url, json={"readings": readings_to_send})
                        if resp.status_code in (200, 201):
                            readings_sent += len(readings_to_send)
                            logger.info(
                                f"Sent {len(readings_to_send)} reading(s) (scenario='{active_scenario}'). "
                                f"Status: {resp.status_code}"
                            )
                        else:
                            logger.warning(f"API returned status {resp.status_code}: {resp.text}")
                    except httpx.ConnectError:
                        logger.error(
                            f"Could not connect to {self.api_url}. Is the FusionHeal FastAPI service running?"
                        )
                    except Exception as exc:
                        logger.error(f"Transmission error: {exc}")

                # Sleep until next sampling tick
                time.sleep(self.interval_seconds)

        logger.info(f"Simulator completed. Total readings transmitted: {readings_sent}.")
        return readings_sent


def main():
    parser = argparse.ArgumentParser(description="FusionHeal Continuous Telemetry Simulator")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/telemetry", help="Target Ingestion API URL")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--duration", type=int, default=None, help="Simulation duration in seconds (optional)")
    parser.add_argument("--fault", default="normal", help="Fault scenario (normal, missing_message, missing_value, frozen, spike, drift, invalid, gap, duplicate, out_of_order)")
    parser.add_argument("--fault-after", type=int, default=5, help="Seconds before injecting configured fault")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--assets", default="boiler,pump", help="Comma-separated assets to simulate (boiler,pump)")
    parser.add_argument("--sensors", default="boiler_01:temperature_01,pump_01:vibration_01", help="Comma-separated sensors to simulate")
    args = parser.parse_args()

    assets_list = [a.strip() for a in args.assets.split(",") if a.strip()]
    sensors_list = [s.strip() for s in args.sensors.split(",") if s.strip()]

    runner = TelemetrySimulatorRunner(
        api_url=args.url,
        interval_seconds=args.interval,
        assets=assets_list,
        sensors=sensors_list,
        fault_scenario=args.fault,
        fault_after_seconds=args.fault_after,
        random_seed=args.seed,
    )

    def handle_sigint(sig, frame):
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    runner.run(duration_seconds=args.duration)


if __name__ == "__main__":
    main()
