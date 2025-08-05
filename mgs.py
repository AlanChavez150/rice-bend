import logging
import argparse

import coloredlogs
import scipy.constants

from sim_scene import SimAperature, SimScene

class MGS():
    def __init__(self, freq: float):
        self.log = logging.getLogger()
        # set up simulation scene
        x_min = -0.2
        x_max =  0.2
        self.freq = 150e9
        self.wavelength = scipy.constants.c / freq
        rx_spacing_ratio = 1 / 2.0
        rx_spacing = self.wavelength * rx_spacing_ratio

        rx = SimAperature(
            x_min=x_min,
            x_max=x_max,
            z=0.25,
            dx=rx_spacing
        )
        tx = SimAperature(
            x_min=x_min,
            x_max=0,
            z=0,
            dx=rx_spacing
        )
        tx.make_steer(self.freq, theta_deg=30)

        self.scene = SimScene(
            x_min=x_min,
            x_max=0.2,
            z_min=0.0,
            z_max=0.5,
            spacing=1e-6,
            rx_ap=rx,
            tx_ap=tx
        )

        # Log parameters
        self.log.info(f"Simulation scene:")
        self.log.info(f" - X axis: {self.scene.x_min:0.3f} - {self.scene.x_max:0.3f}")
        self.log.info(f" - Z axis: {self.scene.z_min:0.3f} - {self.scene.z_max:0.3f}")
        self.log.info(f"RX aperature:")
        rx_s_num, rx_s_den = rx_spacing_ratio.as_integer_ratio()
        self.log.info(f" - X axis: {self.scene.rx_ap.x_min:0.3f} {self.scene.rx_ap.x_max:0.3f}. {rx_s_num}/{rx_s_den} wavelength")
        self.log.info(f" - Z: {self.scene.rx_ap.z:0.3f}")
        self.log.info(f"TX aperature")
        self.log.info(f" - X axis: {self.scene.tx_ap.x_min:0.3f} {self.scene.tx_ap.x_max:0.3f}. {rx_s_num}/{rx_s_den} wavelength")
        self.log.info(f" - Z: {self.scene.tx_ap.z:0.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to simulate phase retrieval using a modified gerchberg-saxton algorithm"
    )
    parser.add_argument(
        "--debug",
        help="Enables debug logs",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--freq", "-f",
        help="Frequency in Hz",
        type=float,
        default=150e9
    )
    args = parser.parse_args()

    fmt = "%(levelname)s: %(message)s"
    level = "INFO"
    if bool(args.debug): level = "DEBUG"
    coloredlogs.install(level=level, fmt=fmt)

    mgs = MGS(args.freq)