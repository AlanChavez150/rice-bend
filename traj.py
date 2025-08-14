import argparse
import logging
import time
from typing import Tuple

import coloredlogs
import numpy as np
import scipy
import scipy.integrate
from matplotlib import pyplot as plt

class Traj():
    def __init__(self, accuracy: float, res: int = 1000):
        self.log = logging.getLogger()
        self.freq = 150e9
        self.wave_number = 2 * np.pi / (scipy.constants.c / self.freq)
        size = 0.3
        self.x_min = -1.0 * (size / 2.0)
        self.x_max =  1.0 * (size / 2.0)
        self.z_min = 0.0
        self.z_max = 0.5
        self.res = res
        self.x_axis = np.linspace(self.x_min, self.x_max, self.res)
        self.accuracy = accuracy

    def gen_phase_plate(self, a: float, b: float, c: float):
        z = np.linspace(0, self.z_max, self.res)
        caustic = (a * z**2) + (b * z) + c
        d_caustic = 2*a*z + b

        # compute y = c(z) - z * dc(z)
        y = caustic - z * d_caustic

        dphi_dy = (self.wave_number * d_caustic) / np.sqrt(1 + d_caustic**2)
        sort_idx = np.argsort(y)
        y_sorted = y[sort_idx]
        dphi_dy_sorted = dphi_dy[sort_idx]

        phi = scipy.integrate.cumtrapz(dphi_dy_sorted, y_sorted, initial=0)
        return (y_sorted, phi)

    def search_traj(self, search_profile: np.ndarray):
        search_space = np.linspace(
            start=-1,
            stop=1,
            num=self.accuracy
        )
        min_a = None
        min_b = None
        min_c = None
        min_score = np.inf
        min_plate = None
        min_axis = None
        start_time = time.time()
        total_cnt = len(search_space) ** 3
        attempt_cnt = 0
        for search_a in search_space:
            for search_b in search_space:
                search_c = 0.0
                curr_axis, curr_plate = self.gen_phase_plate(search_a, search_b, search_c)
                score = np.sum(np.abs(np.diff(curr_plate - search_profile)))

                attempt_cnt += 1
                if score < min_score:
                    min_score = score
                    min_a = search_a
                    min_b = search_b
                    min_c = search_c
                    min_plate = curr_plate
                    min_axis = curr_axis
                    complete_perc = (attempt_cnt+1)/total_cnt
                    complete_perc *= 100
                    #self.log.info(f"{complete_perc:02.2f}% a {search_a:03.3f} b {search_b:03.3f} {search_c:03.3f}: score {score:0.3f}")


        dur = time.time() - start_time
        self.log.info(f"Finished after {dur}")
        self.log.info(f"Attempts: {attempt_cnt}")
        self.log.info(f"A {min_a:0.3f}")
        self.log.info(f"B {min_b:0.3f}")
        self.log.info(f"C {min_c:0.3f}")
        return (min_axis, min_plate, min_a, min_b, min_c)

    def plot_traj(
            self,
            search_axis: np.ndarray,
            search_profile: np.ndarray,
            found_axis: np.ndarray,
            found_profile: np.ndarray,
            real_traj: Tuple[float, float, float],
            traj: Tuple[float, float, float]
        ):
        self.fig = plt.figure(figsize=(10, 10), layout="constrained")
        self.num_rows = 2
        self.num_cols = 2
        self.next_plot_index = 1

        mod_ax = self.fig.add_subplot(self.num_rows, self.num_cols, self.next_plot_index)
        self.next_plot_index += 1
        mod_ax.set_title("Phase Plate Comparison")
        mod_ax.set_xlabel("x (m)")
        mod_ax.set_ylabel("Phase Plate Wrapped [rad]")
        mod_ax.set_xlim(self.x_min, self.x_max)
        mod_ax.grid(True)

        mod_ax.plot(
            search_axis,
            np.mod(search_profile, 2*np.pi),
            label="real"
        )
        mod_ax.plot(
            found_axis,
            np.mod(found_profile, 2*np.pi),
            "--",
            label=f"Found phase plate"
        )
        mod_ax.legend()

        unwrap_ax = self.fig.add_subplot(self.num_rows, self.num_cols, self.next_plot_index)
        self.next_plot_index += 1
        unwrap_ax.set_title("Phase Plate comparison")
        unwrap_ax.set_xlabel("x (m)")
        unwrap_ax.set_ylabel("Phase unwrapped [rad]")
        unwrap_ax.set_xlim(self.x_min, self.x_max)
        unwrap_ax.grid(True)

        unwrap_ax.plot(
            search_axis,
            search_profile,
            label="real"
        )
        unwrap_ax.plot(
            found_axis,
            found_profile,
            "--",
            label=f"Found phase plate"
        )
        unwrap_ax.legend()

        traj_ax = self.fig.add_subplot(self.num_rows, self.num_cols, self.next_plot_index)
        self.next_plot_index += 1
        traj_ax.set_title("Trajectory comparison")
        traj_ax.set_xlabel("X (m)")
        traj_ax.set_ylabel("Z (m)")
        traj_ax.set_xlim(self.x_min, self.x_max)
        traj_ax.set_ylim(self.z_min, self.z_max)
        traj_ax.grid(True)

        real_a, real_b, real_c = real_traj
        found_a, found_b, found_c = traj
        x_axis = np.linspace(self.x_min, self.x_max, self.res)
        z_axis = np.linspace(self.x_min, self.x_max, self.res)
        real_traj = (- real_b + np.sqrt( real_b**2 - 4* real_a * ( real_c - z_axis))) / (2 *  real_a)
        pred_traj = (-found_b + np.sqrt(found_b**2 - 4*found_a * (found_c - z_axis))) / (2 * found_a)

        traj_ax.plot(
            x_axis,
            real_traj,
            label="Real Trajectory"
        )
        traj_ax.plot(
            x_axis,
            pred_traj,
            "--",
            label="Computing Trajectory"
        )
        unwrap_ax.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "A",
        type=float,
        help="Ax^2 + bx + c"
    )
    parser.add_argument(
        "B",
        type=float,
        help="Ax^2 + bx + c"
    )
    parser.add_argument(
        "C",
        type=float,
        help="Ax^2 + bx + c"
    )
    parser.add_argument(
        "--accuracy", "-ac",
        type=int,
        help="",
        default=100
    )
    args = parser.parse_args()

    fmt = "%(levelname)s: %(message)s"
    coloredlogs.install(level="INFO", fmt=fmt)

    traj = Traj(args.accuracy)
    real_axis, real_plate = traj.gen_phase_plate(args.A, args.B, args.C)
    f_axis, f_p, f_a, f_b, f_c = traj.search_traj(real_plate)
    traj.plot_traj(
        real_axis,
        real_plate,
        f_axis,
        f_p,
        (args.A, args.B, args.C),
        (f_a, f_b, f_c)
    )