import logging
import math
import statistics
import time
import random

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama for colored logs
init(autoreset=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our algorithm classes
from algorithms import MinMaxDivider, QuickSelector


def estimate_complexity(sizes, times):
    """
    Tries to guess the Big-O complexity by comparing time data
    with typical complexity functions like:
      1) O(n)        -> f(n) = n
      2) O(n log n)  -> f(n) = n * log(n)
      3) O(n^2)      -> f(n) = n^2
      4) O(log n)    -> f(n) = log(n)

    Returns a dict with:
      {
        'best_label': str,                 # e.g. "O(n)"
        'best_mean': float,                # mean ratio T/f(n) for that best
        'functions': {label: (mean, stdev) # details for all tested complexities
                      ...}
      }

    :param sizes: list of n-values
    :param times: list of corresponding times T(n)
    :return: dict
    """

    candidates = {
        "O(n)": lambda n: n,
        "O(n log n)": lambda n: n * math.log2(n) if n > 1 else n,
        "O(n^2)": lambda n: n**2,
        "O(log n)": lambda n: math.log2(n) if n > 1 else 1,
    }

    valid_data = [(n, t) for n, t in zip(sizes, times) if n > 0 and t > 0]
    if len(valid_data) < 2:
        return {"best_label": "Not enough data", "best_mean": None, "functions": {}}

    results = {}
    for label, f in candidates.items():
        ratios = []
        for n, t in valid_data:
            fn = f(n)
            if fn == 0:
                continue
            ratios.append(t / fn)

        if len(ratios) < 2:
            results[label] = (float("inf"), float("inf"))
            continue

        mean_val = statistics.mean(ratios)
        stdev_val = statistics.pstdev(ratios)
        results[label] = (mean_val, stdev_val)

    # Pick the candidate with the smallest stdev
    best_label = None
    best_std = float("inf")
    best_mean = None

    for label, (m, s) in results.items():
        if s < best_std:
            best_std = s
            best_label = label
            best_mean = m

    return {"best_label": best_label, "best_mean": best_mean, "functions": results}


class MinMaxManager:
    """
    Manager class for testing the MinMaxDivider algorithm with
    different data that emulate best, worst and average cases.
    """

    def __init__(self):
        self.divider = MinMaxDivider()

    def generate_data(self, data_type, size):
        """
        For MinMax, the time complexity is O(n) in all cases.
        We'll define:
         - 'best': array of identical elements,
         - 'worst': sorted array,
         - 'average': random array.
        """
        if data_type == "best":
            return [0] * size
        elif data_type == "worst":
            return list(range(size))
        else:  # 'average'
            return [random.randint(-10000, 10000) for _ in range(size)]

    def measure_time(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, (end - start)

    def analyze_time_complexity(self, n, elapsed):
        data = {"Time/n": elapsed / n if n else None}
        if n > 1:
            data["Time/(n log n)"] = elapsed / (n * math.log2(n))
        else:
            data["Time/(n log n)"] = None
        return data

    def run_tests(self):
        records = []
        data_types = ["best", "worst", "average"]
        sizes = [1, 100, 1000, 3000, 10000, 50000]

        for dt in data_types:
            for sz in sizes:
                arr = self.generate_data(dt, sz)
                (min_max_res, elapsed) = self.measure_time(
                    self.divider.find_min_max_divide_conquer, arr
                )
                complexity = self.analyze_time_complexity(sz, elapsed)

                records.append(
                    {
                        "Algorithm": "MinMaxDivider",
                        "Case": dt,
                        "Size": sz,
                        "Result": min_max_res,
                        "Time (s)": elapsed,
                        **complexity,
                    }
                )

        df = pd.DataFrame(records)
        return df

    def plot_results(self, df):
        df_mmd = df[df["Algorithm"] == "MinMaxDivider"]

        # Plot 1: Size vs Time(s)
        plt.figure(figsize=(10, 5))
        for dt in df_mmd["Case"].unique():
            subset = df_mmd[df_mmd["Case"] == dt]
            plt.plot(subset["Size"], subset["Time (s)"], marker="o", label=dt)
        plt.title("MinMaxDivider: Time(s) vs Size")
        plt.xlabel("Size")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot 2: Size vs Time/n
        plt.figure(figsize=(10, 5))
        for dt in df_mmd["Case"].unique():
            subset = df_mmd[df_mmd["Case"] == dt]
            plt.plot(subset["Size"], subset["Time/n"], marker="o", label=dt)
        plt.title("MinMaxDivider: Time/n vs Size")
        plt.xlabel("Size")
        plt.ylabel("Time/n")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class QuickSelectManager:
    """
    Manager class for testing the QuickSelector algorithm with
    data that emulate best, worst, and average cases.
    """

    def __init__(self):
        self.qs = QuickSelector()

    def generate_data(self, data_type, size):
        """
        - best: distinct random -> hoping pivot ~ median => O(n).
        - worst: sorted -> pivot often min or max => O(n^2).
        - average: random -> ~O(n).
        """
        if data_type == "best":
            # Use choices instead of sample, or expand range to avoid errors
            if size <= 200000:
                # Enough range for distinct
                return random.sample(range(-200000, 200001), size)
            else:
                # fallback if size > 400k
                return random.choices(range(-200000, 200001), k=size)
        elif data_type == "worst":
            return list(range(size))
        else:  # 'average'
            return [random.randint(-10000, 10000) for _ in range(size)]

    def measure_time(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, (end - start)

    def analyze_time_complexity(self, n, elapsed):
        data = {"Time/n": elapsed / n if n else None}
        if n > 1:
            data["Time/(n log n)"] = elapsed / (n * math.log2(n))
        else:
            data["Time/(n log n)"] = None
        return data

    def run_tests(self):
        records = []
        data_types = ["best", "worst", "average"]
        sizes = [1, 100, 1000, 3000, 10000, 50000]
        k = 3

        for dt in data_types:
            for sz in sizes:
                if sz < k:
                    continue
                arr = self.generate_data(dt, sz)
                (kth_value, elapsed) = self.measure_time(self.qs.quick_select, arr, k)
                complexity = self.analyze_time_complexity(sz, elapsed)

                records.append(
                    {
                        "Algorithm": "QuickSelector",
                        "Case": dt,
                        "Size": sz,
                        "k": k,
                        "k-th Smallest": kth_value,
                        "Time (s)": elapsed,
                        **complexity,
                    }
                )

        df = pd.DataFrame(records)
        return df

    def plot_results(self, df):
        df_qs = df[df["Algorithm"] == "QuickSelector"]

        # Plot 1: Size vs Time (s)
        plt.figure(figsize=(10, 5))
        for dt in df_qs["Case"].unique():
            subset = df_qs[df_qs["Case"] == dt]
            plt.plot(subset["Size"], subset["Time (s)"], marker="o", label=dt)
        plt.title("QuickSelector: Time(s) vs. Size")
        plt.xlabel("Size")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot 2: Size vs Time/n
        plt.figure(figsize=(10, 5))
        for dt in df_qs["Case"].unique():
            subset = df_qs[df_qs["Case"] == dt]
            plt.plot(subset["Size"], subset["Time/n"], marker="o", label=dt)
        plt.title("QuickSelector: Time/n vs. Size")
        plt.xlabel("Size")
        plt.ylabel("Time/n")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class MainManager:
    """
    Main class that orchestrates both MinMaxManager and QuickSelectManager.
    Also integrates an empirical complexity estimation step.
    """

    def __init__(self):
        self.minmax_mgr = MinMaxManager()
        self.quick_mgr = QuickSelectManager()

    def plot_complexity_overlay(
        self, sizes_list, times_list, complexity_info, case_label, algo_name
    ):
        """
        Plots actual time points alongside the fitted complexity curve
        determined by 'best_label' from complexity_info.
        """

        best_label = complexity_info["best_label"]
        best_mean = complexity_info["best_mean"]

        # If not enough data or no best_label
        if best_label == "Not enough data" or (best_mean is None):
            logger.warning("Skipping overlay plot: not enough data or no best_label.")
            return

        # We'll define the function according to the best_label
        func_map = {
            "O(n)": lambda n: n,
            "O(n log n)": lambda n: n * math.log2(n) if n > 1 else n,
            "O(n^2)": lambda n: n**2,
            "O(log n)": lambda n: math.log2(n) if n > 1 else 1,
        }

        if best_label not in func_map:
            logger.warning(
                f"Skipping overlay plot: best_label={best_label} not recognized."
            )
            return

        f = func_map[best_label]

        # Build predicted times: predicted_time[i] = best_mean * f(sizes_list[i])
        predicted_times = []
        for n in sizes_list:
            val = f(n)
            predicted_times.append(best_mean * val)

        plt.figure(figsize=(8, 5))
        plt.plot(sizes_list, times_list, "o-", label="Actual time")
        plt.plot(sizes_list, predicted_times, "r--", label=f"Fitted {best_label}")
        plt.title(f"{algo_name}, case='{case_label}': Actual vs {best_label}")
        plt.xlabel("Size")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def finalize_analysis(self, df, algo_name):
        """
        For each 'Case' in the DataFrame, collect (Size, Time(s))
        and call 'estimate_complexity' to guess the empirical O(...)
        Then plot an overlay of actual data vs. fitted curve.
        """
        subset = df[df["Algorithm"] == algo_name]
        cases = subset["Case"].unique()

        for c in cases:
            sub = subset[subset["Case"] == c].sort_values("Size")
            sizes_list = sub["Size"].tolist()
            times_list = sub["Time (s)"].tolist()

            # 1) Determine complexity
            info = estimate_complexity(sizes_list, times_list)
            best_label = info["best_label"]

            logger.info(
                Fore.YELLOW
                + f"{algo_name}, case='{c}': empirical complexity guess => {best_label}"
                + Style.RESET_ALL
            )

            # 2) Plot actual vs. fitted
            self.plot_complexity_overlay(sizes_list, times_list, info, c, algo_name)

    def main(self):
        logger.info(
            Fore.MAGENTA + "Starting tests for MinMaxDivider..." + Style.RESET_ALL
        )
        df_minmax = self.minmax_mgr.run_tests()
        logger.info(Fore.GREEN + "MinMaxDivider tests completed." + Style.RESET_ALL)

        # Show the DataFrame in a nice tabular format
        logger.info(Fore.CYAN + "\n--- MinMaxDivider Results ---" + Style.RESET_ALL)
        print(
            tabulate(
                df_minmax,
                headers="keys",
                tablefmt="fancy_grid",
                showindex=False,
                floatfmt=".6f",
            )
        )

        # Plot standard results
        self.minmax_mgr.plot_results(df_minmax)

        # Estimate complexity and overlay
        self.finalize_analysis(df_minmax, "MinMaxDivider")

        logger.info(
            Fore.MAGENTA + "Starting tests for QuickSelector..." + Style.RESET_ALL
        )
        df_quick = self.quick_mgr.run_tests()
        logger.info(Fore.GREEN + "QuickSelector tests completed." + Style.RESET_ALL)

        logger.info(Fore.CYAN + "\n--- QuickSelector Results ---" + Style.RESET_ALL)
        print(
            tabulate(
                df_quick,
                headers="keys",
                tablefmt="fancy_grid",
                showindex=False,
                floatfmt=".6f",
            )
        )

        self.quick_mgr.plot_results(df_quick)

        # Estimate complexity and overlay
        self.finalize_analysis(df_quick, "QuickSelector")

        logger.info(Fore.MAGENTA + "All tests finished." + Style.RESET_ALL)


if __name__ == "__main__":
    app = MainManager()
    app.main()
