import logging
import random
from colorama import init, Fore, Style

# Initialize colorama for colored logs
init(autoreset=True)

logger = logging.getLogger(__name__)


class MinMaxDivider:
    """
    This class implements the 'Divide and Conquer' approach
    to find the minimum and maximum elements in an array.
    """

    def find_min_max_divide_conquer(self, arr):
        """
        Returns a tuple (min_val, max_val) using recursion.
        Average Time Complexity: O(n)
        """
        n = len(arr)
        logger.debug(Fore.BLUE + f"Called find_min_max_divide_conquer with array: {arr}" + Style.RESET_ALL)

        # Base cases
        if n == 1:
            return arr[0], arr[0]
        if n == 2:
            return min(arr[0], arr[1]), max(arr[0], arr[1])

        mid = n // 2
        left_min, left_max = self.find_min_max_divide_conquer(arr[:mid])
        right_min, right_max = self.find_min_max_divide_conquer(arr[mid:])

        return min(left_min, right_min), max(left_max, right_max)


class QuickSelector:
    """
    This class implements the Quick Select algorithm
    to find the k-th smallest element in an array.
    """

    def quick_select(self, arr, k):
        """
        Returns the k-th smallest element (1-based index).
        Average Time Complexity: O(n)
        Worst-case Time Complexity: O(n^2)
        """
        logger.debug(Fore.BLUE + f"Called quick_select with arr={arr}, k={k}" + Style.RESET_ALL)

        # Base case
        if len(arr) == 1:
            return arr[0]

        pivot = random.choice(arr)

        left_part = [x for x in arr if x < pivot]
        right_part = [x for x in arr if x > pivot]
        pivot_count = arr.count(pivot)
        left_size = len(left_part)

        if left_size >= k:
            return self.quick_select(left_part, k)
        elif left_size < k <= left_size + pivot_count:
            return pivot
        else:
            return self.quick_select(right_part, k - left_size - pivot_count)
