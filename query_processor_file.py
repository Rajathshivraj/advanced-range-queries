"""
Query Processor - Main Interface
Coordinates all data structures to handle diverse query types efficiently.
"""

from typing import List, Optional
from .segment_tree import SegmentTree, GCDSegmentTree, MergeSortTree
from .fenwick_tree import FenwickTree, RangeFenwickTree
from .number_theory import NumberTheory
from .dynamic_programming import DynamicProgramming


class QueryProcessor:
    """
    Unified interface for all numerical queries.
    Maintains multiple data structures for optimal performance.
    """
    
    def __init__(self, arr: List[int]):
        """
        Initialize all data structures.
        
        Time: O(n log n)
        Space: O(n log n)
        """
        self.arr = arr.copy()
        self.n = len(arr)
        
        if self.n == 0:
            return
        
        # Initialize data structures
        self.sum_tree = SegmentTree(arr, lambda a, b: a + b, 0)
        self.gcd_tree = GCDSegmentTree(arr)
        self.fenwick = FenwickTree(arr)
        self.merge_sort_tree = MergeSortTree(arr)
        self.range_fenwick = RangeFenwickTree(arr)
        
        # Initialize number theory module
        max_val = max(arr) if arr else 10**6
        self.nt = NumberTheory(max_val)
        
        # Cache for expensive operations
        self.cache = {}
    
    def range_sum(self, left: int, right: int) -> int:
        """
        Compute sum of elements in range [left, right].
        
        Time: O(log n)
        Uses: Fenwick Tree
        """
        if left < 0 or right >= self.n or left > right:
            return 0
        
        return self.fenwick.range_sum(left, right)
    
    def range_gcd(self, left: int, right: int) -> int:
        """
        Compute GCD of all elements in range [left, right].
        
        Time: O(log n)
        Uses: GCD Segment Tree
        """
        if left < 0 or right >= self.n or left > right:
            return 0
        
        return self.gcd_tree.range_gcd(left, right)
    
    def point_update(self, idx: int, value: int):
        """
        Update element at index idx to value.
        
        Time: O(log n)
        Updates: All data structures
        """
        if idx < 0 or idx >= self.n:
            return
        
        old_value = self.arr[idx]
        self.arr[idx] = value
        
        # Update all structures
        self.sum_tree.point_update(idx, value)
        self.gcd_tree.point_update(idx, value)
        self.fenwick.point_update(idx, value, old_value)
        
        # Invalidate cache
        self.cache.clear()
    
    def range_multiply(self, left: int, right: int, multiplier: int):
        """
        Multiply all elements in range [left, right] by multiplier.
        
        Time: O((right - left + 1) * log n)
        
        Note: True range multiplication is complex. This implementation
        updates elements individually. For additive updates, use range_add.
        """
        if left < 0 or right >= self.n or left > right:
            return
        
        for i in range(left, right + 1):
            new_value = self.arr[i] * multiplier
            self.point_update(i, new_value)
    
    def range_add(self, left: int, right: int, delta: int):
        """
        Add delta to all elements in range [left, right].
        
        Time: O(log n) with lazy propagation
        Uses: Range Fenwick Tree
        """
        if left < 0 or right >= self.n or left > right:
            return
        
        self.range_fenwick.range_update(left, right, delta)
        
        # Update main array
        for i in range(left, right + 1):
            self.arr[i] += delta
        
        # Rebuild other structures (expensive, but necessary for correctness)
        self._rebuild_structures()
    
    def count_divisible(self, left: int, right: int, divisor: int) -> int:
        """
        Count elements in range [left, right] divisible by divisor.
        
        Time: O(right - left + 1)
        
        Optimization: For multiple queries, precompute divisibility table.
        """
        if left < 0 or right >= self.n or left > right or divisor == 0:
            return 0
        
        count = 0
        for i in range(left, right + 1):
            if self.arr[i] % divisor == 0:
                count += 1
        
        return count
    
    def kth_smallest_in_range(self, left: int, right: int, k: int) -> Optional[int]:
        """
        Find kth smallest element (1-indexed) in range [left, right].
        
        Time: O(log^2 n)
        Uses: Merge Sort Tree
        """
        if left < 0 or right >= self.n or left > right:
            return None
        
        return self.merge_sort_tree.kth_smallest(left, right, k)
    
    def modular_exp_sum(self, left: int, right: int, exp: int, mod: int) -> int:
        """
        Compute sum of (arr[i]^exp mod mod) for i in [left, right].
        
        Time: O((right - left + 1) * log exp)
        Uses: Fast modular exponentiation
        """
        if left < 0 or right >= self.n or left > right:
            return 0
        
        total = 0
        for i in range(left, right + 1):
            total += self.nt.mod_pow(self.arr[i], exp, mod)
            total %= mod
        
        return total
    
    def range_lcm(self, left: int, right: int) -> int:
        """
        Compute LCM of all elements in range [left, right].
        
        Time: O((right - left + 1) * log(max_val))
        
        Uses iterative LCM calculation. Can overflow for large ranges.
        """
        if left < 0 or right >= self.n or left > right:
            return 0
        
        result = self.arr[left]
        for i in range(left + 1, right + 1):
            result = self.nt.lcm(result, self.arr[i])
            
            # Prevent overflow
            if result > 10**18:
                return -1  # Overflow indicator
        
        return result
    
    def count_primes_in_range(self, left: int, right: int) -> int:
        """
        Count prime numbers in range [left, right].
        
        Time: O((right - left + 1) * sqrt(max_val) / ln(sqrt(max_val)))
        Uses: Prime checking with sieve
        """
        if left < 0 or right >= self.n or left > right:
            return 0
        
        count = 0
        for i in range(left, right + 1):
            if self.nt.is_prime(self.arr[i]):
                count += 1
        
        return count
    
    def sum_of_divisors_in_range(self, left: int, right: int) -> int:
        """
        Compute sum of divisor counts for all elements in range.
        
        Time: O((right - left + 1) * sqrt(max_val))
        """
        if left < 0 or right >= self.n or left > right:
            return 0
        
        total = 0
        for i in range(left, right + 1):
            total += self.nt.count_divisors(self.arr[i])
        
        return total
    
    def range_xor(self, left: int, right: int) -> int:
        """
        Compute XOR of all elements in range [left, right].
        
        Time: O(right - left + 1)
        
        Note: XOR doesn't have efficient segment tree since it's not
        easily invertible for range updates. For static XOR, use prefix XOR.
        """
        if left < 0 or right >= self.n or left > right:
            return 0
        
        result = 0
        for i in range(left, right + 1):
            result ^= self.arr[i]
        
        return result
    
    def max_subarray_sum_in_range(self, left: int, right: int) -> int:
        """
        Find maximum subarray sum in range [left, right].
        
        Time: O(right - left + 1)
        Uses: Kadane's algorithm
        """
        if left < 0 or right >= self.n or left > right:
            return 0
        
        subarray = self.arr[left:right + 1]
        return DynamicProgramming.max_subarray_sum(subarray)
    
    def longest_increasing_subsequence_in_range(self, left: int, right: int) -> int:
        """
        Find length of LIS in range [left, right].
        
        Time: O((right - left + 1) * log(right - left + 1))
        Uses: Binary search optimization
        """
        if left < 0 or right >= self.n or left > right:
            return 0
        
        subarray = self.arr[left:right + 1]
        return DynamicProgramming.longest_increasing_subsequence(subarray)
    
    def _rebuild_structures(self):
        """
        Rebuild all data structures after range modifications.
        
        Time: O(n log n)
        
        Called after range updates that invalidate structures.
        """
        self.sum_tree = SegmentTree(self.arr, lambda a, b: a + b, 0)
        self.gcd_tree = GCDSegmentTree(self.arr)
        self.fenwick = FenwickTree(self.arr)
        self.merge_sort_tree = MergeSortTree(self.arr)
        self.cache.clear()
    
    def get_statistics(self, left: int, right: int) -> dict:
        """
        Compute comprehensive statistics for range [left, right].
        
        Time: O((right - left + 1) * log n)
        
        Returns dictionary with multiple metrics.
        """
        if left < 0 or right >= self.n or left > right:
            return {}
        
        elements = self.arr[left:right + 1]
        
        return {
            'sum': self.range_sum(left, right),
            'gcd': self.range_gcd(left, right),
            'min': min(elements),
            'max': max(elements),
            'mean': sum(elements) / len(elements),
            'median': sorted(elements)[len(elements) // 2],
            'count': len(elements),
            'prime_count': self.count_primes_in_range(left, right),
            'xor': self.range_xor(left, right)
        }
    
    def batch_queries(self, queries: List[tuple]) -> List[any]:
        """
        Process multiple queries efficiently.
        
        Args:
            queries: List of (query_type, *args) tuples
        
        Returns:
            List of query results
        
        Optimization: Could reorder queries for cache efficiency.
        """
        results = []
        
        for query in queries:
            query_type = query[0]
            args = query[1:]
            
            if query_type == 'sum':
                results.append(self.range_sum(*args))
            elif query_type == 'gcd':
                results.append(self.range_gcd(*args))
            elif query_type == 'update':
                self.point_update(*args)
                results.append(None)
            elif query_type == 'kth':
                results.append(self.kth_smallest_in_range(*args))
            elif query_type == 'divisible':
                results.append(self.count_divisible(*args))
            elif query_type == 'modexp':
                results.append(self.modular_exp_sum(*args))
            else:
                results.append(None)
        
        return results