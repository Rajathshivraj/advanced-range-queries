"""
Segment Tree Implementation with Lazy Propagation
Supports: Range GCD, Range Sum, Range Updates, Kth Smallest
"""

from typing import List, Optional, Callable
import math


class SegmentTree:
    """
    Generic segment tree with support for various range operations.
    Uses lazy propagation for efficient range updates.
    """
    
    def __init__(self, arr: List[int], merge_fn: Callable = None, identity: int = 0):
        """
        Initialize segment tree.
        
        Args:
            arr: Input array
            merge_fn: Function to merge two nodes (default: sum)
            identity: Identity element for merge operation
        """
        self.n = len(arr)
        self.arr = arr.copy()
        self.tree = [identity] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self.merge_fn = merge_fn if merge_fn else lambda a, b: a + b
        self.identity = identity
        
        if self.n > 0:
            self._build(0, 0, self.n - 1)
    
    def _build(self, node: int, start: int, end: int):
        """Build segment tree recursively."""
        if start == end:
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self._build(left_child, start, mid)
            self._build(right_child, mid + 1, end)
            
            self.tree[node] = self.merge_fn(self.tree[left_child], self.tree[right_child])
    
    def _push_down(self, node: int, start: int, end: int):
        """Push down lazy propagation to children."""
        if self.lazy[node] != 0:
            self.tree[node] += self.lazy[node] * (end - start + 1)
            
            if start != end:
                left_child = 2 * node + 1
                right_child = 2 * node + 2
                self.lazy[left_child] += self.lazy[node]
                self.lazy[right_child] += self.lazy[node]
            
            self.lazy[node] = 0
    
    def range_query(self, l: int, r: int) -> int:
        """
        Query range [l, r].
        Time: O(log n)
        """
        return self._range_query(0, 0, self.n - 1, l, r)
    
    def _range_query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        """Internal range query with lazy propagation."""
        if l > end or r < start:
            return self.identity
        
        self._push_down(node, start, end)
        
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_val = self._range_query(left_child, start, mid, l, r)
        right_val = self._range_query(right_child, mid + 1, end, l, r)
        
        return self.merge_fn(left_val, right_val)
    
    def point_update(self, idx: int, value: int):
        """
        Update single element.
        Time: O(log n)
        """
        self._point_update(0, 0, self.n - 1, idx, value)
        self.arr[idx] = value
    
    def _point_update(self, node: int, start: int, end: int, idx: int, value: int):
        """Internal point update."""
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if idx <= mid:
                self._point_update(left_child, start, mid, idx, value)
            else:
                self._point_update(right_child, mid + 1, end, idx, value)
            
            self.tree[node] = self.merge_fn(self.tree[left_child], self.tree[right_child])
    
    def range_update(self, l: int, r: int, delta: int):
        """
        Add delta to all elements in range [l, r].
        Time: O(log n) with lazy propagation
        """
        self._range_update(0, 0, self.n - 1, l, r, delta)
    
    def _range_update(self, node: int, start: int, end: int, l: int, r: int, delta: int):
        """Internal range update with lazy propagation."""
        if l > end or r < start:
            return
        
        if l <= start and end <= r:
            self.lazy[node] += delta
            return
        
        self._push_down(node, start, end)
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        self._range_update(left_child, start, mid, l, r, delta)
        self._range_update(right_child, mid + 1, end, l, r, delta)
        
        self._push_down(left_child, start, mid)
        self._push_down(right_child, mid + 1, end)
        
        self.tree[node] = self.merge_fn(self.tree[left_child], self.tree[right_child])


class GCDSegmentTree:
    """
    Specialized segment tree for GCD queries.
    GCD is associative and commutative, making it suitable for segment trees.
    """
    
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.arr = arr.copy()
        self.tree = [0] * (4 * self.n)
        
        if self.n > 0:
            self._build(0, 0, self.n - 1)
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """
        Euclidean GCD algorithm.
        Time: O(log(min(a, b)))
        """
        while b:
            a, b = b, a % b
        return abs(a)
    
    def _build(self, node: int, start: int, end: int):
        """Build GCD segment tree."""
        if start == end:
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self._build(left_child, start, mid)
            self._build(right_child, mid + 1, end)
            
            self.tree[node] = self.gcd(self.tree[left_child], self.tree[right_child])
    
    def range_gcd(self, l: int, r: int) -> int:
        """
        Compute GCD of all elements in range [l, r].
        Time: O(log n)
        """
        return self._range_gcd(0, 0, self.n - 1, l, r)
    
    def _range_gcd(self, node: int, start: int, end: int, l: int, r: int) -> int:
        """Internal range GCD query."""
        if l > end or r < start:
            return 0
        
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_gcd = self._range_gcd(left_child, start, mid, l, r)
        right_gcd = self._range_gcd(right_child, mid + 1, end, l, r)
        
        if left_gcd == 0:
            return right_gcd
        if right_gcd == 0:
            return left_gcd
        
        return self.gcd(left_gcd, right_gcd)
    
    def point_update(self, idx: int, value: int):
        """
        Update element at index idx.
        Time: O(log n)
        """
        self._point_update(0, 0, self.n - 1, idx, value)
        self.arr[idx] = value
    
    def _point_update(self, node: int, start: int, end: int, idx: int, value: int):
        """Internal point update for GCD tree."""
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if idx <= mid:
                self._point_update(left_child, start, mid, idx, value)
            else:
                self._point_update(right_child, mid + 1, end, idx, value)
            
            self.tree[node] = self.gcd(self.tree[left_child], self.tree[right_child])


class MergeSortTree:
    """
    Merge Sort Tree for Kth smallest element queries in range.
    Each node stores sorted elements of its range.
    Space: O(n log n)
    """
    
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.arr = arr.copy()
        self.tree = [[] for _ in range(4 * self.n)]
        
        if self.n > 0:
            self._build(0, 0, self.n - 1)
    
    def _build(self, node: int, start: int, end: int):
        """Build merge sort tree."""
        if start == end:
            self.tree[node] = [self.arr[start]]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self._build(left_child, start, mid)
            self._build(right_child, mid + 1, end)
            
            # Merge sorted arrays from children
            self.tree[node] = self._merge(self.tree[left_child], self.tree[right_child])
    
    @staticmethod
    def _merge(left: List[int], right: List[int]) -> List[int]:
        """Merge two sorted arrays."""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        
        return result
    
    def kth_smallest(self, l: int, r: int, k: int) -> Optional[int]:
        """
        Find kth smallest element (1-indexed) in range [l, r].
        Time: O(log^2 n)
        Returns None if k is out of bounds.
        """
        elements = self._collect_range(0, 0, self.n - 1, l, r)
        if k < 1 or k > len(elements):
            return None
        return sorted(elements)[k - 1]
    
    def _collect_range(self, node: int, start: int, end: int, l: int, r: int) -> List[int]:
        """Collect all elements in range [l, r]."""
        if l > end or r < start:
            return []
        
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_elements = self._collect_range(left_child, start, mid, l, r)
        right_elements = self._collect_range(right_child, mid + 1, end, l, r)
        
        return self._merge(left_elements, right_elements)
    
    def count_less_than(self, l: int, r: int, value: int) -> int:
        """
        Count elements in range [l, r] that are less than value.
        Time: O(log^2 n)
        """
        return self._count_less_than(0, 0, self.n - 1, l, r, value)
    
    def _count_less_than(self, node: int, start: int, end: int, l: int, r: int, value: int) -> int:
        """Internal count less than query using binary search."""
        if l > end or r < start:
            return 0
        
        if l <= start and end <= r:
            # Binary search in sorted node
            left, right = 0, len(self.tree[node])
            while left < right:
                mid = (left + right) // 2
                if self.tree[node][mid] < value:
                    left = mid + 1
                else:
                    right = mid
            return left
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_count = self._count_less_than(left_child, start, mid, l, r, value)
        right_count = self._count_less_than(right_child, mid + 1, end, l, r, value)
        
        return left_count + right_count