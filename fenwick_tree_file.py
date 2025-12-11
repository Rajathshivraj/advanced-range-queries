"""
Fenwick Tree (Binary Indexed Tree) Implementation
Optimized for prefix sum queries and point updates using bit manipulation.
"""

from typing import List


class FenwickTree:
    """
    Binary Indexed Tree for efficient prefix sum queries.
    
    Key insight: Each index i is responsible for elements in range
    [i - (i & -i) + 1, i]. The bit trick (i & -i) isolates the 
    lowest set bit, creating an implicit binary tree structure.
    
    Time Complexity:
        - Build: O(n log n) or O(n) with optimization
        - Update: O(log n)
        - Prefix Query: O(log n)
        - Range Query: O(log n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, arr: List[int]):
        """
        Initialize Fenwick tree from array.
        Uses 1-indexed internally for cleaner bit manipulation.
        """
        self.n = len(arr)
        self.tree = [0] * (self.n + 1)
        
        # Build tree in O(n) time
        for i in range(self.n):
            self.tree[i + 1] = arr[i]
        
        # Transform into Fenwick tree structure
        for i in range(1, self.n + 1):
            parent = i + (i & -i)
            if parent <= self.n:
                self.tree[parent] += self.tree[i]
    
    def update(self, idx: int, delta: int):
        """
        Add delta to element at index idx (0-indexed).
        
        Time: O(log n)
        
        Bit manipulation insight:
        i & -i gives the lowest set bit of i.
        Adding this moves to the parent in the implicit tree.
        """
        idx += 1  # Convert to 1-indexed
        
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx  # Move to parent
    
    def prefix_sum(self, idx: int) -> int:
        """
        Compute sum of elements from index 0 to idx (inclusive, 0-indexed).
        
        Time: O(log n)
        
        Bit manipulation insight:
        Subtracting (i & -i) moves to the previous responsible range,
        effectively traversing ancestors in the implicit tree.
        """
        idx += 1  # Convert to 1-indexed
        total = 0
        
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & -idx  # Move to previous range
        
        return total
    
    def range_sum(self, left: int, right: int) -> int:
        """
        Compute sum of elements in range [left, right] (0-indexed).
        
        Time: O(log n)
        
        Uses the property: sum[l, r] = prefix_sum[r] - prefix_sum[l-1]
        """
        if left > right:
            return 0
        
        if left == 0:
            return self.prefix_sum(right)
        
        return self.prefix_sum(right) - self.prefix_sum(left - 1)
    
    def point_update(self, idx: int, value: int, current_value: int):
        """
        Set element at index idx to value.
        Requires knowing the current value at idx.
        
        Time: O(log n)
        """
        delta = value - current_value
        self.update(idx, delta)
    
    def lower_bound(self, target_sum: int) -> int:
        """
        Find the smallest index where prefix_sum >= target_sum.
        Uses binary lifting on the Fenwick tree structure.
        
        Time: O(log^2 n)
        
        Returns -1 if no such index exists (target_sum > total_sum).
        """
        if target_sum <= 0:
            return 0
        
        total = self.prefix_sum(self.n - 1)
        if target_sum > total:
            return -1
        
        # Binary search on answer
        left, right = 0, self.n - 1
        
        while left < right:
            mid = (left + right) // 2
            if self.prefix_sum(mid) >= target_sum:
                right = mid
            else:
                left = mid + 1
        
        return left


class FenwickTree2D:
    """
    2D Fenwick Tree for 2D range sum queries.
    
    Time Complexity:
        - Update: O(log n * log m)
        - Query: O(log n * log m)
    
    Space Complexity: O(n * m)
    """
    
    def __init__(self, matrix: List[List[int]]):
        """Initialize 2D Fenwick tree from matrix."""
        if not matrix or not matrix[0]:
            self.n = self.m = 0
            self.tree = []
            return
        
        self.n = len(matrix)
        self.m = len(matrix[0])
        self.tree = [[0] * (self.m + 1) for _ in range(self.n + 1)]
        
        for i in range(self.n):
            for j in range(self.m):
                self._add(i, j, matrix[i][j])
    
    def _add(self, row: int, col: int, delta: int):
        """Internal add operation (1-indexed logic)."""
        row += 1
        
        while row <= self.n:
            c = col + 1
            while c <= self.m:
                self.tree[row][c] += delta
                c += c & -c
            row += row & -row
    
    def update(self, row: int, col: int, delta: int):
        """
        Add delta to element at (row, col).
        Time: O(log n * log m)
        """
        self._add(row, col, delta)
    
    def prefix_sum(self, row: int, col: int) -> int:
        """
        Compute sum of rectangle from (0, 0) to (row, col) inclusive.
        Time: O(log n * log m)
        """
        row += 1
        col += 1
        total = 0
        
        r = row
        while r > 0:
            c = col
            while c > 0:
                total += self.tree[r][c]
                c -= c & -c
            r -= r & -r
        
        return total
    
    def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """
        Compute sum of rectangle from (r1, c1) to (r2, c2) inclusive.
        
        Time: O(log n * log m)
        
        Uses inclusion-exclusion principle:
        sum[r1:r2, c1:c2] = prefix[r2,c2] - prefix[r1-1,c2] 
                           - prefix[r2,c1-1] + prefix[r1-1,c1-1]
        """
        if r1 > r2 or c1 > c2:
            return 0
        
        result = self.prefix_sum(r2, c2)
        
        if r1 > 0:
            result -= self.prefix_sum(r1 - 1, c2)
        
        if c1 > 0:
            result -= self.prefix_sum(r2, c1 - 1)
        
        if r1 > 0 and c1 > 0:
            result += self.prefix_sum(r1 - 1, c1 - 1)
        
        return result


class RangeFenwickTree:
    """
    Fenwick Tree with range update and range query support.
    Uses difference array technique with two Fenwick trees.
    
    For range update [l, r] += delta:
        - Add delta to diff[l]
        - Add -delta to diff[r+1]
    
    For range sum query [l, r]:
        - Use two Fenwick trees to efficiently compute the sum
    
    Time Complexity:
        - Range Update: O(log n)
        - Range Query: O(log n)
    """
    
    def __init__(self, arr: List[int]):
        """Initialize with two Fenwick trees for range operations."""
        self.n = len(arr)
        self.tree1 = FenwickTree([0] * self.n)
        self.tree2 = FenwickTree([0] * self.n)
        
        # Initialize with point updates
        for i in range(self.n):
            self.range_update(i, i, arr[i])
    
    def range_update(self, left: int, right: int, delta: int):
        """
        Add delta to all elements in range [left, right].
        Time: O(log n)
        """
        self.tree1.update(left, delta)
        self.tree1.update(right + 1, -delta)
        
        self.tree2.update(left, delta * (left - 1))
        self.tree2.update(right + 1, -delta * right)
    
    def prefix_sum(self, idx: int) -> int:
        """
        Compute prefix sum up to index idx.
        Time: O(log n)
        """
        return self.tree1.prefix_sum(idx) * idx - self.tree2.prefix_sum(idx)
    
    def range_sum(self, left: int, right: int) -> int:
        """
        Compute sum in range [left, right].
        Time: O(log n)
        """
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)
    
    def point_query(self, idx: int) -> int:
        """
        Get value at single index.
        Time: O(log n)
        """
        return self.range_sum(idx, idx)