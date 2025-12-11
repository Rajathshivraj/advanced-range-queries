"""
Unit Tests for Fenwick Tree Implementations
"""

import pytest
import sys
sys.path.insert(0, '../src')

from src.fenwick_tree import FenwickTree, FenwickTree2D, RangeFenwickTree


class TestFenwickTree:
    """Test cases for 1D Fenwick Tree."""
    
    def test_prefix_sum(self):
        arr = [1, 3, 5, 7, 9, 11]
        ft = FenwickTree(arr)
        
        assert ft.prefix_sum(0) == 1
        assert ft.prefix_sum(2) == 9
        assert ft.prefix_sum(5) == 36
    
    def test_range_sum(self):
        arr = [1, 3, 5, 7, 9, 11]
        ft = FenwickTree(arr)
        
        assert ft.range_sum(0, 2) == 9
        assert ft.range_sum(1, 4) == 24
        assert ft.range_sum(3, 5) == 27
    
    def test_update(self):
        arr = [1, 3, 5, 7, 9]
        ft = FenwickTree(arr)
        
        ft.update(2, 10)  # Add 10 to index 2
        assert ft.prefix_sum(2) == 19
        assert ft.range_sum(0, 4) == 35
    
    def test_point_update(self):
        arr = [1, 3, 5, 7, 9]
        ft = FenwickTree(arr)
        
        ft.point_update(2, 15, 5)  # Change arr[2] from 5 to 15
        assert ft.range_sum(0, 4) == 35
    
    def test_multiple_updates(self):
        arr = [1, 2, 3, 4, 5]
        ft = FenwickTree(arr)
        
        ft.update(0, 5)
        ft.update(2, 5)
        ft.update(4, 5)
        
        assert ft.prefix_sum(4) == 30
    
    def test_lower_bound(self):
        arr = [1, 2, 3, 4, 5]
        ft = FenwickTree(arr)
        
        # prefix sums: [1, 3, 6, 10, 15]
        assert ft.lower_bound(1) == 0
        assert ft.lower_bound(3) == 1
        assert ft.lower_bound(7) == 3
        assert ft.lower_bound(100) == -1


class TestFenwickTree2D:
    """Test cases for 2D Fenwick Tree."""
    
    def test_2d_prefix_sum(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        ft2d = FenwickTree2D(matrix)
        
        assert ft2d.prefix_sum(0, 0) == 1
        assert ft2d.prefix_sum(1, 1) == 12
        assert ft2d.prefix_sum(2, 2) == 45
    
    def test_2d_range_sum(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        ft2d = FenwickTree2D(matrix)
        
        # Sum of middle element
        assert ft2d.range_sum(1, 1, 1, 1) == 5
        
        # Sum of 2x2 submatrix (top-left)
        assert ft2d.range_sum(0, 0, 1, 1) == 12
        
        # Sum of entire matrix
        assert ft2d.range_sum(0, 0, 2, 2) == 45
    
    def test_2d_update(self):
        matrix = [
            [1, 2],
            [3, 4]
        ]
        ft2d = FenwickTree2D(matrix)
        
        ft2d.update(0, 0, 5)  # Add 5 to (0,0)
        assert ft2d.range_sum(0, 0, 1, 1) == 15
    
    def test_2d_empty(self):
        matrix = []
        ft2d = FenwickTree2D(matrix)
        assert ft2d.n == 0


class TestRangeFenwickTree:
    """Test cases for Range Fenwick Tree."""
    
    def test_range_update_and_query(self):
        arr = [1, 2, 3, 4, 5]
        rft = RangeFenwickTree(arr)
        
        # Add 10 to range [1, 3]
        rft.range_update(1, 3, 10)
        
        assert rft.range_sum(0, 0) == 1
        assert rft.range_sum(1, 1) == 12
        assert rft.range_sum(2, 2) == 13
        assert rft.range_sum(3, 3) == 14
        assert rft.range_sum(4, 4) == 5
    
    def test_multiple_range_updates(self):
        arr = [0, 0, 0, 0, 0]
        rft = RangeFenwickTree(arr)
        
        rft.range_update(0, 2, 5)
        rft.range_update(1, 3, 3)
        
        assert rft.point_query(0) == 5
        assert rft.point_query(1) == 8
        assert rft.point_query(2) == 8
        assert rft.point_query(3) == 3
        assert rft.point_query(4) == 0
    
    def test_range_sum_after_updates(self):
        arr = [1, 1, 1, 1, 1]
        rft = RangeFenwickTree(arr)
        
        rft.range_update(1, 3, 5)
        
        assert rft.range_sum(0, 4) == 20
        assert rft.range_sum(1, 3) == 18


class TestFenwickPerformance:
    """Performance and stress tests."""
    
    def test_large_array(self):
        n = 10000
        arr = list(range(1, n + 1))
        ft = FenwickTree(arr)
        
        # Test correctness with large array
        assert ft.prefix_sum(99) == 5050
        assert ft.range_sum(0, 999) == 500500
    
    def test_many_updates(self):
        arr = [1] * 1000
        ft = FenwickTree(arr)
        
        for i in range(100):
            ft.update(i, 1)
        
        assert ft.prefix_sum(99) == 200
    
    def test_alternating_operations(self):
        arr = [1, 2, 3, 4, 5]
        ft = FenwickTree(arr)
        
        for _ in range(100):
            ft.update(2, 1)
            _ = ft.range_sum(0, 4)
        
        assert ft.range_sum(0, 4) == 115


class TestBitManipulation:
    """Test bit manipulation properties."""
    
    def test_lowbit_property(self):
        """Verify i & -i gives lowest set bit."""
        assert 12 & -12 == 4  # 1100 & 0100 = 0100
        assert 10 & -10 == 2  # 1010 & 0010 = 0010
        assert 7 & -7 == 1    # 0111 & 0001 = 0001
    
    def test_parent_child_relationship(self):
        """Verify parent-child navigation in Fenwick tree."""
        arr = [1] * 16
        ft = FenwickTree(arr)
        
        # Index 8 should aggregate [1-8]
        # Index 12 should aggregate [9-12]
        # Index 14 should aggregate [13-14]
        # These relationships ensure O(log n) operations
        
        assert ft.prefix_sum(7) == 8
        assert ft.prefix_sum(11) == 12
        assert ft.prefix_sum(13) == 14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])