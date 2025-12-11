"""
Unit Tests for Segment Tree Implementations
"""

import pytest
import sys
sys.path.insert(0, '../src')

from src.segment_tree import SegmentTree, GCDSegmentTree, MergeSortTree


class TestSegmentTree:
    """Test cases for basic Segment Tree."""
    
    def test_range_sum_query(self):
        arr = [1, 3, 5, 7, 9, 11]
        tree = SegmentTree(arr, lambda a, b: a + b, 0)
        
        assert tree.range_query(0, 2) == 9
        assert tree.range_query(1, 4) == 24
        assert tree.range_query(0, 5) == 36
    
    def test_point_update(self):
        arr = [1, 3, 5, 7, 9, 11]
        tree = SegmentTree(arr, lambda a, b: a + b, 0)
        
        tree.point_update(2, 10)
        assert tree.range_query(0, 2) == 14
        assert tree.range_query(2, 2) == 10
    
    def test_range_update(self):
        arr = [1, 3, 5, 7, 9]
        tree = SegmentTree(arr, lambda a, b: a + b, 0)
        
        tree.range_update(1, 3, 5)
        # After update: [1, 8, 10, 12, 9]
        assert tree.range_query(1, 3) == 30
    
    def test_empty_array(self):
        arr = []
        tree = SegmentTree(arr, lambda a, b: a + b, 0)
        assert tree.n == 0
    
    def test_single_element(self):
        arr = [42]
        tree = SegmentTree(arr, lambda a, b: a + b, 0)
        assert tree.range_query(0, 0) == 42


class TestGCDSegmentTree:
    """Test cases for GCD Segment Tree."""
    
    def test_gcd_basic(self):
        assert GCDSegmentTree.gcd(48, 18) == 6
        assert GCDSegmentTree.gcd(100, 75) == 25
        assert GCDSegmentTree.gcd(17, 19) == 1
    
    def test_range_gcd_query(self):
        arr = [12, 18, 24, 30, 36]
        tree = GCDSegmentTree(arr)
        
        assert tree.range_gcd(0, 4) == 6
        assert tree.range_gcd(0, 2) == 6
        assert tree.range_gcd(2, 4) == 6
    
    def test_gcd_coprime(self):
        arr = [7, 11, 13, 17]
        tree = GCDSegmentTree(arr)
        
        assert tree.range_gcd(0, 3) == 1
    
    def test_gcd_update(self):
        arr = [12, 18, 24]
        tree = GCDSegmentTree(arr)
        
        tree.point_update(1, 36)
        assert tree.range_gcd(0, 2) == 12
    
    def test_gcd_powers_of_two(self):
        arr = [8, 16, 32, 64]
        tree = GCDSegmentTree(arr)
        
        assert tree.range_gcd(0, 3) == 8


class TestMergeSortTree:
    """Test cases for Merge Sort Tree."""
    
    def test_kth_smallest_basic(self):
        arr = [7, 3, 9, 1, 5]
        tree = MergeSortTree(arr)
        
        assert tree.kth_smallest(0, 4, 1) == 1
        assert tree.kth_smallest(0, 4, 3) == 5
        assert tree.kth_smallest(0, 4, 5) == 9
    
    def test_kth_smallest_subrange(self):
        arr = [7, 3, 9, 1, 5, 8, 2]
        tree = MergeSortTree(arr)
        
        assert tree.kth_smallest(1, 5, 2) == 5
        assert tree.kth_smallest(2, 6, 3) == 8
    
    def test_kth_invalid(self):
        arr = [1, 2, 3, 4, 5]
        tree = MergeSortTree(arr)
        
        assert tree.kth_smallest(0, 4, 0) is None
        assert tree.kth_smallest(0, 4, 6) is None
    
    def test_count_less_than(self):
        arr = [3, 1, 4, 1, 5, 9, 2, 6]
        tree = MergeSortTree(arr)
        
        assert tree.count_less_than(0, 7, 5) == 5
        assert tree.count_less_than(0, 7, 10) == 8
        assert tree.count_less_than(0, 7, 1) == 0
    
    def test_sorted_array(self):
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        tree = MergeSortTree(arr)
        
        assert tree.kth_smallest(0, 9, 5) == 5
        assert tree.kth_smallest(2, 7, 3) == 5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_negative_numbers(self):
        arr = [-5, -2, 3, 7, -1]
        tree = SegmentTree(arr, lambda a, b: a + b, 0)
        
        assert tree.range_query(0, 4) == 2
        assert tree.range_query(0, 1) == -7
    
    def test_large_numbers(self):
        arr = [10**9, 10**9, 10**9]
        tree = SegmentTree(arr, lambda a, b: a + b, 0)
        
        assert tree.range_query(0, 2) == 3 * 10**9
    
    def test_all_same_elements(self):
        arr = [5, 5, 5, 5, 5]
        gcd_tree = GCDSegmentTree(arr)
        
        assert gcd_tree.range_gcd(0, 4) == 5
    
    def test_two_elements(self):
        arr = [10, 20]
        tree = SegmentTree(arr, lambda a, b: a + b, 0)
        
        assert tree.range_query(0, 1) == 30
        
        tree.point_update(0, 15)
        assert tree.range_query(0, 1) == 35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])