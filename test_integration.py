"""
Integration Tests for Query Processor
Tests the complete system with multiple data structures working together.
"""

import pytest
import sys
sys.path.insert(0, '../src')

from src.query_processor import QueryProcessor


class TestBasicIntegration:
    """Test basic integration of query processor."""
    
    def test_initialization(self):
        arr = [1, 2, 3, 4, 5]
        qp = QueryProcessor(arr)
        
        assert qp.n == 5
        assert qp.arr == [1, 2, 3, 4, 5]
    
    def test_multiple_query_types(self):
        arr = [12, 18, 24, 30, 36]
        qp = QueryProcessor(arr)
        
        # Test different query types
        assert qp.range_sum(0, 4) == 120
        assert qp.range_gcd(0, 4) == 6
        assert qp.kth_smallest_in_range(0, 4, 3) == 24
    
    def test_queries_after_update(self):
        arr = [10, 20, 30, 40, 50]
        qp = QueryProcessor(arr)
        
        initial_sum = qp.range_sum(0, 4)
        
        qp.point_update(2, 100)
        
        new_sum = qp.range_sum(0, 4)
        assert new_sum == initial_sum + 70


class TestComplexScenarios:
    """Test complex scenarios with multiple operations."""
    
    def test_sequential_updates_and_queries(self):
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        qp = QueryProcessor(arr)
        
        # Initial query
        assert qp.range_sum(0, 9) == 55
        
        # Update and re-query
        qp.point_update(5, 100)
        assert qp.range_sum(0, 9) == 149
        
        # Another update
        qp.point_update(0, 50)
        assert qp.range_sum(0, 9) == 198
    
    def test_divisibility_queries(self):
        arr = [6, 12, 18, 24, 30, 35, 42, 48]
        qp = QueryProcessor(arr)
        
        # Count multiples of 6
        count_6 = qp.count_divisible(0, 7, 6)
        assert count_6 == 7  # All except 35
        
        # Count multiples of 5
        count_5 = qp.count_divisible(0, 7, 5)
        assert count_5 == 2  # 30, 35
    
    def test_modular_exponentiation(self):
        arr = [2, 3, 5, 7]
        qp = QueryProcessor(arr)
        
        mod = 1000000007
        result = qp.modular_exp_sum(0, 3, 2, mod)
        
        expected = (4 + 9 + 25 + 49) % mod
        assert result == expected
    
    def test_range_lcm(self):
        arr = [12, 18, 24]
        qp = QueryProcessor(arr)
        
        lcm = qp.range_lcm(0, 2)
        assert lcm == 72


class TestBatchProcessing:
    """Test batch query processing."""
    
    def test_batch_mixed_queries(self):
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        qp = QueryProcessor(arr)
        
        queries = [
            ('sum', 0, 9),
            ('gcd', 2, 5),
            ('update', 5, 100),
            ('sum', 0, 9),
            ('kth', 0, 9, 5),
        ]
        
        results = qp.batch_queries(queries)
        
        assert results[0] == 55  # Initial sum
        assert results[1] == 1   # GCD of 3,4,5,6
        assert results[2] is None  # Update has no return
        assert results[3] == 149  # Sum after update
        assert results[4] == 5 or results[4] == 100  # Kth smallest
    
    def test_batch_all_sums(self):
        arr = list(range(1, 11))
        qp = QueryProcessor(arr)
        
        queries = [('sum', i, i + 2) for i in range(8)]
        results = qp.batch_queries(queries)
        
        assert len(results) == 8
        assert results[0] == 6  # 1+2+3


class TestStatistics:
    """Test statistical functions."""
    
    def test_comprehensive_statistics(self):
        arr = [5, 2, 8, 1, 9, 3, 7, 4, 6]
        qp = QueryProcessor(arr)
        
        stats = qp.get_statistics(0, 8)
        
        assert stats['sum'] == 45
        assert stats['min'] == 1
        assert stats['max'] == 9
        assert stats['count'] == 9
        assert stats['median'] == 5
        assert abs(stats['mean'] - 5.0) < 0.001
    
    def test_statistics_subrange(self):
        arr = list(range(1, 21))
        qp = QueryProcessor(arr)
        
        stats = qp.get_statistics(5, 14)
        
        assert stats['count'] == 10
        assert stats['sum'] == 105  # 6+7+...+15


class TestPrimeOperations:
    """Test prime-related operations."""
    
    def test_count_primes(self):
        arr = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        qp = QueryProcessor(arr)
        
        prime_count = qp.count_primes_in_range(0, 9)
        assert prime_count == 5  # 2, 3, 5, 7, 11
    
    def test_sum_of_divisors(self):
        arr = [6, 12, 18, 24]
        qp = QueryProcessor(arr)
        
        divisor_count_sum = qp.sum_of_divisors_in_range(0, 3)
        # 6: 4 divisors, 12: 6, 18: 6, 24: 8
        assert divisor_count_sum == 24


class TestXOROperations:
    """Test XOR-related queries."""
    
    def test_range_xor(self):
        arr = [1, 2, 3, 4, 5]
        qp = QueryProcessor(arr)
        
        xor_result = qp.range_xor(0, 4)
        expected = 1 ^ 2 ^ 3 ^ 4 ^ 5
        assert xor_result == expected
    
    def test_xor_identity(self):
        arr = [7, 7, 7, 7]
        qp = QueryProcessor(arr)
        
        # XOR of even count of same numbers is 0
        assert qp.range_xor(0, 3) == 0
        
        # XOR of odd count
        assert qp.range_xor(0, 2) == 7


class TestDynamicProgrammingIntegration:
    """Test DP algorithms integrated with query processor."""
    
    def test_max_subarray_sum(self):
        arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
        qp = QueryProcessor(arr)
        
        max_sum = qp.max_subarray_sum_in_range(0, 8)
        assert max_sum == 6  # [4, -1, 2, 1]
    
    def test_longest_increasing_subsequence(self):
        arr = [10, 9, 2, 5, 3, 7, 101, 18]
        qp = QueryProcessor(arr)
        
        lis_length = qp.longest_increasing_subsequence_in_range(0, 7)
        assert lis_length == 4  # [2, 3, 7, 101] or [2, 3, 7, 18]


class TestEdgeCasesIntegration:
    """Test edge cases in integration."""
    
    def test_empty_array(self):
        arr = []
        qp = QueryProcessor(arr)
        
        assert qp.n == 0
        assert qp.range_sum(0, 0) == 0
    
    def test_single_element(self):
        arr = [42]
        qp = QueryProcessor(arr)
        
        assert qp.range_sum(0, 0) == 42
        assert qp.range_gcd(0, 0) == 42
        assert qp.kth_smallest_in_range(0, 0, 1) == 42
    
    def test_invalid_ranges(self):
        arr = [1, 2, 3, 4, 5]
        qp = QueryProcessor(arr)
        
        assert qp.range_sum(-1, 2) == 0
        assert qp.range_sum(3, 10) == 0
        assert qp.range_sum(4, 2) == 0
    
    def test_large_values(self):
        arr = [10**9, 10**9, 10**9]
        qp = QueryProcessor(arr)
        
        assert qp.range_sum(0, 2) == 3 * 10**9


class TestPerformanceIntegration:
    """Test performance with larger datasets."""
    
    def test_large_array_operations(self):
        n = 1000
        arr = list(range(1, n + 1))
        qp = QueryProcessor(arr)
        
        # Multiple operations should complete quickly
        assert qp.range_sum(0, 999) == n * (n + 1) // 2
        assert qp.range_gcd(0, 999) == 1
        
        qp.point_update(500, 2000)
        assert qp.range_sum(0, 999) > n * (n + 1) // 2
    
    def test_many_updates(self):
        arr = [1] * 100
        qp = QueryProcessor(arr)
        
        for i in range(100):
            qp.point_update(i, i + 1)
        
        assert qp.range_sum(0, 99) == sum(range(1, 101))


class TestRealWorldScenario:
    """Test real-world scenario simulation."""
    
    def test_financial_analysis(self):
        # Stock prices over 30 days
        prices = [100 + i * 2 + (i % 7) * 3 for i in range(30)]
        qp = QueryProcessor(prices)
        
        # Weekly analysis
        week1_sum = qp.range_sum(0, 6)
        week2_sum = qp.range_sum(7, 13)
        
        assert week1_sum > 0
        assert week2_sum > week1_sum
        
        # Find best 5-day period
        best_sum = max(qp.range_sum(i, i + 4) for i in range(26))
        assert best_sum > 0
    
    def test_data_stream_processing(self):
        # Simulate incoming data stream
        arr = list(range(1, 51))
        qp = QueryProcessor(arr)
        
        # Process in chunks
        chunk_results = []
        for i in range(0, 50, 10):
            chunk_sum = qp.range_sum(i, min(i + 9, 49))
            chunk_results.append(chunk_sum)
        
        assert len(chunk_results) == 5
        assert sum(chunk_results) == sum(arr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])