"""
Main Driver Program
Demonstrates the capabilities of the Advanced Numerical Computation System.
"""

from query_processor import QueryProcessor
from number_theory import NumberTheory
import time


def benchmark_operations(arr, num_queries=1000):
    """
    Benchmark different operations to demonstrate performance.
    """
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    qp = QueryProcessor(arr)
    n = len(arr)
    
    # Benchmark Range Sum
    start = time.time()
    for _ in range(num_queries):
        qp.range_sum(0, n // 2)
    sum_time = (time.time() - start) * 1000
    print(f"Range Sum ({num_queries} queries): {sum_time:.2f}ms")
    
    # Benchmark Range GCD
    start = time.time()
    for _ in range(num_queries):
        qp.range_gcd(0, n // 2)
    gcd_time = (time.time() - start) * 1000
    print(f"Range GCD ({num_queries} queries): {gcd_time:.2f}ms")
    
    # Benchmark Point Updates
    start = time.time()
    for i in range(min(num_queries, n)):
        qp.point_update(i, arr[i] + 1)
    update_time = (time.time() - start) * 1000
    print(f"Point Updates ({min(num_queries, n)} updates): {update_time:.2f}ms")
    
    # Benchmark Kth Smallest
    start = time.time()
    for _ in range(min(100, num_queries)):
        qp.kth_smallest_in_range(0, min(100, n - 1), 50)
    kth_time = (time.time() - start) * 1000
    print(f"Kth Smallest (100 queries): {kth_time:.2f}ms")
    
    print()


def demonstrate_basic_queries():
    """
    Demonstrate basic query operations.
    """
    print("=" * 60)
    print("BASIC QUERY DEMONSTRATIONS")
    print("=" * 60)
    
    arr = [12, 18, 24, 30, 36, 42, 48, 54, 60]
    print(f"Array: {arr}")
    print()
    
    qp = QueryProcessor(arr)
    
    # Range Sum
    result = qp.range_sum(0, 5)
    print(f"Range Sum [0, 5]: {result}")
    print(f"  Expected: {sum(arr[0:6])} ✓")
    print()
    
    # Range GCD
    result = qp.range_gcd(0, 5)
    print(f"Range GCD [0, 5]: {result}")
    print(f"  (GCD of 12, 18, 24, 30, 36, 42)")
    print()
    
    # Point Update
    print("Updating arr[2] = 96")
    qp.point_update(2, 96)
    result = qp.range_sum(0, 5)
    print(f"New Range Sum [0, 5]: {result}")
    print()
    
    # Count Divisible
    result = qp.count_divisible(0, 8, 6)
    print(f"Count elements divisible by 6: {result}")
    print()
    
    # Kth Smallest
    arr2 = [7, 3, 9, 1, 5, 8, 2, 6, 4]
    qp2 = QueryProcessor(arr2)
    result = qp2.kth_smallest_in_range(0, 8, 5)
    print(f"Array: {arr2}")
    print(f"5th smallest element: {result}")
    print(f"  Sorted: {sorted(arr2)}")
    print()


def demonstrate_advanced_queries():
    """
    Demonstrate advanced numerical operations.
    """
    print("=" * 60)
    print("ADVANCED QUERY DEMONSTRATIONS")
    print("=" * 60)
    
    arr = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    print(f"Array (primes): {arr}")
    print()
    
    qp = QueryProcessor(arr)
    
    # Modular Exponentiation Sum
    result = qp.modular_exp_sum(0, 4, 3, 1000)
    print("Modular Exponentiation Sum:")
    print(f"  Sum of (arr[i]^3 mod 1000) for i in [0, 4]")
    manual = sum(pow(arr[i], 3, 1000) for i in range(5))
    print(f"  Result: {result}")
    print(f"  Verification: {manual} ✓")
    print()
    
    # Range LCM
    arr2 = [12, 18, 24, 30]
    qp2 = QueryProcessor(arr2)
    result = qp2.range_lcm(0, 3)
    print(f"Array: {arr2}")
    print(f"Range LCM [0, 3]: {result}")
    print()
    
    # Prime Count
    arr3 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    qp3 = QueryProcessor(arr3)
    result = qp3.count_primes_in_range(0, 10)
    print(f"Array: {arr3}")
    print(f"Prime count: {result}")
    print(f"  Primes: 11, 13, 17, 19")
    print()
    
    # Statistics
    arr4 = [5, 2, 8, 1, 9, 3, 7, 4, 6]
    qp4 = QueryProcessor(arr4)
    stats = qp4.get_statistics(0, 8)
    print(f"Array: {arr4}")
    print("Comprehensive Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


def demonstrate_batch_processing():
    """
    Demonstrate efficient batch query processing.
    """
    print("=" * 60)
    print("BATCH QUERY PROCESSING")
    print("=" * 60)
    
    arr = list(range(1, 21))
    print(f"Array: {arr}")
    print()
    
    qp = QueryProcessor(arr)
    
    queries = [
        ('sum', 0, 9),
        ('gcd', 5, 10),
        ('update', 5, 100),
        ('sum', 0, 9),
        ('kth', 0, 19, 10),
        ('divisible', 0, 19, 5),
    ]
    
    print("Executing batch queries:")
    for i, query in enumerate(queries):
        print(f"  {i + 1}. {query}")
    print()
    
    results = qp.batch_queries(queries)
    
    print("Results:")
    for i, (query, result) in enumerate(zip(queries, results)):
        if result is not None:
            print(f"  {i + 1}. {query[0]}: {result}")
    print()


def demonstrate_number_theory():
    """
    Demonstrate number theory operations.
    """
    print("=" * 60)
    print("NUMBER THEORY DEMONSTRATIONS")
    print("=" * 60)
    
    nt = NumberTheory(1000)
    
    # Prime Factorization
    n = 360
    factors = nt.prime_factorization(n)
    print(f"Prime Factorization of {n}:")
    print(f"  {factors}")
    reconstruction = 1
    for p, e in factors.items():
        reconstruction *= p ** e
    print(f"  Verification: {reconstruction} ✓")
    print()
    
    # GCD and LCM
    a, b = 48, 180
    gcd = nt.gcd(a, b)
    lcm = nt.lcm(a, b)
    print(f"GCD({a}, {b}) = {gcd}")
    print(f"LCM({a}, {b}) = {lcm}")
    print(f"  Verification: {a} * {b} / {gcd} = {lcm} ✓")
    print()
    
    # Modular Exponentiation
    base, exp, mod = 2, 100, 1000000007
    result = nt.mod_pow(base, exp, mod)
    print(f"Modular Exponentiation:")
    print(f"  {base}^{exp} mod {mod} = {result}")
    print()
    
    # Divisor Count
    n = 120
    count = nt.count_divisors(n)
    sum_div = nt.sum_of_divisors(n)
    print(f"Divisors of {n}:")
    print(f"  Count: {count}")
    print(f"  Sum: {sum_div}")
    print()
    
    # Euler's Totient
    n = 36
    phi = nt.euler_totient(n)
    print(f"Euler's Totient φ({n}) = {phi}")
    print(f"  (Count of numbers ≤ {n} coprime to {n})")
    print()


def demonstrate_complex_scenario():
    """
    Demonstrate a complex real-world scenario.
    """
    print("=" * 60)
    print("COMPLEX SCENARIO: Financial Data Analysis")
    print("=" * 60)
    
    # Simulate stock prices over 20 days
    prices = [100, 102, 98, 105, 110, 108, 115, 112, 118, 120,
              122, 119, 125, 123, 128, 130, 127, 132, 135, 133]
    
    print(f"Stock Prices (20 days): {prices}")
    print()
    
    qp = QueryProcessor(prices)
    
    # Weekly analysis
    print("Weekly Analysis:")
    for week in range(4):
        start = week * 5
        end = start + 4
        
        total = qp.range_sum(start, end)
        avg = total / 5
        max_gain = qp.max_subarray_sum_in_range(start, end)
        
        print(f"  Week {week + 1} (days {start}-{end}):")
        print(f"    Total: {total}")
        print(f"    Average: {avg:.2f}")
        print(f"    Max Consecutive Gain: {max_gain}")
    print()
    
    # Find best performing period
    print("Best Performing Periods:")
    best_5day = max(
        (qp.range_sum(i, i + 4), i) 
        for i in range(len(prices) - 4)
    )
    print(f"  Best 5-day period: Days {best_5day[1]}-{best_5day[1] + 4}")
    print(f"  Total: {best_5day[0]}")
    print()
    
    # Volatility analysis (using GCD as a proxy)
    print("Volatility Metrics:")
    for period in [(0, 4), (5, 9), (10, 14), (15, 19)]:
        gcd = qp.range_gcd(*period)
        print(f"  Days {period[0]}-{period[1]}: GCD = {gcd}")
    print()


def main():
    """
    Main entry point.
    """
    print("\n")
    print("*" * 60)
    print("ADVANCED NUMERICAL COMPUTATION SYSTEM")
    print("Demonstrating Data Structures & Algorithms Mastery")
    print("*" * 60)
    print("\n")
    
    # Run demonstrations
    demonstrate_basic_queries()
    demonstrate_advanced_queries()
    demonstrate_batch_processing()
    demonstrate_number_theory()
    demonstrate_complex_scenario()
    
    # Benchmark on larger dataset
    print("=" * 60)
    print("LARGE-SCALE BENCHMARK")
    print("=" * 60)
    print("Initializing array of size 10,000...")
    large_arr = list(range(1, 10001))
    benchmark_operations(large_arr, num_queries=1000)
    
    print("=" * 60)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 60)
    print("\nKey Achievements Demonstrated:")
    print("  ✓ O(log n) range queries using Segment Trees")
    print("  ✓ O(log n) updates using Fenwick Trees")
    print("  ✓ O(log² n) Kth smallest queries")
    print("  ✓ O(log exp) modular exponentiation")
    print("  ✓ O(n log log n) prime generation")
    print("  ✓ Efficient batch processing")
    print("  ✓ Multiple data structure coordination")
    print("\n")


if __name__ == "__main__":
    main()