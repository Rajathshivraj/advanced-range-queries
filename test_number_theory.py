"""
Unit Tests for Number Theory Module
"""

import pytest
import sys
sys.path.insert(0, '../src')

from src.number_theory import NumberTheory


class TestPrimeGeneration:
    """Test prime generation using Sieve of Eratosthenes."""
    
    def test_small_primes(self):
        primes = NumberTheory._sieve_of_eratosthenes(30)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert primes == expected
    
    def test_first_prime(self):
        primes = NumberTheory._sieve_of_eratosthenes(2)
        assert primes == [2]
    
    def test_no_primes(self):
        primes = NumberTheory._sieve_of_eratosthenes(1)
        assert primes == []
    
    def test_prime_count(self):
        primes = NumberTheory._sieve_of_eratosthenes(100)
        assert len(primes) == 25  # There are 25 primes < 100


class TestGCDLCM:
    """Test GCD and LCM calculations."""
    
    def test_gcd_basic(self):
        assert NumberTheory.gcd(48, 18) == 6
        assert NumberTheory.gcd(100, 75) == 25
        assert NumberTheory.gcd(17, 19) == 1
    
    def test_gcd_commutative(self):
        assert NumberTheory.gcd(48, 18) == NumberTheory.gcd(18, 48)
    
    def test_gcd_with_zero(self):
        assert NumberTheory.gcd(0, 5) == 5
        assert NumberTheory.gcd(10, 0) == 10
    
    def test_lcm_basic(self):
        assert NumberTheory.lcm(12, 18) == 36
        assert NumberTheory.lcm(21, 6) == 42
    
    def test_lcm_coprime(self):
        assert NumberTheory.lcm(7, 11) == 77
    
    def test_gcd_lcm_relationship(self):
        a, b = 48, 180
        assert a * b == NumberTheory.gcd(a, b) * NumberTheory.lcm(a, b)


class TestExtendedGCD:
    """Test Extended Euclidean Algorithm."""
    
    def test_extended_gcd(self):
        gcd, x, y = NumberTheory.extended_gcd(48, 18)
        assert gcd == 6
        assert 48 * x + 18 * y == gcd
    
    def test_extended_gcd_coprime(self):
        gcd, x, y = NumberTheory.extended_gcd(17, 19)
        assert gcd == 1
        assert 17 * x + 19 * y == gcd


class TestPrimeFactorization:
    """Test prime factorization."""
    
    def test_factorization_basic(self):
        nt = NumberTheory(1000)
        
        factors = nt.prime_factorization(60)
        assert factors == {2: 2, 3: 1, 5: 1}
        
        factors = nt.prime_factorization(100)
        assert factors == {2: 2, 5: 2}
    
    def test_factorization_prime(self):
        nt = NumberTheory(100)
        
        factors = nt.prime_factorization(17)
        assert factors == {17: 1}
    
    def test_factorization_one(self):
        nt = NumberTheory(100)
        
        factors = nt.prime_factorization(1)
        assert factors == {}
    
    def test_factorization_power_of_two(self):
        nt = NumberTheory(1000)
        
        factors = nt.prime_factorization(256)
        assert factors == {2: 8}
    
    def test_reconstruction_from_factors(self):
        nt = NumberTheory(1000)
        
        n = 360
        factors = nt.prime_factorization(n)
        
        reconstructed = 1
        for prime, exp in factors.items():
            reconstructed *= prime ** exp
        
        assert reconstructed == n


class TestDivisors:
    """Test divisor counting and sum."""
    
    def test_count_divisors(self):
        nt = NumberTheory(1000)
        
        assert nt.count_divisors(12) == 6  # 1,2,3,4,6,12
        assert nt.count_divisors(28) == 6  # 1,2,4,7,14,28
        assert nt.count_divisors(17) == 2  # 1,17
    
    def test_sum_of_divisors(self):
        nt = NumberTheory(1000)
        
        assert nt.sum_of_divisors(12) == 28  # 1+2+3+4+6+12
        assert nt.sum_of_divisors(6) == 12   # 1+2+3+6
    
    def test_perfect_number(self):
        nt = NumberTheory(1000)
        
        # 6 is perfect: sum of proper divisors = 6
        assert nt.sum_of_divisors(6) - 6 == 6


class TestModularArithmetic:
    """Test modular arithmetic operations."""
    
    def test_mod_pow_basic(self):
        assert NumberTheory.mod_pow(2, 10, 1000) == 24
        assert NumberTheory.mod_pow(3, 5, 100) == 43
    
    def test_mod_pow_large_exponent(self):
        result = NumberTheory.mod_pow(2, 100, 1000000007)
        assert result == pow(2, 100, 1000000007)
    
    def test_mod_pow_zero_exponent(self):
        assert NumberTheory.mod_pow(5, 0, 100) == 1
    
    def test_mod_inverse(self):
        a, mod = 3, 11
        inv = NumberTheory.mod_inverse(a, mod)
        assert (a * inv) % mod == 1
    
    def test_mod_inverse_error(self):
        with pytest.raises(ValueError):
            NumberTheory.mod_inverse(6, 9)  # gcd(6, 9) = 3 != 1


class TestPrimalityTesting:
    """Test prime checking."""
    
    def test_is_prime_basic(self):
        nt = NumberTheory(1000)
        
        assert nt.is_prime(2)
        assert nt.is_prime(17)
        assert nt.is_prime(97)
        
        assert not nt.is_prime(1)
        assert not nt.is_prime(4)
        assert not nt.is_prime(100)
    
    def test_is_prime_edge_cases(self):
        nt = NumberTheory(1000)
        
        assert not nt.is_prime(0)
        assert not nt.is_prime(-5)
        assert nt.is_prime(2)


class TestEulerTotient:
    """Test Euler's totient function."""
    
    def test_totient_basic(self):
        assert NumberTheory.euler_totient(1) == 1
        assert NumberTheory.euler_totient(9) == 6
        assert NumberTheory.euler_totient(10) == 4
    
    def test_totient_prime(self):
        assert NumberTheory.euler_totient(17) == 16
        assert NumberTheory.euler_totient(31) == 30
    
    def test_totient_with_factors(self):
        nt = NumberTheory(1000)
        factors = nt.prime_factorization(36)
        assert NumberTheory.euler_totient(36, factors) == 12


class TestChineseRemainderTheorem:
    """Test Chinese Remainder Theorem."""
    
    def test_crt_basic(self):
        remainders = [2, 3, 2]
        moduli = [3, 5, 7]
        
        x = NumberTheory.chinese_remainder_theorem(remainders, moduli)
        
        assert x % 3 == 2
        assert x % 5 == 3
        assert x % 7 == 2
    
    def test_crt_single(self):
        x = NumberTheory.chinese_remainder_theorem([5], [13])
        assert x == 5


class TestBinomialCoefficient:
    """Test binomial coefficient calculation."""
    
    def test_binomial_basic(self):
        assert NumberTheory.binomial_coefficient(5, 2) == 10
        assert NumberTheory.binomial_coefficient(10, 3) == 120
    
    def test_binomial_edge_cases(self):
        assert NumberTheory.binomial_coefficient(5, 0) == 1
        assert NumberTheory.binomial_coefficient(5, 5) == 1
        assert NumberTheory.binomial_coefficient(5, 6) == 0
    
    def test_binomial_with_mod(self):
        result = NumberTheory.binomial_coefficient(100, 50, 1000000007)
        assert result > 0


class TestAdvancedFunctions:
    """Test advanced number theory functions."""
    
    def test_next_prime(self):
        assert NumberTheory.next_prime(10) == 11
        assert NumberTheory.next_prime(13) == 17
    
    def test_power_in_factorial(self):
        # 10! = 2^8 * 3^4 * 5^2 * 7
        assert NumberTheory.power_of_prime_in_factorial(10, 2) == 8
        assert NumberTheory.power_of_prime_in_factorial(10, 3) == 4
        assert NumberTheory.power_of_prime_in_factorial(10, 5) == 2
        assert NumberTheory.power_of_prime_in_factorial(10, 7) == 1


class TestCaching:
    """Test caching mechanisms."""
    
    def test_factorization_cache(self):
        nt = NumberTheory(1000)
        
        # First call
        factors1 = nt.prime_factorization(360)
        
        # Second call (should use cache)
        factors2 = nt.prime_factorization(360)
        
        assert factors1 == factors2
        assert 360 in nt.factorization_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])