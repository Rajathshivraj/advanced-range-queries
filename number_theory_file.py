"""
Number Theory Module
Provides optimized algorithms for:
- Prime generation (Sieve of Eratosthenes)
- Prime factorization
- GCD and LCM
- Modular arithmetic
- Divisor counting
"""

from typing import List, Dict, Tuple
import math


class NumberTheory:
    """
    Collection of number theory algorithms optimized for competitive programming.
    """
    
    def __init__(self, max_n: int = 10**6):
        """
        Initialize with sieve preprocessing.
        
        Args:
            max_n: Maximum number for sieve (default 10^6)
        """
        self.max_n = max_n
        self.primes = self._sieve_of_eratosthenes(int(math.sqrt(max_n)) + 1)
        self.is_prime_cache = {}
        self.factorization_cache = {}
    
    @staticmethod
    def _sieve_of_eratosthenes(n: int) -> List[int]:
        """
        Generate all primes up to n using Sieve of Eratosthenes.
        
        Time: O(n log log n)
        Space: O(n)
        
        Optimization: Only check odd numbers after 2.
        """
        if n < 2:
            return []
        
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        # Only need to check up to sqrt(n)
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                # Mark multiples as composite
                # Start from i*i (smaller multiples already marked)
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        
        return [i for i in range(n + 1) if is_prime[i]]
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """
        Euclidean algorithm for GCD.
        
        Time: O(log(min(a, b)))
        
        Key insight: gcd(a, b) = gcd(b, a mod b)
        Base case: gcd(a, 0) = a
        """
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def lcm(a: int, b: int) -> int:
        """
        Least Common Multiple using GCD.
        
        Time: O(log(min(a, b)))
        
        Formula: lcm(a, b) = (a * b) / gcd(a, b)
        """
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // NumberTheory.gcd(a, b)
    
    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean algorithm.
        Returns (gcd, x, y) such that a*x + b*y = gcd(a, b)
        
        Time: O(log(min(a, b)))
        
        Useful for modular multiplicative inverse.
        """
        if b == 0:
            return a, 1, 0
        
        gcd, x1, y1 = NumberTheory.extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        
        return gcd, x, y
    
    def prime_factorization(self, n: int) -> Dict[int, int]:
        """
        Compute prime factorization of n.
        Returns dict mapping prime -> exponent.
        
        Time: O(sqrt(n) / ln(sqrt(n))) with sieve preprocessing
              O(sqrt(n)) without preprocessing
        
        Optimization: Use precomputed primes for trial division.
        """
        if n in self.factorization_cache:
            return self.factorization_cache[n]
        
        if n <= 1:
            return {}
        
        factors = {}
        original_n = n
        
        # Trial division with precomputed primes
        for p in self.primes:
            if p * p > n:
                break
            
            while n % p == 0:
                factors[p] = factors.get(p, 0) + 1
                n //= p
        
        # If n > 1, it's a prime factor
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        self.factorization_cache[original_n] = factors
        return factors
    
    def count_divisors(self, n: int) -> int:
        """
        Count number of divisors of n.
        
        Time: O(sqrt(n) / ln(sqrt(n)))
        
        Formula: If n = p1^a1 * p2^a2 * ... * pk^ak
        Then divisors(n) = (a1+1) * (a2+1) * ... * (ak+1)
        """
        factors = self.prime_factorization(n)
        count = 1
        
        for exp in factors.values():
            count *= (exp + 1)
        
        return count
    
    def sum_of_divisors(self, n: int) -> int:
        """
        Compute sum of all divisors of n.
        
        Time: O(sqrt(n) / ln(sqrt(n)))
        
        Formula: If n = p^a, then sum = (p^(a+1) - 1) / (p - 1)
        For multiple primes, multiply the sums.
        """
        factors = self.prime_factorization(n)
        total = 1
        
        for prime, exp in factors.items():
            # Sum of geometric series: (p^(e+1) - 1) / (p - 1)
            total *= (pow(prime, exp + 1) - 1) // (prime - 1)
        
        return total
    
    @staticmethod
    def mod_pow(base: int, exp: int, mod: int) -> int:
        """
        Modular exponentiation: (base^exp) % mod
        
        Time: O(log exp)
        
        Binary exponentiation algorithm:
        - If exp is even: a^exp = (a^(exp/2))^2
        - If exp is odd: a^exp = a * a^(exp-1)
        
        Critical for large exponents (e.g., 10^9).
        """
        result = 1
        base %= mod
        
        while exp > 0:
            if exp & 1:  # If exp is odd
                result = (result * base) % mod
            
            base = (base * base) % mod
            exp >>= 1  # Divide exp by 2
        
        return result
    
    @staticmethod
    def mod_inverse(a: int, mod: int) -> int:
        """
        Modular multiplicative inverse of a modulo mod.
        Returns x such that (a * x) % mod = 1
        
        Time: O(log mod)
        
        Uses extended Euclidean algorithm.
        Requires gcd(a, mod) = 1.
        """
        gcd, x, _ = NumberTheory.extended_gcd(a, mod)
        
        if gcd != 1:
            raise ValueError(f"Modular inverse doesn't exist for {a} mod {mod}")
        
        return (x % mod + mod) % mod
    
    def is_prime(self, n: int) -> bool:
        """
        Check if n is prime.
        
        Time: O(sqrt(n) / ln(sqrt(n))) with sieve
              O(sqrt(n)) without sieve
        """
        if n in self.is_prime_cache:
            return self.is_prime_cache[n]
        
        if n < 2:
            return False
        
        if n == 2:
            return True
        
        if n % 2 == 0:
            return False
        
        # Check divisibility by precomputed primes
        for p in self.primes:
            if p * p > n:
                break
            if n % p == 0:
                self.is_prime_cache[n] = False
                return False
        
        self.is_prime_cache[n] = True
        return True
    
    @staticmethod
    def euler_totient(n: int, factors: Dict[int, int] = None) -> int:
        """
        Euler's totient function φ(n): count of numbers ≤ n coprime to n.
        
        Time: O(sqrt(n)) if factors not provided
        
        Formula: φ(n) = n * ∏(1 - 1/p) for all prime factors p
        """
        if factors is None:
            factors = NumberTheory().prime_factorization(n)
        
        result = n
        
        for prime in factors:
            result -= result // prime
        
        return result
    
    @staticmethod
    def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
        """
        Solve system of congruences:
        x ≡ a1 (mod m1)
        x ≡ a2 (mod m2)
        ...
        
        Time: O(n log(max modulus))
        
        Requires moduli to be pairwise coprime.
        """
        if len(remainders) != len(moduli):
            raise ValueError("Remainders and moduli must have same length")
        
        total = 0
        prod = 1
        
        for m in moduli:
            prod *= m
        
        for r, m in zip(remainders, moduli):
            p = prod // m
            total += r * p * NumberTheory.mod_inverse(p, m)
        
        return total % prod
    
    @staticmethod
    def binomial_coefficient(n: int, k: int, mod: int = None) -> int:
        """
        Compute C(n, k) = n! / (k! * (n-k)!)
        
        Time: O(k) or O(k log mod) with modular arithmetic
        
        Optimization: Use formula C(n,k) = C(n,k-1) * (n-k+1) / k
        """
        if k > n or k < 0:
            return 0
        
        if k == 0 or k == n:
            return 1
        
        # Optimize: C(n, k) = C(n, n-k)
        k = min(k, n - k)
        
        result = 1
        
        for i in range(k):
            result *= (n - i)
            result //= (i + 1)
            
            if mod:
                result %= mod
        
        return result if not mod else result % mod
    
    def count_divisible_in_range(self, arr: List[int], l: int, r: int, divisor: int) -> int:
        """
        Count elements in arr[l:r+1] divisible by divisor.
        
        Time: O(r - l + 1)
        
        Optimization: Could use preprocessing with counting arrays
        for multiple queries on same array.
        """
        count = 0
        for i in range(l, r + 1):
            if arr[i] % divisor == 0:
                count += 1
        return count
    
    @staticmethod
    def next_prime(n: int) -> int:
        """
        Find the smallest prime greater than n.
        
        Time: O(n log log n) in worst case
        """
        candidate = n + 1
        
        while True:
            if NumberTheory().is_prime(candidate):
                return candidate
            candidate += 1
    
    @staticmethod
    def power_of_prime_in_factorial(n: int, p: int) -> int:
        """
        Find the exponent of prime p in n!
        
        Time: O(log_p(n))
        
        Uses Legendre's formula: sum of floor(n/p^i) for i = 1, 2, 3, ...
        """
        count = 0
        power = p
        
        while power <= n:
            count += n // power
            power *= p
        
        return count