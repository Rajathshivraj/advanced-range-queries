"""
Dynamic Programming Module
Optimized DP solutions for numerical computation problems.
"""

from typing import List, Dict, Tuple
from functools import lru_cache


class DynamicProgramming:
    """
    Collection of DP algorithms for numerical problems.
    """
    
    @staticmethod
    def max_subarray_sum(arr: List[int]) -> int:
        """
        Kadane's algorithm for maximum subarray sum.
        
        Time: O(n)
        Space: O(1)
        
        Classic DP: At each position, decide whether to extend current
        subarray or start new one.
        """
        if not arr:
            return 0
        
        max_sum = current_sum = arr[0]
        
        for i in range(1, len(arr)):
            current_sum = max(arr[i], current_sum + arr[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    @staticmethod
    def longest_increasing_subsequence(arr: List[int]) -> int:
        """
        Find length of longest increasing subsequence.
        
        Time: O(n log n) using binary search
        Space: O(n)
        
        Maintain array of smallest tail elements for each length.
        """
        if not arr:
            return 0
        
        tails = []
        
        for num in arr:
            # Binary search for position
            left, right = 0, len(tails)
            
            while left < right:
                mid = (left + right) // 2
                if tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            
            if left == len(tails):
                tails.append(num)
            else:
                tails[left] = num
        
        return len(tails)
    
    @staticmethod
    def coin_change(coins: List[int], amount: int) -> int:
        """
        Minimum coins needed to make amount (unbounded knapsack).
        
        Time: O(amount * len(coins))
        Space: O(amount)
        
        DP formula: dp[i] = min(dp[i], dp[i-coin] + 1) for each coin
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    @staticmethod
    def matrix_chain_multiplication(dimensions: List[int]) -> int:
        """
        Minimum scalar multiplications for matrix chain.
        
        Time: O(n^3)
        Space: O(n^2)
        
        Classic interval DP. For matrices A1...An with dimensions
        p0×p1, p1×p2, ..., p(n-1)×pn, find optimal parenthesization.
        """
        n = len(dimensions) - 1
        
        if n <= 1:
            return 0
        
        # dp[i][j] = min cost to multiply matrices i through j
        dp = [[0] * n for _ in range(n)]
        
        # l is chain length
        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l - 1
                dp[i][j] = float('inf')
                
                # Try all split points
                for k in range(i, j):
                    cost = (dp[i][k] + dp[k + 1][j] + 
                            dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n - 1]
    
    @staticmethod
    def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
        """
        0/1 Knapsack problem.
        
        Time: O(n * capacity)
        Space: O(capacity) with space optimization
        
        DP formula: dp[w] = max(dp[w], dp[w-weight[i]] + value[i])
        """
        n = len(weights)
        dp = [0] * (capacity + 1)
        
        for i in range(n):
            # Traverse backwards to avoid using same item twice
            for w in range(capacity, weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
        
        return dp[capacity]
    
    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        """
        Minimum edit distance (Levenshtein distance).
        
        Time: O(m * n)
        Space: O(min(m, n)) with optimization
        
        Operations: insert, delete, replace
        """
        m, n = len(s1), len(s2)
        
        # Space optimization: only keep two rows
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            curr[0] = i
            
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j],      # delete
                                      curr[j - 1],   # insert
                                      prev[j - 1])   # replace
            
            prev, curr = curr, prev
        
        return prev[n]
    
    @staticmethod
    def subset_sum_count(arr: List[int], target: int) -> int:
        """
        Count subsets with given sum.
        
        Time: O(n * target)
        Space: O(target)
        
        DP formula: dp[sum] += dp[sum - arr[i]] for each element
        """
        dp = [0] * (target + 1)
        dp[0] = 1
        
        for num in arr:
            for s in range(target, num - 1, -1):
                dp[s] += dp[s - num]
        
        return dp[target]
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def partition_dp(n: int, k: int) -> int:
        """
        Number of ways to partition n into k parts.
        
        Time: O(n * k) with memoization
        Space: O(n * k)
        
        Recurrence: p(n, k) = p(n-1, k-1) + p(n-k, k)
        """
        if n < k or k < 1:
            return 0
        if n == k or k == 1:
            return 1
        
        return DynamicProgramming.partition_dp(n - 1, k - 1) + \
               DynamicProgramming.partition_dp(n - k, k)
    
    @staticmethod
    def range_sum_2d(matrix: List[List[int]]) -> List[List[int]]:
        """
        Precompute 2D prefix sums for O(1) range queries.
        
        Time: O(m * n) preprocessing
        Space: O(m * n)
        
        Query: sum[r1:r2][c1:c2] = prefix[r2][c2] - prefix[r1-1][c2]
                                    - prefix[r2][c1-1] + prefix[r1-1][c1-1]
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                prefix[i][j] = (matrix[i - 1][j - 1] + 
                                prefix[i - 1][j] + 
                                prefix[i][j - 1] - 
                                prefix[i - 1][j - 1])
        
        return prefix
    
    @staticmethod
    def palindrome_partitioning_min_cuts(s: str) -> int:
        """
        Minimum cuts to partition string into palindromes.
        
        Time: O(n^2)
        Space: O(n^2)
        
        Two-phase DP:
        1. Precompute palindrome table
        2. Find minimum cuts
        """
        n = len(s)
        if n <= 1:
            return 0
        
        # is_palindrome[i][j] = True if s[i:j+1] is palindrome
        is_palindrome = [[False] * n for _ in range(n)]
        
        # Every single character is palindrome
        for i in range(n):
            is_palindrome[i][i] = True
        
        # Check all lengths
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if s[i] == s[j]:
                    if length == 2:
                        is_palindrome[i][j] = True
                    else:
                        is_palindrome[i][j] = is_palindrome[i + 1][j - 1]
        
        # dp[i] = minimum cuts for s[0:i+1]
        dp = [float('inf')] * n
        
        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0
            else:
                for j in range(i):
                    if is_palindrome[j + 1][i]:
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n - 1]
    
    @staticmethod
    def egg_drop(eggs: int, floors: int) -> int:
        """
        Minimum trials to find critical floor (egg drop problem).
        
        Time: O(eggs * floors^2)
        Space: O(eggs * floors)
        
        DP formula: dp[e][f] = 1 + min(max(dp[e-1][k-1], dp[e][f-k]))
        for all k in [1, f]
        """
        # dp[e][f] = min trials with e eggs and f floors
        dp = [[0] * (floors + 1) for _ in range(eggs + 1)]
        
        # Base cases
        for i in range(1, eggs + 1):
            dp[i][0] = 0  # 0 floors = 0 trials
            dp[i][1] = 1  # 1 floor = 1 trial
        
        for j in range(1, floors + 1):
            dp[1][j] = j  # 1 egg = linear search
        
        # Fill table
        for e in range(2, eggs + 1):
            for f in range(2, floors + 1):
                dp[e][f] = float('inf')
                
                for k in range(1, f + 1):
                    # Max of: egg breaks (try lower), egg doesn't break (try higher)
                    trials = 1 + max(dp[e - 1][k - 1], dp[e][f - k])
                    dp[e][f] = min(dp[e][f], trials)
        
        return dp[eggs][floors]
    
    @staticmethod
    def optimal_bst(keys: List[int], freq: List[int]) -> int:
        """
        Minimum cost of optimal binary search tree.
        
        Time: O(n^3)
        Space: O(n^2)
        
        Cost = sum of (depth[i] + 1) * freq[i] for all keys.
        """
        n = len(keys)
        
        # cost[i][j] = min cost for keys[i:j+1]
        cost = [[0] * n for _ in range(n)]
        
        # freq_sum[i][j] = sum of frequencies for keys[i:j+1]
        freq_sum = [[0] * n for _ in range(n)]
        
        # Base case: single keys
        for i in range(n):
            cost[i][i] = freq[i]
            freq_sum[i][i] = freq[i]
        
        # Build for increasing lengths
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                cost[i][j] = float('inf')
                freq_sum[i][j] = freq_sum[i][j - 1] + freq[j]
                
                # Try each key as root
                for r in range(i, j + 1):
                    left_cost = cost[i][r - 1] if r > i else 0
                    right_cost = cost[r + 1][j] if r < j else 0
                    
                    total_cost = left_cost + right_cost + freq_sum[i][j]
                    cost[i][j] = min(cost[i][j], total_cost)
        
        return cost[0][n - 1]
    
    @staticmethod
    def digit_dp_sum(n: int) -> int:
        """
        Count numbers from 1 to n with digit sum divisible by 3.
        
        Time: O(log n * digit_sum * 10)
        Space: O(log n * digit_sum)
        
        Digit DP technique: process digits left to right with memoization.
        """
        if n <= 0:
            return 0
        
        digits = [int(d) for d in str(n)]
        memo = {}
        
        def dp(pos: int, sum_mod: int, tight: bool) -> int:
            """
            pos: current digit position
            sum_mod: current digit sum mod 3
            tight: whether we're still bounded by n
            """
            if pos == len(digits):
                return 1 if sum_mod == 0 else 0
            
            state = (pos, sum_mod, tight)
            if state in memo:
                return memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_sum_mod = (sum_mod + digit) % 3
                new_tight = tight and (digit == limit)
                result += dp(pos + 1, new_sum_mod, new_tight)
            
            memo[state] = result
            return result
        
        # Subtract 1 to exclude 0
        return dp(0, 0, True) - 1