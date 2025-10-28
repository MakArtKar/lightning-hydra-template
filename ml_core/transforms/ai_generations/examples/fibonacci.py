def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number.

    The Fibonacci sequence is: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
    Where each number is the sum of the two preceding ones.

    Args:
        n: The position in the Fibonacci sequence (0-indexed)

    Returns:
        The n-th Fibonacci number
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
