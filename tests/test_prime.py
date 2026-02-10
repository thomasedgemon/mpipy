"""Basic unit tests for the primality check."""

from mpipy.prime import is_prime


def test_small_primes():
    assert is_prime(2) is True
    assert is_prime(3) is True
    assert is_prime(4) is False
    assert is_prime(17) is True
    assert is_prime(18) is False


def test_edge_cases():
    assert is_prime(0) is False
    assert is_prime(1) is False
    assert is_prime(-7) is False
