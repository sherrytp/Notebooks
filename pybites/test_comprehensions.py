from comprehensions import filter_positive_even_numbers


def test_filter_positive_negative_numbers(): 
    nums = list(range(-10, 11))
    assert filter_positive_even_numbers(nums) == [2, 4, 6, 8, 10]


def test_positive_numbers(): 
    numbers = [2, 4, 50, 51, 44, 47, 8]
    assert filter_positive_even_numbers(numbers) == [2, 4, 50, 44, 8]


def test_filter_negetive_zero(): 
    nums = list(range(-10, 1))
    assert filter_positive_even_numbers(nums) == []

