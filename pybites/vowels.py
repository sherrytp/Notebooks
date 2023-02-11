
m typing import Tuple

TEXT = """
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
"""
VOWELS = 'aeiou'


def strip_vowels(text: str = TEXT) -> Tuple[str, int]:
    new_str = []
    chars = list(text)
    num_vowels = 0

    for c in chars:
        if c.lower() in VOWELS:
            c = '*'
            num_vowels += 1
            new_str.append(c)

    return ''.join(new_str), num_vowels

