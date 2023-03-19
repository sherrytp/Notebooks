from unittest.mock import patch

from color import print_colors

NOT_VALID = 'Not a valid color'


def call_print_colors(): 
    try: 
        print_colors()
    except SystemExit: 
        pass


@patch("builtins.input", side_effect=['blue', 'white', 'quit'])
def test_valid_then_invalid_color_then_quit(input_mock, capsys):
    call_print_colors()
    actual = capsys.readouterr()[0].strip()
    expected = f'blue\n{NOT_VALID}\nbye'
    assert actual == expected 


@patch("builtins.input", side_effect=['green', 'quit'])
def test_invalid_color_then_quit(input_mock,capsys): 
    call_print_colors()
    actual = capsys.readouterr()[0].strip()
    expected = f'{NOT_VALID}\nbye'
    assert actual == expected 

