from text import split_in_columns

def text_split_in_columns(): 
    
    expected_msg = "Hello world!|We hope that you are learning a lot of Python.|Have fun with our Bites of Py.|Keep calm and code in Python!|Become a PyBites ninja!"
    actual_msg = split_in_columns()
    assert expected_msg == actual_msg 

    expected = "Hello world:|I am coding in Python :)|How awesome!" 
    msg = "Hello world:\nI am coding in Python :)\nHow awesome!"
    actual = split_in_columns(msg)
    assert expected == actual 

