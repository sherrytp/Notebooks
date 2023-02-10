def hello_name(name): 
    return f'hello {name}'


# Unittest - self.assertEqual
import unittest

class TestHello(unittest.TestCase): 

    def test_hello_name(self): 
        self.assertEqual(hello_name('bob'), 'hello bob')


if __name__ == '__main__': 
    unittest.main()



# Pytest - simpler assert function

def test_hello_name(): 
    assert hello_name('bob') == 'hello bob'


