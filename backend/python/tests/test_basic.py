import unittest
from image_detection import __version__

class TestBasic(unittest.TestCase):
    def test_version(self):
        self.assertEqual(__version__, "0.1.0")

if __name__ == "__main__":
    unittest.main()
