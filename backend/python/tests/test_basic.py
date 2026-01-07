"""基础测试用例"""


def test_import():
    """测试包导入"""
    import image_detection
    assert image_detection.__version__ == "0.1.0"


def test_basic():
    """基础测试"""
    assert 1 + 1 == 2
