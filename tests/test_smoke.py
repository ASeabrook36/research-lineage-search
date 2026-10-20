def test_package_imports():
    from research.db import DIMENSIONS
    assert DIMENSIONS == 384
