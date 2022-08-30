def pytest_addoption(parser):
    parser.addoption("--runslow", default=None, action="store_true", help="run slow tests")
