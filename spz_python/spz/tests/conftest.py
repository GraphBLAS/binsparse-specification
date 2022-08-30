import pytest


@pytest.fixture(scope="session", autouse=True)
def ic():
    """Make `ic` available everywhere for easier debugging"""
    try:
        import icecream
    except ImportError:
        return
    icecream.install()
    # icecream.ic.disable()  # do ic.enable() to re-enable
    return icecream.ic


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
