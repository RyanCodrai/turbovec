def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running thread-safety stress cells (~1.5 s each)",
    )
