"""Shared test fixtures: paths to the mini curated inputs."""

import os

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def amyload_mini_path():
    return os.path.join(FIXTURES, "amyload_mini.csv")
