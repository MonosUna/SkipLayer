from setuptools import find_namespace_packages, setup

setup(
    name="skiplayer",
    version="0.1.0",
    description="Layer-Skip LLM training framework",
    packages=find_namespace_packages(where=".", include=["src*"]),
    python_requires=">=3.10",
)
