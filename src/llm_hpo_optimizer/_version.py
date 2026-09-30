from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-hpo-optimizer")
except PackageNotFoundError:  # package not installed (e.g. running from source)
    __version__ = "0.0.0+unknown"
