try:
    from importlib.metadata import version
    __version__: str = version("statscpp")
except Exception:
    __version__ = "dev"
