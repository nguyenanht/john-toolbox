try:
    from importlib_metadata import version

    __version__ = version(__package__)
except:
    pass