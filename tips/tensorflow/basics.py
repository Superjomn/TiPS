import ctypes
import atexit


class TipsBasics(object):
    """Wrapper class for the basic TiPS APIs."""

    def __init__(self, pkg_path):
        full_path = pkg_path
        self.CORE_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

        self.init()

    def init(self):
        """Initialize TiPS."""
        self.CORE_CTYPES.tips_init()

        atexit.register(self.shutdown)

    def shutdown(self):
        """A function to shutdown TiPS service."""
        self.CORE_CTYPES.tips_shutdown()

    def initialized(self):
        return self.CORE_CTYPES.tips_is_initialized()

    def size(self):
        return self.CORE_CTYPES.tips_size()

    def rank(self):
        return self.CORE_CTYPES.tips_rank()
