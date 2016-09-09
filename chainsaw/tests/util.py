import logging


class MockLoggingHandler(logging.Handler):
    """Mock logging handler to check for expected logs."""

    def __init__(self, *args, **kwargs):
        self.reset()
        logging.Handler.__init__(self, *args, **kwargs)

    def emit(self, record):
        self.messages[record.levelname.lower()].append(record.getMessage())

    def reset(self):
        self.messages = {
            'debug': [],
            'info': [],
            'warning': [],
            'error': [],
            'critical': [],
        }


def _setup_testing():
    # setup function for testing
    from chainsaw import config
    # do not cache trajectory info in user directory (temp traj files)
    config.use_trajectory_lengths_cache = False
    config.show_progress_bars = False


def _monkey_patch_testing_apply_setting():
    """ this monkey patches the init methods of unittest.TestCase and doctest.DocTestFinder,
    in order to apply internal settings. """
    import unittest
    import doctest
    _old_init = unittest.TestCase.__init__
    def _new_init(self, *args, **kwargs):
        _old_init(self, *args, **kwargs)
        _setup_testing()

    unittest.TestCase.__init__ = _new_init

    _old_init_doc_test_finder = doctest.DocTestFinder.__init__

    def _patched_init(self, *args, **kw):
        _setup_testing()
        _old_init_doc_test_finder(self, *args, **kw)

    doctest.DocTestFinder.__init__ = _patched_init
