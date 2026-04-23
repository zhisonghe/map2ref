from datetime import datetime


def get_logger(module: str):
    """Return a progress-logging callable bound to *module*.

    Usage::

        from helpers.log import get_logger
        _log = get_logger('my_module')
        _log('Something happened')
        # prints: [2026-04-23 12:34:56] [my_module] Something happened
    """
    def _log(msg: str) -> None:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [{module}] {msg}')
    return _log
