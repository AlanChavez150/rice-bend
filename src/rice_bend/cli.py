"""The scaffolding all three entry points share.

Only logging setup lives here. After the flag diet the shared argparse surface is
three flags, and hiding add_argument calls behind add_config_args/add_output_args
helpers makes each main() harder to read, not easier.
"""

import logging

import coloredlogs


def setup_logging(debug: bool) -> logging.Logger:
    """Install the project's log format at INFO, or DEBUG when asked."""
    coloredlogs.install(level="DEBUG" if debug else "INFO",
                        fmt="%(levelname)s: %(message)s")
    return logging.getLogger()
