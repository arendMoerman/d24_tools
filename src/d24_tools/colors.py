PURPLE = '\033[95m'
CYAN = '\033[96m'
DARKCYAN = '\033[36m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m'

def _green(string, bold=True):
    """
    Formatter for green terminal output.

    @param string String to be formatted.

    @returns Formatted string
    """
    if bold:
        return GREEN + BOLD + string + END
    return GREEN + string + END

def _yellow(string, bold=True):
    """
    Formatter for yellow terminal output.

    @param string String to be formatted.

    @returns Formatted string
    """
    if bold:
        return YELLOW + BOLD + string + END
    return YELLOW + string + END
