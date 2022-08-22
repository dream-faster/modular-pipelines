from pprint import pformat
import textwrap
import re
from enum import Enum


def pprint_indent(text, indent=" " * 4 + "┃ ") -> None:
    text = pformat(text)
    print("".join([indent + l for l in text.splitlines(True)]))


class PrintFormats(Enum):
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class LogModes(Enum):
    MULTILINE = "multiline"
    BOX = "box"


class LogLevels(Enum):
    zero = 0
    one = 1
    two = 2


class DocumentWrapper(textwrap.TextWrapper):
    def wrap(self, text):
        split_text = text.split("\n")
        lines = [
            line
            for para in split_text
            for line in textwrap.TextWrapper.wrap(self, para)
        ]
        return lines


def remove_ansi_escape(s: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", s)


class LogWrapper:
    modes = LogModes
    formats = PrintFormats
    levels = LogLevels

    def __init__(self):
        pass

    def log(
        text: str,
        level: LogLevels = LogLevels.zero,
        mode: LogModes = None,
        *args,
        **kwargs
    ) -> None:

        if mode == LogModes.MULTILINE:
            print(multi_line_formatter(text, level))
        if mode == LogModes.BOX:
            print(string_in_box_formatter(text, thickness_level=level))


logger = LogWrapper()


def multi_line_formatter(text: str, level: int = 0) -> None:

    base_indent = " " * 4

    if level == 0:
        initial_indent = base_indent + "┣━━━ "
        subsequent_indent = base_indent + "┃ "
    if level == 1:
        initial_indent = base_indent + "┃  ├── "
        subsequent_indent = base_indent + "┃  │   "
    if level == 2:
        initial_indent = base_indent + "┃    "
        subsequent_indent = base_indent + "┃    "

    d = DocumentWrapper(
        width=100,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
    )

    return d.fill(text)


def string_in_box_formatter(
    text: str, width: int = 100, height: int = 1, thickness_level: int = 1
) -> None:
    if thickness_level == 0:
        top_left = "┌"
        top_right = "┐"
        vertical = "│"
        bottom_left = "└"
        bottom_right = "┘"
        horizontal = "─"
        t_down = "┬"
    elif thickness_level == 1:
        top_left = "┏"
        top_right = "┓"
        vertical = "┃"
        bottom_left = "┗"
        bottom_right = "┛"
        horizontal = "━"
        t_down = "┳"
    else:
        top_left = "┏"
        top_right = "┓"
        vertical = "┃"
        bottom_left = "┗"
        bottom_right = "┛"
        horizontal = "━"
        t_down = "┳"

    complex_string = top_left + horizontal * width + top_right + "\n"

    for _ in range(height):
        complex_string += vertical + " " * width + vertical + "\n"

    for sub_string in text.split("\n"):
        text_length = len(remove_ansi_escape(sub_string))
        whitespace_around_text = int(max(0, (width - text_length) / 2))

        complex_string += (
            vertical
            + " " * whitespace_around_text
            + sub_string
            + " " * max(0, (width - whitespace_around_text - text_length))
            + vertical
            + "\n"
        )

    for _ in range(height):
        complex_string += vertical + " " * width + vertical + "\n"

    complex_string += (
        bottom_left + horizontal * 3 + t_down + horizontal * (width - 4) + bottom_right
    )

    return complex_string
