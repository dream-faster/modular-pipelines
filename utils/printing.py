from pprint import pformat
import textwrap


def pprint_indent(text, indent=" " * 4 + "┃ ") -> None:
    text = pformat(text)
    print("".join([indent + l for l in text.splitlines(True)]))


class PrintFormats:
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


def multi_line_print(text: str, level: int = 0) -> None:

    base_indent = " " * 4

    if level == 0:
        initial_indent = base_indent + "┣━━━ "
        subsequent_indent = base_indent + "┃ "
    if level == 1:
        initial_indent = base_indent + "┃  ├── "
        subsequent_indent = base_indent + "┃  │   "

    print(
        textwrap.fill(
            text,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            width=100,
        )
    )
