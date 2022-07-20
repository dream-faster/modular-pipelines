from pprint import pformat


def pprint_indent(text, indent=" " * 4) -> None:
    text = pformat(text)
    print("".join([indent + l for l in text.splitlines(True)]))
