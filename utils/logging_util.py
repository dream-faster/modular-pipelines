import logging

FORMAT = "%(splits)s %(message)s"
# formatter = logging.Formatter(FORMAT)  # , defaults={"lines": None, "splits": None}
# logging.setFormatter(formatter)
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)  # configure root logger
