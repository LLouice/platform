import logging


def init_logger():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')


class CustomFormatter(logging.Formatter):

    Black = "\u001b[30m"
    Green = "\u001b[32m"
    Red = "\u001b[31m"
    BoldRed = "\u001b[1m\u001b[31m"
    Yellow = "\u001b[33m"
    Blue = "\u001b[34m"
    Magenta = "\u001b[35m"
    Cyan = "\u001b[36m"
    White = "\u001b[37m"
    Reset = "\u001b[0m"
    # format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format_str = "%(asctime)s - {}%(levelname)s{} - %(message)s (%(filename)s:%(lineno)d)"

    # FORMATS = {
    #     logging.DEBUG: Magenta + format_str + Reset,
    #     logging.INFO: Blue + format_str + Reset,
    #     logging.WARNING: Yellow + format_str + Reset,
    #     logging.ERROR: Red + format_str + Reset,
    #     logging.CRITICAL: BoldRed + format_str + Reset
    # }

    FORMATS = {
        logging.DEBUG: format_str.format(Magenta, Reset),
        logging.INFO: format_str.format(Green, Reset),
        logging.WARNING: format_str.format(Yellow, Reset),
        logging.ERROR: format_str.format(Red, Reset),
        logging.CRITICAL: format_str.format(BoldRed, Reset),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    return logger
