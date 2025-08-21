# train_model.py
import sys
import time
from loguru import logger

from functions import train_model

# Configure loguru to log to a file (append mode)
logger.add(
    "results.log",
    rotation="10 MB",   # Rotate after 10MB, can be time-based too
    #retention="7 days", # Keep logs for 7 days
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True,
    format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
)

# Redirect stdout and stderr to loguru
class StreamToLogger:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():
            self.level(message.strip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logger.info)
sys.stderr = StreamToLogger(logger.error)


def main():
    sample_size = int(sys.argv[1])
    path = sys.argv[2]
    model = sys.argv[3]
    start = time.time()
    report = train_model(sample_size, path, model)
    print(report)
    stop = time.time()
    print(f"Training time: {stop - start:.2f} seconds")


if __name__ == "__main__":
    main()
