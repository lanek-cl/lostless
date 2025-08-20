# train_model.py
import logging
import sys
import time

from functions import train_model

# Configure logging
logging.basicConfig(
    filename="results.log",
    filemode="a",  # Append to the file
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Create a custom stream to redirect print statements
class PrintLogger:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        # Skip empty lines
        if message.strip():
            self.level(message.strip())

    def flush(self):
        pass  # No need to implement flush for this example


# Redirect stdout and stderr
sys.stdout = PrintLogger(logging.info)
sys.stderr = PrintLogger(logging.error)


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
