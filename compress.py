# train_model.py
import time

import joblib


def compress_models(sample_size):
    # Load the saved model and encoder
    model_path = f"../lostless_data/models/rf_model_{sample_size}.joblib"
    encoder_path = f"../lostless_data/encoders/encoder_{sample_size}.joblib"
    clf = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    joblib.dump(clf, model_path, compress=("zlib", 3))
    joblib.dump(encoder, encoder_path, compress=("zlib", 3))


def main():
    start = time.time()
    compress_models(755349)
    stop = time.time()
    print(f"Training time: {stop - start:.2f} seconds")


if __name__ == "__main__":
    main()
