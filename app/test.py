from settings import PLOTS_DIR, TRAINING_DIR, NPY_EXT

import os
import numpy as np
from model import kohonen as k
import matplotlib.pyplot as plt


def test(shape, input_data, max_iterations, save_as):

    # Save training input
    # ------------------
    np.save(os.path.join(TRAINING_DIR, f"{save_as}{NPY_EXT}"), input_data)

    # Training
    # --------
    model = k.Kohonen(shape, max_iterations=max_iterations, debug_mode=False)
    message = model.train(input_data)
    print(message)

    weights = model.get_weights()

    # Saves under 'saved_models/' dir under name '<name>.npy'
    # --------------------------------------------------------
    # model.save(save_as)

    # Testing load weights
    # -------------------
    # model.load(save_as)

    # Predictions / Labels input data to grid nodes positions according to
    # best match i.e. closest node's index is assigned against testing item.
    # -----------------------------------------------------------------------
    predictions = model.predict(input_data)
    print(predictions)

    # Save grid to .png image.
    # ------------------------
    plt.imsave(os.path.join(PLOTS_DIR, f"{save_as}.png"), weights)


if __name__ == "__main__":
    sample_count = 10

    # Uncomment to run via command line. ```python test.py`

    input_data = np.random.randint(0, 255, size=(sample_count, 3)) / 255.0
    # plt.imsave(f"input_data.png", input_data)

    shape = [10, 10]
    test(shape, input_data, max_iterations=100, save_as="sample_10X10_100N")
    # test(shape, input_data, max_iterations=200, save_as="sample_10X10_200N")
    # test(shape, input_data, max_iterations=500, save_as="sample_10X10_500N")

    # shape = [100, 100]
    # test(shape, input_data, max_iterations=2, save_as="sample_100X100_1000N")