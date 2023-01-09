import os


def save_model(nn_model, path, name) -> None:

    if not os.path.exists(path):
        os.makedirs(path)

    fn = os.path.join(path, name)
    nn_model.save(fn)
