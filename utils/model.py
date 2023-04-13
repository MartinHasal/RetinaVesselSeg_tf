import os
import tensorflow as tf

def save_model(nn_model, path, name) -> None:

    if not os.path.exists(path):
        os.makedirs(path)
    
    #save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    
    fn = os.path.join(path, name)
    nn_model.save(fn) # , options=save_option)
