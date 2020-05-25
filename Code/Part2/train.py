from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from SfMLearner import SfMLearner
import os
def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass


flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "/media/patodichayan/DATA/733/PFinal/SfMData", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("smooth_weight", 0.5, "Weight for smoothness")
flags.DEFINE_float("explain_reg_weight", 0.0, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("num_source", None, "number of source images")
flags.DEFINE_integer("num_scales", None, "number of used image scales")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("max_steps", 200000, "Maximum number of training iterations")
flags.DEFINE_integer("step_update", 14000, "Update learning rate.")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
FLAGS = flags.FLAGS

def main(_):
    tensorflow_shutup()
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
        
    sfm = SfMLearner()
    sfm.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
