from __future__ import division
import tensorflow as tf
import numpy as np
import os
# import scipy.misc
import PIL.Image as pil
from SfMLearner import SfMLearner

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

tensorflow_shutup()

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_string("dataset_dir", "/media/patodichayan/DATA/733/PFinal/SfM/Code/Part2/Test/", "Dataset directory")
flags.DEFINE_string("output_dir", "/media/patodichayan/DATA/733/PFinal/SfM/Code/Part2/SFMLearner/Output_Pred/Depth/", "Output directory")
flags.DEFINE_string("ckpt_file", "/media/patodichayan/DATA/733/PFinal/SfM/Code/Part2/SFMLearner/Checkpoint/model.latest", "checkpoint file")
FLAGS = flags.FLAGS

def main(_):
    with open('/media/patodichayan/DATA/733/PFinal/SfM/Code/Part2/Misc/testfiles.txt', 'r') as f:
        test_file = f.readlines()
    	test_file = [l.rstrip() for l in test_file]
        
        test_files = []

        for filename in test_file:
   
            test_files.append(FLAGS.dataset_dir + filename)
        
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        basename = os.path.basename(FLAGS.ckpt_file)
        output_file = FLAGS.output_dir + '/' + basename
        sfm = SfMLearner()
        sfm.setup_inference(img_height=FLAGS.img_height,
                            img_width=FLAGS.img_width,
                            batch_size=FLAGS.batch_size,
                            mode='depth')
        saver = tf.train.Saver([var for var in tf.model_variables()]) 
      
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, FLAGS.ckpt_file)
            pred_all = []
            for t in range(0, len(test_files), FLAGS.batch_size):
                if t % 100 == 0:
                    print('processing %s: %d/%d' % (basename, t, len(test_files)))
                inputs = np.zeros(
                    (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3), 
                    dtype=np.uint8)
                for b in range(FLAGS.batch_size):
                    idx = t + b
                    if idx >= len(test_files):
                        break
                    fh = open(test_files[idx], 'r')
                    raw_im = pil.open(fh)
                    scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                    inputs[b] = np.array(scaled_im)
                    # im = scipy.misc.imread(test_files[idx])
                    # inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
                pred = sfm.inference(inputs, sess, mode='depth')
                for b in range(FLAGS.batch_size):
                    idx = t + b
                    if idx >= len(test_files):
                        break
                    pred_all.append(pred['depth'][b,:,:,0])
            np.save(output_file, pred_all)

if __name__ == '__main__':
    tf.app.run()
