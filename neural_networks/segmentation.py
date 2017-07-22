""" Do instance segmentation on COCO. """
import matplotlib
matplotlib.use('Agg')
from pycocotools.coco import COCO
import numpy as np
import skimage.transform
import skimage.color
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import tensorflow as tf

from neural_networks.fcnn import FCNN, bn_relu_conv


home = '/home/kchalupk/'
dataDir = home + 'projects/COCO/coco/'
dataType = 'train2014'
imageDir = home + 'projects/COCO/' + dataType
annFile = dataDir + 'annotations/instances_{}.json'.format(dataType)

# Initialize COCO api for instance annotations.
coco=COCO(annFile)

def get_img_and_mask(img_id, cat_id, coco):
    img = coco.loadImgs(int(img_id))[0]
    image = io.imread('{}/{}'.format(imageDir, img['file_name']))
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=[cat_id], iscrowd=None)
    ann = coco.loadAnns(annIds)
    mask = np.zeros((image.shape[0], image.shape[1]))
    for ann_single in ann:
        mask += coco.annToMask(ann_single)
    mask[mask > 1] = 1
    return image, mask

def get_coco_batch(category, batch_size, im_size, coco, data_type='traffic light'):
    cat_id = coco.getCatIds(catNms=[category])[0]
    st0 = np.random.get_state()
    np.random.seed(1)
    img_ids = np.random.permutation(coco.getImgIds(catIds=cat_id))
    np.random.set_state(st0)
    n_train = int(len(img_ids) * .8)
    n_valid = int(len(img_ids) * .1)
    if data_type == 'train':
        img_ids = np.random.choice(img_ids[:n_train], batch_size, replace=True)
    elif data_type == 'val':
        img_ids = np.random.choice(img_ids[n_train:n_train+n_valid],
            batch_size, replace=True)
    elif data_type == 'test':
        img_ids = np.random.choice(img_ids[n_train+n_valid:],
            batch_size, replace=True)
    ims = np.zeros((batch_size, im_size[0], im_size[1], 3))
    masks = np.zeros((batch_size, im_size[0], im_size[1], 1))
    for img_id_id, img_id in enumerate(img_ids):
        img, mask = get_img_and_mask(img_id, cat_id, coco)
        img = skimage.transform.resize(img, im_size[:2])
        if len(img.shape) == 2:
            ims[img_id_id] = skimage.color.gray2rgb(img)
        else:
            ims[img_id_id] = img
        masks[img_id_id, :, :, 0] = skimage.transform.resize(mask, im_size[:2])
    return ims, masks

class SegmentNN(FCNN):
    """ A segmentation neural net consisting of a subsampling module, and FCNN 
    module, and a deconvolution module. Uses the cross-entropy loss. """
    def __init__(self, pos_weight=1., **kwargs):
        self.pos_weight = pos_weight # Weight for the positive class.
        FCNN.__init__(self, **kwargs)
    
    def predict(self, x, sess):
        """ Compute the output for given data.

        Args:
            x (n_samples, h, w, c): Input data.
            sess: Tensorflow session.

        Returns:
            y (n_samples, y_dim): Predicted outputs.
        """
        feed_dict = {self.x_tf: x, self.is_training_tf: False}
        y_pred = sess.run(tf.nn.sigmoid(self.y_pred), feed_dict)
        return y_pred

    def define_fcnn(self, **kwargs):
        """ Define the segmentation network. """
        y_pred = self.x_tf

        # During training only: apply random transformations to the input image
        # to augment the training set size and reduce overfitting.
        #y_pred = tf.cond(self.is_training_tf,
        #        true_fn=lambda: tf.map_fn(tf.image.random_flip_left_right, self.x_tf),
        #        false_fn=lambda: y_pred)

        #y_pred = tf.cond(self.is_training_tf,
        #        true_fn=lambda: tf.map_fn(
        #            lambda img: tf.image.random_brightness(img, .3), self.x_tf),
        #        false_fn=lambda: y_pred)

        #y_pred = tf.cond(self.is_training_tf,
        #        true_fn=lambda: tf.map_fn(
        #            lambda img: tf.image.random_contrast(img, .9, 1/.9), self.x_tf),
        #        false_fn=lambda: y_pred)

        # Downsample the images. E.g. a 224x224x3 image should be shrunk to
        # 56x56x64 after these operations.
        with tf.variable_scope('downsample'):
            y_pred = bn_relu_conv(y_pred, self.is_training_tf,
                n_filters=self.n_filters[0], residual=False,
                kernel_size=7, stride=(2, 2))
            y_pred = tf.layers.max_pooling2d(
                y_pred, pool_size=2, strides=(2, 2), padding='same')

        # Push the image through a resnet. Image shape should remain constant.
        for layer_id in range(self.n_layers):
            with tf.variable_scope('layer{}'.format(layer_id)):
                y_pred = bn_relu_conv(y_pred, self.is_training_tf,
                    n_filters=self.n_filters[layer_id], stride=(1, 1), 
                    residual=self.res, reuse=self.reuse)

        # Deconvolve the image back to its original shape.
        with tf.variable_scope('upsample'):
            y_pred = bn_relu_conv(y_pred, self.is_training_tf,
                n_filters=1, kernel_size=1, residual=False, reuse=self.reuse)
            y_pred = tf.image.resize_images(y_pred, self.x_shape[:2])

        return y_pred
    

    def define_loss(self):
        """ Use weighted cross-entropy. The larger pos_weight, the more
        attention the loss pays to the positive class. """
        loss = tf.nn.weighted_cross_entropy_with_logits(
            self.y_tf, self.y_pred, pos_weight=self.pos_weight)
        return tf.reduce_mean(loss)

if __name__ == "__main__":
    """ Check that the network works as expected. Denoise MNIST. 
    Takes about a minute on a Titan X GPU.
    """
    im_shape = [224, 224] # Images will be reshaped to this shape.
    n_layers = 4

    def fetch_data(batch_size, data_type):
        ims, masks = get_coco_batch(category='person', batch_size=batch_size,
            im_size=im_shape, coco=coco, data_type=data_type)
        return ims, masks

    # Define the graph.
    fcnn = SegmentNN(x_shape = im_shape + [3], y_channels=1,
        n_layers=n_layers, res=False, n_filters=np.array([96] * n_layers),
        save_fname='segment_nn', pos_weight=1.)


    # Create a Tensorflow session and train the net.
    with tf.Session() as sess:
        # Define the Tensorflow session, and its initializer op.
        sess.run(tf.global_variables_initializer())
        #fcnn.saver.restore(sess, 'segment_nn')

        # Use a writer object for Tensorboard visualization.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/fcnn')
        writer.add_graph(sess.graph)

        # Fit the net.
        #fcnn.fit(sess, fetch_data, epochs=100,
        #    batch_size=256, lr=0.1, writer=writer, summary=summary)

        # Predict.
        ims_ts, masks_ts = fetch_data(640, 'test')
        Y_pred = fcnn.predict(ims_ts, sess)

    # Show the results.
    plt.figure(figsize=(24, 9))
    for im_id, im in enumerate(ims_ts[:8]):
        plt.subplot2grid((3, 8), (0, im_id))
        plt.imshow(im, interpolation='nearest')

        plt.subplot2grid((3, 8), (1, im_id))
        plt.imshow(im, interpolation='nearest')
        plt.imshow(Y_pred[im_id].squeeze(), cmap='gray', alpha=.8)

        plt.subplot2grid((3, 8), (2, im_id))
        plt.imshow(im, interpolation='nearest')
        plt.imshow(masks_ts[im_id].squeeze() > .5, cmap='gray', alpha=.8)
    plt.savefig('segmentation_results.png')
