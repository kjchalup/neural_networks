""" Do instance segmentation on COCO. """
import matplotlib
matplotlib.use('Agg')
from pycocotools.coco import COCO
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
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

def get_img_and_mask(img_ids, im_size,  cat_id, coco):
    """ Choose an id at random from image_ids. Check whether corresponding mask
    contains at least 1% positive class. If so, crop out a random patch of
    im_size and return it. If not, return good_image=False.
    """
    img_id = np.random.choice(img_ids)
    image = coco.loadImgs(int(img_id))[0]
    annIds = coco.getAnnIds(imgIds=image['id'], catIds=[cat_id], iscrowd=None)
    image = io.imread('{}/{}'.format(imageDir, image['file_name']))
    ann = coco.loadAnns(annIds)
    mask = np.zeros((image.shape[0], image.shape[1]))
    for ann_single in ann:
        mask += coco.annToMask(ann_single)
    mask[mask > 1] = 1
    image = image / 255.
    if image.shape[0] > im_size[0] and image.shape[1] > im_size[1]:
        row = np.random.choice(image.shape[0] - im_size[0])
        col = np.random.choice(image.shape[1] - im_size[1])
        image = image[row:row+im_size[0], col:col+im_size[1]]
        mask = mask[row:row+im_size[0], col:col+im_size[1]]
        if mask.sum() > mask.size / 100.:
            return image, mask, True
    return None, None, False

def get_coco_batch(category, batch_size, im_size, coco, data_type='train'):
    # Get ids of current category.
    cat_id = coco.getCatIds(catNms=[category])[0]

    # Split into train/validation/test.
    st0 = np.random.get_state()
    np.random.seed(1)
    img_ids = np.random.permutation(coco.getImgIds(catIds=cat_id))
    np.random.set_state(st0)
    n_train = int(len(img_ids) * .8)
    n_valid = int(len(img_ids) * .1)
    if data_type == 'train':
        img_ids = img_ids[:n_train]
    elif data_type == 'val':
        img_ids = img_ids[n_train:n_train+n_valid]
    elif data_type == 'test':
        img_ids = img_ids[n_train+n_valid:]

    # Fetch the batch.
    ims = np.zeros((batch_size, im_size[0], im_size[1], 3))
    masks = np.zeros((batch_size, im_size[0], im_size[1], 1))
    for img_id in range(batch_size):
        good_image = False
        while not good_image:
            image, mask, good_image = get_img_and_mask(img_ids, im_size, cat_id, coco)
        if len(image.shape) == 2:
            ims[img_id] = skimage.color.gray2rgb(image)
        else:
            ims[img_id] = image
        masks[img_id, :, :, 0] = mask
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

        # Push the image through a resnet. Image shape should remain constant.
        y_pred = tf.layers.conv2d(
            y_pred, filters=self.n_filters[0], kernel_size=(1, 1), strides=(1, 1),
            padding='same', activation=None, reuse=self.reuse)
    
        for layer_id in range(8):
            with tf.variable_scope('7x7layer{}'.format(layer_id)):
                y_pred = bn_relu_conv(y_pred, self.is_training_tf,
                    n_filters=16, stride=(1, 1), bn=True,
                    kernel_size=(5, 5), residual=self.res, reuse=self.reuse)

        with tf.variable_scope('collapse'):
            y_pred = tf.nn.sigmoid(tf.layers.batch_normalization(
                y_pred, center=True, scale=True, training=self.is_training_tf))
            y_pred = tf.layers.conv2d(
                y_pred, filters=1, kernel_size=(1, 1), strides=(1, 1),
                padding='same', activation=None, reuse=self.reuse)
        tf.summary.histogram('output', y_pred)

        return y_pred


    def define_loss(self):
        """ Use weighted cross-entropy. The larger pos_weight, the more
        attention the loss pays to the positive class. """
        #loss_pos = tf.nn.sigmoid(-.1 * tf.reduce_mean(
        #    tf.boolean_mask(self.y_pred, self.y_tf)))
        #loss_neg = tf.nn.sigmoid(.1 * tf.reduce_mean(
        #    tf.boolean_mask(self.y_pred, tf.logical_not(self.y_tf))))
        loss_pos = tf.reduce_mean(tf.nn.sigmoid(-.1 * 
            tf.boolean_mask(self.y_pred, self.y_tf)))
        loss_neg = tf.reduce_mean(tf.nn.sigmoid(.1 *
            tf.boolean_mask(self.y_pred, tf.logical_not(self.y_tf))))
        tf.summary.scalar('loss_pos', loss_pos)
        tf.summary.scalar('loss_neg', loss_neg)
        return loss_pos + loss_neg

if __name__ == "__main__":
    """ Check that the network works as expected. Denoise MNIST. 
    Takes about a minute on a Titan X GPU.
    """
    im_shape = [200, 200] # Images will be reshaped to this shape.
    n_layers = 4
    batch_size = 80

    def fetch_data(batch_size, data_type):
        ims, masks = get_coco_batch(category='person', batch_size=batch_size,
            im_size=im_shape, coco=coco, data_type=data_type)
        return ims, masks.astype(bool)

    # Define the graph.
    y_tf = tf.placeholder(tf.bool, [None, im_shape[0], im_shape[1], 1])
    fcnn = SegmentNN(x_shape = im_shape + [3], y_tf=y_tf, y_channels=1,
        n_layers=n_layers, res=False, n_filters=np.array(
            [32] * n_layers), save_fname='segment_person3')


    # Create a Tensorflow session and train the net.
    with tf.Session() as sess:
        # Define the Tensorflow session, and its initializer op.
        sess.run(tf.global_variables_initializer())
        fcnn.saver.restore(sess, 'segment_person2')

        # Use a writer object for Tensorboard visualization.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/fcnn')
        writer.add_graph(sess.graph)

        # Fit the net.
        #fcnn.fit(sess, fetch_data, epochs=10000,
        #    batch_size=batch_size, lr=0.01, writer=writer, summary=summary)
        #fcnn.fit(sess, fetch_data, epochs=10000,
        #    batch_size=batch_size, lr=0.01, writer=writer, summary=summary)
        # Predict.
        ims_ts, masks_ts = fetch_data(batch_size, 'test')
        Y_pred = fcnn.predict(ims_ts, sess)

    # Show the results.
    plt.figure(figsize=(24, 9))
    for im_id_id, im_id in enumerate(range(batch_size-8, batch_size)):
        print(im_id)
        plt.subplot2grid((3, 8), (0, im_id_id))
        plt.imshow(ims_ts[im_id], interpolation='nearest', vmin=0, vmax=1)

        plt.subplot2grid((3, 8), (1, im_id_id))
        plt.imshow(ims_ts[im_id], interpolation='nearest', vmin=0, vmax=1)
        plt.imshow(Y_pred[im_id].squeeze() > .99, cmap='gray', alpha=.8)

        plt.subplot2grid((3, 8), (2, im_id_id))
        plt.imshow(ims_ts[im_id], interpolation='nearest', vmin=0, vmax=1)
        plt.imshow(masks_ts[im_id].squeeze() > .5, cmap='gray', alpha=.8)
    plt.savefig('segmentation_results.png')
