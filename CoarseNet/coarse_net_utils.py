# TODO : fix rules
# pylint: disable=missing-function-docstring
"""Code for FineNet in paper "Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge" at ICB 2018
  https://arxiv.org/pdf/1712.09401.pdf

  If you use whole or partial function in this code, please cite paper:

  @inproceedings{Nguyen_MinutiaeNet,
    author    = {Dinh-Luan Nguyen and Kai Cao and Anil keras_backend. Jain},
    title     = {Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge},
    booktitle = {The 11th International Conference on Biometrics, 2018},
    year      = {2018},
    }
"""
from functools import partial, reduce
from multiprocessing import Pool
import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage, signal, sparse
from tensorflow.keras import models, layers, backend as keras_backend

from . import minutiae_net_utils


def sub_load_data(data, img_size, aug):
    img_name, dataset = data

    img = cv2.imread(dataset+'img_files/'+img_name+'.bmp', cv2.IMREAD_GRAYSCALE)

    seg = cv2.imread(dataset + 'seg_files/' + img_name + '.bmp', cv2.IMREAD_GRAYSCALE)

    if seg is None:
        seg = np.ones_like(img)

    ali = cv2.imread(dataset+'ori_files/'+img_name+'.jpg', cv2.IMREAD_GRAYSCALE)

    if ali is None:
        ali = np.zeros_like(img)

    mnt = np.array(minutiae_net_utils.mnt_reader(dataset+'mnt_files/'+img_name+'.mnt'), dtype=float)

    if any(img.shape != img_size):
        # random pad mean values to reach required shape

        if np.random.rand() < aug:
            tra = np.random.rand(2)*(np.array(img_size)-np.array(img.shape))
        else:
            tra = 0.5*(np.array(img_size)-np.array(img.shape))

        tra = tra.astype(np.int32)

        img_t = np.ones(img_size)*np.mean(img)
        seg_t = np.zeros(img_size)
        ali_t = np.ones(img_size)*np.mean(ali)

        img_t[tra[0]:tra[0]+img.shape[0], tra[1]:tra[1]+img.shape[1]] = img
        seg_t[tra[0]:tra[0]+img.shape[0], tra[1]:tra[1]+img.shape[1]] = seg
        ali_t[tra[0]:tra[0]+img.shape[0], tra[1]:tra[1]+img.shape[1]] = ali

        img = img_t
        seg = seg_t
        ali = ali_t
        mnt = mnt+np.array([tra[1], tra[0], 0])

    if np.random.rand() < aug:
        # random rotation [0 - 360] & translation img_size / 4
        rot = np.random.rand() * 360
        tra = (np.random.rand(2)-0.5) / 2 * img_size
        img = ndimage.rotate(img, rot, reshape=False, mode='reflect')
        img = ndimage.shift(img, tra, mode='reflect')
        seg = ndimage.rotate(seg, rot, reshape=False, mode='constant')
        seg = ndimage.shift(seg, tra, mode='constant')
        ali = ndimage.rotate(ali, rot, reshape=False, mode='reflect')
        ali = ndimage.shift(ali, tra, mode='reflect')
        mnt_r = minutiae_net_utils.point_rot(mnt[:, :2], rot/180*np.pi, img.shape, img.shape)
        mnt = np.column_stack((mnt_r+tra[[1, 0]], mnt[:, 2]-rot/180*np.pi))

    # only keep mnt that stay in pic & not on border
    mnt = mnt[(mnt[:, 0] >= 8)*(mnt[:, 0] < img_size[1]-8)
              * (mnt[:, 1] >= 8)*(mnt[:, 1] < img_size[0]-8), :]
    return img, seg, ali, mnt


USE_MULTIPROCESSING = False


def load_data(dataset, tra_ori_model, rand=False, aug=0.0, batch_size=1, sample_rate=None):

    if isinstance(dataset[0], str):
        img_name, folder_name, img_size = get_maximum_img_size_and_names(dataset, sample_rate)
    else:
        img_name, folder_name, img_size = dataset

    if rand:
        rand_idx = np.arange(len(img_name))
        np.random.shuffle(rand_idx)
        img_name = img_name[rand_idx]
        folder_name = folder_name[rand_idx]

    if batch_size > 1 and USE_MULTIPROCESSING is True:
        p = Pool(batch_size)

    p_sub_load_data = partial(sub_load_data, img_size=img_size, aug=aug)

    for i in range(0, len(img_name), batch_size):
        have_alignment = np.ones([batch_size, 1, 1, 1])
        image = np.zeros((batch_size, img_size[0], img_size[1], 1))
        segment = np.zeros((batch_size, img_size[0], img_size[1], 1))
        alignment = np.zeros((batch_size, img_size[0], img_size[1], 1))

        minutiae_w = np.zeros((batch_size, img_size[0]/8, img_size[1]/8, 1))-1
        minutiae_h = np.zeros((batch_size, img_size[0]/8, img_size[1]/8, 1))-1
        minutiae_o = np.zeros((batch_size, img_size[0]/8, img_size[1]/8, 1))-1

        batch_name = [img_name[(i+j) % len(img_name)] for j in range(batch_size)]
        batch_f_name = [folder_name[(i+j) % len(img_name)] for j in range(batch_size)]

        if batch_size > 1 and USE_MULTIPROCESSING is True:
            results = p.map(p_sub_load_data, list(zip(batch_name, batch_f_name)))
        else:
            results = list(map(p_sub_load_data, list(zip(batch_name, batch_f_name))))

        for j in range(batch_size):
            img, seg, ali, mnt = results[j]
            if np.sum(ali) == 0:
                have_alignment[j, 0, 0, 0] = 0
            image[j, :, :, 0] = img / 255.0
            segment[j, :, :, 0] = seg / 255.0
            alignment[j, :, :, 0] = ali / 255.0
            minutiae_w[j, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int), 0] = mnt[:, 0] % 8
            minutiae_h[j, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int), 0] = mnt[:, 1] % 8
            minutiae_o[j, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int), 0] = mnt[:, 2]

        # get seg
        label_seg = segment[:, ::8, ::8, :]
        label_seg[label_seg > 0] = 1
        label_seg[label_seg <= 0] = 0
        minutiae_seg = (minutiae_o != -1).astype(float)

        # get ori & mnt
        orientation = tra_ori_model.predict(alignment)
        orientation = orientation/np.pi*180+90
        orientation[orientation >= 180.0] = 0.0  # orientation [0, 180)
        minutiae_o = minutiae_o/np.pi*180+90  # [90, 450)
        minutiae_o[minutiae_o > 360] = minutiae_o[minutiae_o >
                                                  360]-360  # to current coordinate system [0, 360)
        minutiae_ori_o = np.copy(minutiae_o)  # copy one
        minutiae_ori_o[minutiae_ori_o >= 180] = minutiae_ori_o[minutiae_ori_o >=
                                                               180]-180  # for strong ori label [0,180)

        # ori 2 gaussian
        gaussian_pdf = signal.gaussian(361, 3)
        y = np.reshape(np.arange(1, 180, 2), [1, 1, 1, -1])
        delta = np.array(np.abs(orientation - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori = gaussian_pdf[delta]

        # ori_o 2 gaussian
        delta = np.array(np.abs(minutiae_ori_o - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori_o = gaussian_pdf[delta]

        # mnt_o 2 gaussian
        y = np.reshape(np.arange(1, 360, 2), [1, 1, 1, -1])
        delta = np.array(np.abs(minutiae_o - y), dtype=int)
        delta = np.minimum(delta, 360-delta)+180
        label_mnt_o = gaussian_pdf[delta]

        # w 2 gaussian
        gaussian_pdf = signal.gaussian(17, 2)
        y = np.reshape(np.arange(0, 8), [1, 1, 1, -1])
        delta = (minutiae_w-y+8).astype(int)
        label_mnt_w = gaussian_pdf[delta]

        # h 2 gaussian
        delta = (minutiae_h-y+8).astype(int)
        label_mnt_h = gaussian_pdf[delta]

        # mnt cls label -1:neg, 0:no care, 1:pos
        label_mnt_s = np.copy(minutiae_seg)
        label_mnt_s[label_mnt_s == 0] = -1  # neg to -1
        label_mnt_s = (label_mnt_s+ndimage.maximum_filter(label_mnt_s,
                       size=(1, 3, 3, 1)))/2  # around 3*3 pos -> 0

        # apply segmentation
        label_ori = label_ori * label_seg * have_alignment
        label_ori_o = label_ori_o * minutiae_seg
        label_mnt_o = label_mnt_o * minutiae_seg
        label_mnt_w = label_mnt_w * minutiae_seg
        label_mnt_h = label_mnt_h * minutiae_seg
        yield image, label_ori, label_ori_o, label_seg, label_mnt_w, label_mnt_h, label_mnt_o, label_mnt_s, batch_name

    if batch_size > 1 and USE_MULTIPROCESSING is True:
        p.close()
        p.join()
    return


def merge_mul(x):
    return reduce(lambda x, y: x*y, x)


def merge_sum(x):
    return reduce(lambda x, y: x+y, x)


def reduce_sum(x):
    return keras_backend.sum(x, axis=-1, keepdims=True)

# Group with depth


def merge_concat(x):
    return tf.concat(x, 3)


def select_max(x):
    x = x / (keras_backend.max(x, axis=-1, keepdims=True)+keras_backend.epsilon())
    x = tf.where(tf.greater(x, 0.999), x, tf.zeros_like(x))  # select the biggest one
    # prevent two or more ori is selected
    x = x / (keras_backend.sum(x, axis=-1, keepdims=True)+keras_backend.epsilon())
    return x


kernal2angle = np.reshape(
    np.arange(1, 180, 2, dtype=float),
    [1, 1, 1, 90]) / 90. * np.pi  # 2angle = angle*2
sin2angle, cos2angle = np.sin(kernal2angle), np.cos(kernal2angle)


def ori2angle(ori):
    sin2angle_ori = keras_backend.sum(ori*sin2angle, -1, keepdims=True)
    cos2angle_ori = keras_backend.sum(ori*cos2angle, -1, keepdims=True)
    modulus_ori = keras_backend.sqrt(keras_backend.square(
        sin2angle_ori)+keras_backend.square(cos2angle_ori))
    return sin2angle_ori, cos2angle_ori, modulus_ori


# find highest peak using gaussian
def ori_highest_peak(y_pred, length=180):
    glabel = minutiae_net_utils.gausslabel(length=length, stride=2).astype(np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    ori_gau = keras_backend.conv2d(y_pred, glabel, padding='same')
    return ori_gau


def ori_acc_delta_k(y_true, y_pred, k=10, max_delta=180):
    # get ROI
    label_seg = keras_backend.sum(y_true, axis=-1)
    label_seg = tf.cast(tf.greater(label_seg, 0), tf.float32)
    # get pred angle
    angle = keras_backend.cast(keras_backend.argmax(
        ori_highest_peak(y_pred, max_delta),
        axis=-1),
        dtype=tf.float32) * 2.0 + 1.0
    # get gt angle
    angle_t = keras_backend.cast(keras_backend.argmax(y_true, axis=-1), dtype=tf.float32)*2.0+1.0
    # get delta
    angle_delta = keras_backend.abs(angle_t - angle)
    acc = tf.less_equal(keras_backend.minimum(angle_delta, max_delta-angle_delta), k)
    acc = keras_backend.cast(acc, dtype=tf.float32)
    # apply ROI
    acc = acc*label_seg
    acc = keras_backend.sum(acc) / (keras_backend.sum(label_seg)+keras_backend.epsilon())
    return acc


def ori_acc_delta_10(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 10)


def ori_acc_delta_20(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 20)


def mnt_acc_delta_10(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 10, 360)


def mnt_acc_delta_20(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 20, 360)


def seg_acc_pos(y_true, y_pred):
    y_true = tf.where(tf.less(y_true, 0.0), tf.zeros_like(y_true), y_true)
    acc = keras_backend.cast(keras_backend.equal(
        y_true, keras_backend.round(y_pred)), dtype=tf.float32)
    acc = keras_backend.sum(acc * y_true) / (keras_backend.sum(y_true)+keras_backend.epsilon())
    return acc


def seg_acc_neg(y_true, y_pred):
    y_true = tf.where(tf.less(y_true, 0.0), tf.zeros_like(y_true), y_true)
    acc = keras_backend.cast(keras_backend.equal(
        y_true, keras_backend.round(y_pred)), dtype=tf.float32)
    acc = keras_backend.sum(
        acc * (1-y_true)) / (keras_backend.sum(1-y_true)+keras_backend.epsilon())
    return acc


def seg_acc_all(y_true, y_pred):
    y_true = tf.where(tf.less(y_true, 0.0), tf.zeros_like(y_true), y_true)
    return keras_backend.mean(keras_backend.equal(y_true, keras_backend.round(y_pred)))


def mnt_mean_delta(y_true, y_pred):
    # get ROI
    label_seg = keras_backend.sum(y_true, axis=-1)
    label_seg = tf.cast(tf.greater(label_seg, 0), tf.float32)
    # get pred pos
    pos = keras_backend.cast(keras_backend.argmax(y_pred, axis=-1), dtype=tf.float32)
    # get gt pos
    pos_t = keras_backend.cast(keras_backend.argmax(y_true, axis=-1), dtype=tf.float32)
    # get delta
    pos_delta = keras_backend.abs(pos_t - pos)
    # apply ROI
    pos_delta = pos_delta*label_seg
    mean_delta = keras_backend.sum(
        pos_delta) / (keras_backend.sum(label_seg)+keras_backend.epsilon())
    return mean_delta


# currently can only produce one each time
def label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5):
    mnt_s_out = np.squeeze(mnt_s_out)
    mnt_w_out = np.squeeze(mnt_w_out)
    mnt_h_out = np.squeeze(mnt_h_out)
    mnt_o_out = np.squeeze(mnt_o_out)
    assert len(
        mnt_s_out.shape) == 2 and len(
        mnt_w_out.shape) == 3 and len(
        mnt_h_out.shape) == 3 and len(
        mnt_o_out.shape) == 3

    # get cls results
    mnt_sparse = sparse.coo_matrix(mnt_s_out > thresh)
    mnt_list = np.array(list(zip(mnt_sparse.row, mnt_sparse.col)), dtype=np.int32)
    if mnt_list.shape[0] == 0:
        return np.zeros((0, 5))

    # get regression results
    mnt_w_out = np.argmax(mnt_w_out, axis=-1)
    mnt_h_out = np.argmax(mnt_h_out, axis=-1)
    mnt_o_out = np.argmax(mnt_o_out, axis=-1)  # TODO: use ori_highest_peak(np version)

    # get final mnt
    mnt_final = np.zeros((len(mnt_list), 5))
    mnt_final[:, 0] = mnt_sparse.col*8 + mnt_w_out[mnt_list[:, 0], mnt_list[:, 1]]
    mnt_final[:, 1] = mnt_sparse.row*8 + mnt_h_out[mnt_list[:, 0], mnt_list[:, 1]]
    mnt_final[:, 2] = (mnt_o_out[mnt_list[:, 0], mnt_list[:, 1]]*2-89.)/180*np.pi
    mnt_final[mnt_final[:, 2] < 0.0, 2] = mnt_final[mnt_final[:, 2] < 0.0, 2]+2*np.pi
    # New one
    mnt_final[:, 2] = (-mnt_final[:, 2]) % (2*np.pi)
    mnt_final[:, 3] = mnt_s_out[mnt_list[:, 0], mnt_list[:, 1]]
    mnt_final[:, 4] = None

    return mnt_final


# image normalization
def img_normalization(img_input, m_0=0.0, var0=1.0):
    m = keras_backend.mean(img_input, axis=[1, 2, 3], keepdims=True)
    var = keras_backend.var(img_input, axis=[1, 2, 3], keepdims=True)
    after = keras_backend.sqrt(var0*tf.square(img_input-m)/var)
    image_n = tf.where(tf.greater(img_input, m), m_0+after, m_0-after)
    return image_n

# atan2 function


def atan2(y_x):
    y, x = y_x[0], y_x[1]+keras_backend.epsilon()
    atan = tf.atan(y/x)
    angle = tf.where(tf.greater(x, 0.0), atan, tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x, 0.0),  tf.greater_equal(y, 0.0)), atan+np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x, 0.0),  tf.less(y, 0.0)), atan-np.pi, angle)
    return angle

# traditional orientation estimation


def get_orientation(image, stride=8, window=17):
    with tf.name_scope('orientation'):
        assert image.get_shape().as_list()[3] == 1, 'Images must be grayscale'
        strides = [1, stride, stride, 1]
        E = np.ones([window, window, 1, 1])
        sobelx = np.reshape(
            np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]],
                     dtype=float),
            [3, 3, 1, 1])
        sobely = np.reshape(
            np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]],
                     dtype=float),
            [3, 3, 1, 1])
        gaussian = np.reshape(minutiae_net_utils.gaussian2d((5, 5), 1), [5, 5, 1, 1])
        with tf.name_scope('sobel_gradient'):
            i_x = tf.nn.conv2d(image, sobelx, strides=[1, 1, 1, 1], padding='SAME', name='sobel_x')
            i_y = tf.nn.conv2d(image, sobely, strides=[1, 1, 1, 1], padding='SAME', name='sobel_y')
        with tf.name_scope('eltwise_1'):
            i_x2 = tf.multiply(i_x, i_x, name='IxIx')
            i_y2 = tf.multiply(i_y, i_y, name='IyIy')
            i_xy = tf.multiply(i_x, i_y, name='IxIy')
        with tf.name_scope('range_sum'):
            g_xx = tf.nn.conv2d(i_x2, E, strides=strides, padding='SAME', name='Gxx_sum')
            g_yy = tf.nn.conv2d(i_y2, E, strides=strides, padding='SAME', name='Gyy_sum')
            g_xy = tf.nn.conv2d(i_xy, E, strides=strides, padding='SAME', name='Gxy_sum')
        with tf.name_scope('eltwise_2'):
            g_xx_g_yy = tf.subtract(g_xx, g_yy, name='Gxx_Gyy')
            theta = atan2([2*g_xy, g_xx_g_yy]) + np.pi
        # two-dimensional low-pass filter: Gaussian filter here
        with tf.name_scope('gaussian_filter'):
            phi_x = tf.nn.conv2d(
                tf.cos(theta),
                gaussian, strides=[1, 1, 1, 1],
                padding='SAME', name='gaussian_x')
            phi_y = tf.nn.conv2d(
                tf.sin(theta),
                gaussian, strides=[1, 1, 1, 1],
                padding='SAME', name='gaussian_y')
            theta = atan2([phi_y, phi_x])/2
    return theta


def get_tra_ori():
    img_input = layers.Input(shape=(None, None, 1))
    theta = layers.Lambda(get_orientation)(img_input)
    model = models.Model(inputs=[img_input, ], outputs=[theta, ])
    return model


def get_maximum_img_size_and_names(dataset, sample_rate=None):

    if isinstance(dataset, str):
        dataset = [dataset]
    if sample_rate is None:
        sample_rate = [1]*len(dataset)
    img_name, folder_name, img_size = [], [], []

    for folder, rate in zip(dataset, sample_rate):
        _, img_name_t = minutiae_net_utils.get_files_in_folder(folder, 'img_files/*'+'.png')
        img_name.extend(img_name_t.tolist()*rate)
        folder_name.extend([folder]*img_name_t.shape[0]*rate)

        img_size.append(
            np.array(
                cv2.imread(folder + 'img_files/' + img_name_t[0] + '.png',
                           cv2.IMREAD_GRAYSCALE).shape))

    img_name = np.asarray(img_name)
    folder_name = np.asarray(folder_name)
    img_size = np.max(np.asarray(img_size), axis=0)
    # let img_size % 8 == 0
    img_size = np.array(np.ceil(img_size / 8) * 8, dtype=np.int32)
    return img_name, folder_name, img_size
