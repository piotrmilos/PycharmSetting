import numpy as np

def my_fast_warp(img, tf, channels=3, output_shape=(50, 50), mode='constant', order=1, dtype='float32'):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    mode = 'nearest'
    m = tf.params
    res = np.zeros(shape=(output_shape[0], output_shape[1], channels), dtype=dtype)
    from scipy.ndimage import affine_transform
    trans, offset = m[:2, :2], (m[0, 2], m[1, 2])
    for c in range(channels):
        res[:, :, c] = affine_transform(img[:, :, c], trans, offset=offset, output_shape=output_shape, mode=mode, order=order)
    return res
