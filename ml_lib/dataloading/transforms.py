# WARNING! Taken from plankton!!!

import numpy as np
import skimage.transform


tform_identity = skimage.transform.AffineTransform()


def uint_to_float(img):
    return 1 - (img / np.float32(255.0))

def extract_image_patch(chunk_dst, img):
    """
    extract a correctly sized patch from img and place it into chunk_dst,
    which assumed to be preinitialized to zeros.
    """
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 127
    # img[-1, :] = 127
    # img[:, 0] = 127
    # img[:, -1] = 127

    p_x, p_y = chunk_dst.shape
    im_x, im_y = img.shape

    offset_x = (im_x - p_x) // 2
    offset_y = (im_y - p_y) // 2

    if offset_x < 0:
        cx = slice(-offset_x, -offset_x + im_x)
        ix = slice(0, im_x)
    else:
        cx = slice(0, p_x)
        ix = slice(offset_x, offset_x + p_x)

    if offset_y < 0:
        cy = slice(-offset_y, -offset_y + im_y)
        iy = slice(0, im_y)
    else:
        cy = slice(0, p_y)
        iy = slice(offset_y, offset_y + p_y)

    chunk_dst[cx, cy] = uint_to_float(img[ix, iy])



def patches_gen(images, labels, patch_size=(50, 50), chunk_size=4096, num_chunks=100, rng=np.random):
    p_x, p_y = patch_size

    for n in xrange(num_chunks):
        indices = rng.randint(0, len(images), chunk_size)

        chunk_x = np.zeros((chunk_size, p_x, p_y), dtype='float32')
        chunk_y = np.zeros((chunk_size,), dtype='float32')

        for k, idx in enumerate(indices):
            img = images[indices[k]]
            extract_image_patch(chunk_x[k], img)
            chunk_y[k] = labels[indices[k]]

        yield chunk_x, chunk_y


def patches_gen_ordered(images, patch_size=(50, 50), chunk_size=4096):
    p_x, p_y = patch_size

    num_images = len(images)
    num_chunks = int(np.ceil(num_images / float(chunk_size)))

    idx = 0

    for n in xrange(num_chunks):
        chunk_x = np.zeros((chunk_size, p_x, p_y), dtype='float32')
        chunk_length = chunk_size

        for k in xrange(chunk_size):
            if idx >= num_images:
                chunk_length = k
                break

            img = images[idx]
            extract_image_patch(chunk_x[k], img)
            idx += 1

        yield chunk_x, chunk_length


## augmentation

def fast_warp(img, tf, output_shape=(50, 50), mode='constant', order=1, dtype='float32'):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params # tf._matrix is
    res = np.zeros(shape=(output_shape[0], output_shape[1], 3), dtype=dtype)
    res[:, :, 0] = skimage.transform._warps_cy._warp_fast(img[:, :, 0], m, output_shape=output_shape, mode=mode, order=order)
    res[:, :, 1] = skimage.transform._warps_cy._warp_fast(img[:, :, 1], m, output_shape=output_shape, mode=mode, order=order)
    res[:, :, 2] = skimage.transform._warps_cy._warp_fast(img[:, :, 2], m, output_shape=output_shape, mode=mode, order=order)
    return res


def build_centering_transform(image_shape, target_shape=(50, 50)):
    rows, cols = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    # TODO: why this -0.5 here?
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def build_center_uncenter_transforms2(width, height):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array([width, height]) / 2.0
    tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False):
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    return tform_augment


def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True, allow_stretch=False, rng=np.random):
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip:
        flip = (rng.randint(2) > 0) # flip half of the time
    else:
        flip = False

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch = np.exp(rng.uniform(*log_stretch_range))
        zoom_x = zoom * stretch
        zoom_y = zoom / stretch
    elif allow_stretch is True: # avoid bugs, f.e. when it is an integer
        zoom_x = np.exp(rng.uniform(*log_zoom_range))
        zoom_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    area_factor = zoom_x * zoom_y
    return build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip), area_factor


def perturb(img, augmentation_params, target_shape=(50, 50), rng=np.random):
    assert(img.shape[2] < 35)
    w = (img.shape[0], img.shape[1])
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 0.5
    # img[-1, :] = 0.5
    # img[:, 0] = 0.5
    # img[:, -1] = 0.5
    tform_centering = build_centering_transform(w, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(img.shape[0:2])
    tform_augment = random_perturbation_transform(rng=rng, **augmentation_params)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    return fast_war(img, tform_centering + tform_augment, output_shape=target_shape, mode='constant').astype('float32')


def transformation(img, spec, perturb):
    # inter_size = spec['inter_size']
    mean = spec['mean']
    std = spec['std']
    img = img.astype(dtype='float32')
    # img /= 255.0

    def apply_mean_std(img):
        if mean is not None:
            assert (len(mean) == img.shape[2])
            for channel in xrange(img.shape[2]):
                img[:, :, channel] -= mean[channel]

        if std is not None:
            assert (len(std) == img.shape[2])
            for channel in xrange(img.shape[2]):
                img[:, :, channel] /= std[channel]
        return img

    if perturb:
        img = perturb(img, spec['augmentation_params'], target_shape=(spec['target_h'], spec['target_w']))

    # imgs.append(img)
    # img = np.copy(img)

    # PCA
    if spec['pca_data'] is not None:
        evs, U = spec['pca_data']
        ls = evs.astype(float) * np.random.normal(scale=spec['pca_scale'], size=evs.shape[0])
        noise = U.dot(ls).reshape((1, 1, evs.shape[0]))
        # print evs, ls, U
        # print 'noise', noise
        img += noise

    img = apply_mean_std(img)

    # img shape is H x W x C
    return img