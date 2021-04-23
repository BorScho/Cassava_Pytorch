def snapmix_batch_loss(
    is_augmented,
    label_batch,
    y_pred,
    label_batch2=None,
    box_weights1=None,
    box_weights2=None,
):
    """
    Calculates the loss for snap-mix algorithm if is_augmented == True, calculates sparse-categorical-crossentropy loss, if is_augmented == False

    Args:
        is_augmented (bool) : determines if snap-mix loss function is used or not
        label_batch : true labels
        y_pred : predicted labels
        label_batch2 : labels of patched-in images
        box_weights1 : semantic box weights of patched-into images
        box_weights2 : semantic box weights of patched-in images

    Returns:
        snap-mix loss or sparse-categorical-crossentropy loss
    """
    if is_augmented:
        loss1 = tf.keras.losses.sparse_categorical_crossentropy(label_batch, y_pred)
        loss2 = tf.keras.losses.sparse_categorical_crossentropy(label_batch2, y_pred)

        return tf.math.reduce_mean(
            tf.math.multiply(loss1, (1 - box_weights1))
            + tf.math.multiply(loss2, box_weights2),
            axis=0,
        )

    return tf.math.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(label_batch, y_pred)
    )


def snapmix_batch_augmentation(
    class_activation_model, model, img_batch, label_batch, output_layer_name, alpha=0.2
):
    """
    Applies, the SnapMix-augmentation to the images and labels within a data batch with respect to a model.

    Args:
        class_activation_model (model) :
        model (model) :
        img_batch (tf.tensor) : batch with images, all the same shape
        label_batch (numpy list) : batch with labels for the images
        output_layer_name (string) : name of the final output-layer
        alpha (float), optional: parameter for beta-distribution generating image shrinking-factor for box-area

    Returns:
        augmented_images : the augmented input-images
        label_batch2 : the labels of the images that have been patched into the input-images
        box_weights1 : batch of semantic weights of cut-out-boxes
        box_weights2 : batch of semantic weights of patched-in-boxes
    """

    batch_size = img_batch.shape[0]
    img_width = img_batch.shape[1]
    img_height = img_batch.shape[2]

    # get classificator weights:
    classificator_weights = model.get_layer(
        output_layer_name
    ).get_weights()  # returns: (weights, biases)
    classificator_weights = classificator_weights[0]

    box1 = random_box(img_width, img_height, alpha=alpha)
    box2 = random_box(img_width, img_height, alpha=alpha)

    # build another image batch from the input batch:
    rng = np.random.default_rng()
    permutation = rng.permutation(batch_size)
    label_batch = label_batch.numpy().astype(int)

    img_batch2 = np.copy(img_batch)
    img_batch2 = img_batch2[permutation]
    label_batch2 = np.copy(label_batch)
    label_batch2 = label_batch2[permutation]

    # get spm and calculate boxweights:
    SPM1 = batch_semantic_percentage_map(
        class_activation_model=class_activation_model,
        classificator_weights=classificator_weights,
        img_batch=img_batch,
        label_batch=label_batch,
    )

    SPM2 = np.copy(SPM1)
    SPM2 = SPM2[permutation, :, :]
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    cropped_SPM1 = SPM1[:, x11 : (x12 + 1), y11 : (y12 + 1)]
    # box_weights1 = tf.reduce_sum(cropped_SPM1, axis=[1, 2]).numpy()
    box_weights1 = np.sum(cropped_SPM1, axis=(1, 2))
    cropped_SPM2 = SPM2[:, x21 : (x22 + 1), y21 : (y22 + 1)]
    # box_weights2 = tf.reduce_sum(cropped_SPM2, axis=[1, 2]).numpy()
    box_weights2 = np.sum(cropped_SPM2, axis=(1, 2))

    # some normalization for patching with equal labels:
    same_label = label_batch == label_batch2
    tmp = np.copy(box_weights1)
    box_weights1[same_label] += box_weights2[same_label]
    box_weights2[same_label] += tmp[same_label]

    # fix for cases where box_weights are not well defined:
    rel_area1 = (y12 - y11) * (x12 - x11) / (img_width * img_height)
    rel_area2 = (y22 - y21) * (x22 - x21) / (img_width * img_height)
    box_weights1[np.isnan(box_weights1)] = rel_area1
    box_weights2[np.isnan(box_weights2)] = rel_area2

    # crop and paste images:
    # cropped = img_batch2[:, x21: x22, y21: y22]
    cropped = img_batch2[:, x21:x22, y21:y22, :]
    resized_cropped = np.zeros(
        (cropped.shape[0], x12 - x11, y12 - y11, cropped.shape[3])
    )
    # print("cropped.shape: {}".format(cropped.shape))
    # print("resized_cropped.shape: {}".format(resized_cropped.shape))
    for i in range(batch_size):
        resized_cropped[i] = cv2.resize(
            cropped[i, :, :], (y12 - y11, x12 - x11), interpolation=cv2.INTER_CUBIC
        )
    # cropped = tf.image.resize(cropped, (x12 - x11, y12 - y11)).numpy()
    # copy images otherwise originals are spoiled:
    augmented_images = np.copy(img_batch)
    augmented_images[:, x11:x12, y11:y12] = resized_cropped

    return augmented_images, label_batch2, box_weights1, box_weights2


def batch_semantic_percentage_map(
    class_activation_model, classificator_weights, img_batch, label_batch
):
    """
    Calculates the SPM - Semantic Percentage Map of a batch of images.

    Args:
        class_activation_model : the part of the model to calculate the class-activations from (the part before the classifier)
        classificator_weights : the weights of the last layer of the classifier, i.e. for a softmax-layer:
            classificator_weights = model.get_layer("SoftMaxLayerName").get_weights()

    Returns:
        the SPMs (Semantic Percentage Maps) for a batch of images.
    """
    feature_maps_batch = class_activation_model.predict(img_batch)

    # Calculate Class Activation Map (CAM):
    batch_size = feature_maps_batch.shape[0]
    feature_map_width = feature_maps_batch.shape[1]
    feature_map_height = feature_maps_batch.shape[2]
    CAM_batch = np.zeros((batch_size, feature_map_width, feature_map_height))
    clw_matrix = classificator_weights[:, label_batch]
    for i in range(batch_size):
        # CAM_batch[i, :, :] = tf.tensordot(clw_matrix[:, i], feature_maps_batch[i, :, :, :], axes=[[0], [2]])
        CAM_batch[i, :, :] = np.tensordot(
            clw_matrix[:, i], feature_maps_batch[i, :, :, :], axes=([0], [2])
        )

    # upsampling feature map to size of image:
    image_width = img_batch.shape[1]
    image_height = img_batch.shape[2]
    resized_CAM_batch = np.zeros((batch_size, image_width, image_height))
    for i in range(batch_size):
        resized_CAM_batch[i, :, :] = cv2.resize(
            CAM_batch[i, :, :],
            (image_width, image_height),
            interpolation=cv2.INTER_CUBIC,
        )

    # CAM_batch = np.expand_dims(CAM_batch, axis=-1)
    # CAM_batch = tf.image.resize(images=CAM_batch, size=(image_width, image_height), method="bilinear")
    # CAM_batch = np.squeeze(CAM_batch, axis=-1)

    # CAM_batch -= tf.math.reduce_min(CAM_batch)
    resized_CAM_batch -= np.amin(resized_CAM_batch)
    # normalization_factor = tf.reduce_sum(CAM_batch).numpy() + 1e-8
    normalization_factor = np.sum(resized_CAM_batch) + 1e-8
    resized_CAM_batch /= normalization_factor

    return resized_CAM_batch


def random_box(im_width, im_height, alpha, minimal_width=2, minimal_height=2):
    """
    Returns a random box=(x1, y1, x2, y2) with 0 < x1, x2 < im_width
    and 0< y1, y2, < im_height that spans an area equal to
    lambda_img * (x2 - x1) * (y2 - y1), where lambda_img is randomly drawn from a beta-distribution
    beta(alpha, alpha)
    """
    rng = np.random.default_rng()
    random_width = im_width + 1
    random_height = 0

    while (
        random_width > im_width
        or random_height > im_height
        or random_height < minimal_height
        or random_width < minimal_width
    ):
        lambda_img = rng.beta(alpha, alpha)
        if lambda_img < 1 and lambda_img > 0:
            random_width = int(
                rng.integers(minimal_width, im_width) * np.sqrt(lambda_img) // 1
            )
            # random_width = random_width.astype(int)

            random_height = int(
                rng.integers(minimal_height, im_height) * np.sqrt(lambda_img) // 1
            )
            # random_height = random_height.astype(int)

    left_upper_x = rng.integers(0, im_width - random_width, endpoint=True)
    left_upper_y = rng.integers(0, im_height - random_height, endpoint=True)

    box = (
        left_upper_x,
        left_upper_y,
        left_upper_x + random_width - 1,
        left_upper_y + random_height - 1,
    )

    return box