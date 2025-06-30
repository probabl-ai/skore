"""
.. _example_resnet:

============================================
Deep learning with skore: Resnet on ImageNet
============================================

This example shows how to apply skore for deep learning, with Resnet on ImageNet.
This is a multi-class classification task.

TODO: add the following to ``skore/pyproject.toml`` in ``sphinx``:

- "keras_hub"
- "tensorflow"
- "tensorflow_datasets"
"""

# %%
# First of all, we ensure compatibility and avoids common issues with
# parallelism and backend conflicts in Keras/TensorFlow workflows:

# %%
IS_EXECUTE = False

# %%
if IS_EXECUTE:
    import os

    os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["KERAS_BACKEND"] = "torch"

# %%
# Let us load a pretrained ResNet-50 model with ImageNet weights:

# %%
if IS_EXECUTE:
    import keras_hub

    classifier = keras_hub.models.ImageClassifier.from_preset(
        "resnet_50_imagenet",
        activation="softmax",
    )

# %%
if IS_EXECUTE:
    classifier.summary()

# %%
if IS_EXECUTE:
    import tensorflow as tf
    import tensorflow_datasets as tfds

    dataset, info = tfds.load(
        "imagenet_v2", with_info=True, split="test", as_supervised=True
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label

    dataset = dataset.take(1_000).map(preprocess)
    images, labels = zip(*dataset)
    images = tf.stack(images)
    labels = tf.stack(labels)

# %%
if IS_EXECUTE:
    from skore import EstimatorReport

    reporter = EstimatorReport(classifier, fit=False)

# %%
if IS_EXECUTE:
    reporter.help()

# %%
if IS_EXECUTE:
    import numpy as np
    from sklearn.metrics import top_k_accuracy_score

    reporter.metrics.custom_metric(
        metric_function=top_k_accuracy_score,
        response_method="predict",
        data_source="X_y",
        X=images,
        y=labels,
        k=5,
        metric_name="Top-5 Accuracy",
        labels=np.arange(1_000),
    )

# %%
if IS_EXECUTE:
    from sklearn.metrics import make_scorer

    def top_1_accuracy(y_true, y_pred, labels):
        return top_k_accuracy_score(y_true, y_pred, k=1, labels=labels)

    top_1_accuracy_scorer = make_scorer(top_1_accuracy, labels=np.arange(1_000))

    def top_5_accuracy(y_true, y_pred, labels):
        return top_k_accuracy_score(y_true, y_pred, k=5, labels=labels)

    top_5_accuracy_scorer = make_scorer(top_5_accuracy, labels=np.arange(1_000))

# %%
if IS_EXECUTE:
    reporter.metrics.report_metrics(
        scoring=[top_1_accuracy_scorer, top_5_accuracy_scorer],
        scoring_name=["Top-1 Accuracy", "Top-5 Accuracy"],
        data_source="X_y",
        X=images,
        y=labels,
    )


# %%
if IS_EXECUTE:

    def top_10_accuracy(y_true, y_pred, labels):
        return top_k_accuracy_score(y_true, y_pred, k=10, labels=labels)

    top_10_accuracy_scorer = make_scorer(top_10_accuracy, labels=np.arange(1_000))

# %%
if IS_EXECUTE:
    reporter.metrics.report_metrics(
        scoring=[top_1_accuracy_scorer, top_5_accuracy_scorer, top_10_accuracy_scorer],
        scoring_name=["Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy"],
        data_source="X_y",
        X=images,
        y=labels,
    )

# %%
if IS_EXECUTE:
    reporter.clean_cache()

# %%
if IS_EXECUTE:
    reporter.metrics.report_metrics(
        scoring=[top_1_accuracy_scorer, top_5_accuracy_scorer, top_10_accuracy_scorer],
        scoring_name=["Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy"],
        data_source="X_y",
        X=images,
        y=labels,
    )
