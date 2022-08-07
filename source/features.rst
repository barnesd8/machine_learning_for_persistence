Features
*******************

.. automodule:: ml4pers.features.create_features
    :members:

Below is an example of creating features from persistence diagrams using the kernel method::

    from ml4pers.features import kernel_features

    X_train_features, X_test_features = kernel_features(X_train, X_test)