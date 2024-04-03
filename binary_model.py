
class CNNHyperModel(kt.HyperModel):
    def __init__(self, params):
        self.input_shape = params['input_shape']
        self.n_classes = params['n_classes']

    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)  # single beat, 8 lead (1, 300, 8)
        x = inputs
        for i in range(6):
            filters = hp.Int('filters_' + str(i), 32, 128, step=32)
            kernels = hp.Int('kernel_' + str(i), 3, 12, step=1)
            dropouts = hp.Float('dropout_'+str(i), 0, 0.5, step=0.1, default=0.5)
            for _ in range(2):
                x = tf.keras.layers.Convolution2D(
                  filters, kernel_size=(1, kernels), padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation(activation='elu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(1,2),
                    strides=2,
                    padding='valid')(x)
            x = tf.keras.layers.Dropout(dropouts)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            hp.Int('hidden_size', 20, 500, step=30, default=50),
            activation='elu')(x)
        x = tf.keras.layers.Dropout(
                hp.Float('dropout_dense', 0, 0.5, step=0.1, default=0.5))(x)
        outputs = tf.keras.layers.Dense(self.n_classes, activation='sigmoid')(x)

        # binary
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('learning_rate', 5e-5, 2e-3, sampling='log')),
                loss="binary_crossentropy",
                metrics=[
                    tf.keras.metrics.AUC(name='auc', multi_label=False),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                    ])
        return model

def run_model(params, train_x, train_y, val_x, val_y):
    params['train_steps'] = train_x.shape[0] // (params['batch_size']*params['epochs'])
    params['val_steps'] = val_x.shape[0] // (params['batch_size']*params['epochs'])
    params['n_classes'] = len(params['class_labels'])

    print("Training over all hyperparams...")

    hypermodel = CNNHyperModel(params)

    tuner = kt.Hyperband(
        hypermodel,
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=params['epochs'],
        hyperband_iterations=1,
        seed=10,
        directory='checkpoints',
        project_name=params['project_name'])

    tuner.search(train_x,
                 train_y,
                 steps_per_epoch=params['train_steps'],
                 epochs=params['epochs'],
                 validation_data=(val_x, val_y),
                 validation_steps=params['val_steps'],
                 verbose = 2,
                 callbacks=[tf.keras.callbacks.EarlyStopping('val_auc', patience=6, mode='max')])
    tuner.results_summary()
    best_model = tuner.get_best_models(1)[0]

    # Saving all the information from the best run
    filename_best_model = 'models/' + params['project_name'] + '_best' + '.h5'
    best_model.save(filename_best_model)

    return best_model, params, tuner


"""### Run model"""

params = {'input_shape': (1, 2500, 8),
        'batch_size': 8,
        'class_labels': CLASSES,
        'project_name': 'inary_one_echo_8_lead_raw',
        'epochs': 75,
        'max_trials': 75}

best_model, params, tuner = run_model(params, train_x, train_y_all, val_x, val_y_all)

tuner.results_summary()
best_model.summary()