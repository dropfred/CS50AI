# HarvardX CS50AI - CS50's Introduction to Artificial Intelligence with Python

## Week 5 - Traffic project

[Screencast](https://www.youtube.com/watch?v=fQ5VKtALqMM)

### ```load_data```

I wrote a separate ```load_image``` function I could resuse in tests. If necessary, loaded images are cropped before being scaled in order to avoid stretching.

I added a optional ```verbose``` parameter to the ```load_data``` function that somewhat mimic the TensorFlow epoch fitting output to see the loading progress.

Also, I added the ```IMG_CHANNELS``` global variable to handle grayscale images.

### ```get_model```

I tried dozens of models, varying the numbers of convolution, pooling, dropout, and dense layers, and some of their parameters (number of filters, kernel size, rate, number of nodes, etc.).

Also, I noticed that adding some normalization layers often gives better results, although it seems to increase fitting time quite significantly.

In order to mesure the models and pick the best one, I wrote a little python program to store the obtained results into a CSV file, along with a log file:

```python
models = {
    "c32/3|mp2|f|d128|do0.5": [
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ],
    "c32/3|mp2|f|d512|do0.5": [
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ],
    # many more models
    # :
}

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w') as f: f.write('model,time,loss,accuracy\n')

for (n, m) in models.items():
    for _ in range(SAMPLES):
        model = tf.keras.models.Sequential(m)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        print(f"#\n# model={n}\n#")
        model.summary()
        # print(model.to_json())
        t0 = time()
        model.fit(x_train, y_train, epochs=EPOCHS, verbose=2)
        t1 = time()
        e = model.evaluate(x_test,  y_test, verbose=2)
        with open(CSV_FILE, 'a') as f: f.write(f"{n},{t1 - t0:0.1f},{e[0]:0.4f},{e[1]:0.4f}\n")
```

The obtained CSV file, where each model is tested three times:

```
model,time,loss,accuracy
c32/3|mp2|f|d128|do0.5,34.8,3.5029,0.0550
c32/3|mp2|f|d128|do0.5,33.5,3.5031,0.0550
c32/3|mp2|f|d128|do0.5,33.9,3.5031,0.0550
c32/3|mp2|f|d2048|do0.5,272.1,0.3959,0.9364
c32/3|mp2|f|d2048|do0.5,278.5,0.5934,0.9475
c32/3|mp2|f|d2048|do0.5,272.9,0.8839,0.9505
c32/3|mp2|f|do0.5|d128|d128,37.8,0.2744,0.9323
c32/3|mp2|f|do0.5|d128|d128,40.1,0.2604,0.9516
c32/3|mp2|f|do0.5|d128|d128,38.1,0.3038,0.9596
:
```

I endend-up with this model which gives the best results for a reasonable learning time (about 10s per epoch on my computer):

```python
model = Sequential([
    Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
    Conv2D(16, (3, 3), activation="relu"),
    Conv2D(16, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Conv2D(16, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CATEGORIES, activation="softmax")
])
```

The model in action:

```
> python traffic.py gtsrb
Epoch 1/10
500/500 [==============================] - 10s 19ms/step - loss: 2.5396 - accuracy: 0.3530
Epoch 2/10
500/500 [==============================] - 10s 19ms/step - loss: 0.5729 - accuracy: 0.8286
Epoch 3/10
500/500 [==============================] - 10s 19ms/step - loss: 0.2881 - accuracy: 0.9105
Epoch 4/10
500/500 [==============================] - 10s 20ms/step - loss: 0.1982 - accuracy: 0.9396
Epoch 5/10
500/500 [==============================] - 10s 19ms/step - loss: 0.1350 - accuracy: 0.9584
Epoch 6/10
500/500 [==============================] - 10s 19ms/step - loss: 0.1246 - accuracy: 0.9629
Epoch 7/10
500/500 [==============================] - 10s 20ms/step - loss: 0.1051 - accuracy: 0.9685
Epoch 8/10
500/500 [==============================] - 10s 19ms/step - loss: 0.0868 - accuracy: 0.9738
Epoch 9/10
500/500 [==============================] - 10s 19ms/step - loss: 0.0767 - accuracy: 0.9765
Epoch 10/10
500/500 [==============================] - 9s 19ms/step - loss: 0.0644 - accuracy: 0.9793
333/333 - 2s - loss: 0.0466 - accuracy: 0.9884
```

### Remarks

Sometimes, the tested model had good results when applied to the gtsrb set, but quite poor results on other sources. The model I ended-up with gives good results for all the images I have tested.
