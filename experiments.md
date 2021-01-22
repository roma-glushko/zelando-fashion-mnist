# Experiment Notes

## MLP

### Chance Model

- Loss: 2.43648624420166
- Accuracy: 0.12780000269412994

### Dense(300, activation='relu'), Dense(100, activation='relu'), SGD

- Best Val Loss: 0.3154
- Best Val Accuracy: 0.8987
- Best Test Loss: 0.3189171850681305
- Best Test Accuracy: 0.8956999778747559

## Dense(300, activation='relu'), Dense(100, activation='relu'), Adam(lr=0.0005)

- Best Val Loss: 0.3054
- Best Val Accuracy: 0.8972
- Best Test Loss: 0.31720462441444397
- Best Test Accuracy: 0.8913999795913696

## Dense(300, activation='relu'), BatchNorm(), Dense(100, activation='relu'), BatchNorm(), Adam(lr=0.0005)

- Best Val Loss: 0.3020
- Best Val Accuracy: 0.8958
- Best Test Loss: 0.33810487389564514
- Best Test Accuracy: 0.8925999999046326

## Adam(lr=0.0005), Dense(300, activation='relu'), BatchNorm(), Dropout(0.2), Dense(100, activation='relu'), BatchNorm(), Dropout(0.2)

- Best Val Loss: 0.2882
- Best Val Accuracy: 0.8964
- Best Test Loss: 0.3012319505214691
- Best Test Accuracy: 0.8920000195503235

## Adam(lr=0.0005), batch_size=32, Dense(300, activation='relu'), BatchNorm(), Dropout(0.2), Dense(100, activation='relu'), BatchNorm(), Dropout(0.2), Dense(50, activation='relu'), BatchNorm(), Dropout(0.2)

- Best Val Loss: 0.2822
- Best Val Accuracy: 0.9010
- Best Test Loss: 0.29744410514831543
- Best Test Accuracy: 0.8920999765396118

## Adam(lr=0.0005), batch_size=64, Dense(300, activation='relu'), BatchNorm(), Dropout(0.2), Dense(100, activation='relu'), BatchNorm(), Dropout(0.2), Dense(50, activation='relu'), BatchNorm(), Dropout(0.2)

- Best Val Loss: 0.3215
- Best Val Accuracy: 0.9032
- Best Test Loss: 0.29807841777801514
- Best Test Accuracy: 0.8945000171661377

## RMSprop(), batch_size=32, Dense(300, activation='relu'), BatchNorm(), Dropout(0.2), Dense(100, activation='relu'), BatchNorm(), Dropout(0.2), Dense(50, activation='relu'), BatchNorm(), Dropout(0.2)

- Best Val Loss: 0.3024
- Best Val Accuracy: 0.8970
- Best Test Loss: 0.3157300651073456
- Best Test Accuracy: 0.8924000263214111

## RMSprop(), batch_size=256, Dense(300, activation='relu'), BatchNorm(), Dropout(0.2), Dense(100, activation='relu'), BatchNorm(), Dropout(0.2), Dense(50, activation='relu'), BatchNorm(), Dropout(0.2)

- Best Val Loss: 0.2886
- Best Val Accuracy: 0.9017
- Best Test Loss: 0.2969489097595215
- Best Test Accuracy: 0.8981000185012817

## Adam(lr=0.0005), batch_size=256, Dense(300, activation='relu'), BatchNorm(), Dropout(0.2), Dense(100, activation='relu'), BatchNorm(), Dropout(0.2), Dense(50, activation='relu'), BatchNorm(), Dropout(0.2)

- Best Val Loss: 0.2938
- Best Val Accuracy: 0.9023
- Best Test Loss: 0.29765433073043823
- Best Test Accuracy: 0.8996000289916992

## Nadam(lr=0.0005), batch_size=256, Dense(300, activation='selu'), AlphaDropout(0.2), Dense(100, activation='selu'), AlphaDropout(0.2), Dense(50, activation='selu'), AlphaDropout(0.2)

- Best Val Loss: 0.3629
- Best Val Accuracy: 0.9026
- Best Test Loss: 0.39967575669288635
- Best Test Accuracy: 0.8942999839782715

## Nadam(lr=0.001), batch_size=256, Dense(300, activation='selu', kernel_initializer='lecun_normal'), AlphaDropout(0.2), Dense(100, activation='selu', kernel_initializer='lecun_normal'), AlphaDropout(0.2), Dense(50, activation='selu', kernel_initializer='lecun_normal'), AlphaDropout(0.2)

- Best Val Loss: 0.4443
- Best Val Accuracy: 0.9081
- Best Test Loss: 0.5121541023254395
- Best Test Accuracy: 0.8996999859809875

## CNN

### Baseline CNN

- Val Loss: 0.2382
- Test Loss: 0.2433 
- Val Accuracy: 0.9224
- Test Accuracy: 0.9136

```python
cnn = Conv2D(64, 7, activation='relu', name='conv1_low_level')(cnn_input_layer)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(128, 3, activation='relu', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, activation='relu', name='conv2_2_double')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.2, name='dropout1')(cnn)
cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.2, name='dropout2')(cnn)

Nadam(lr=0.001)
batch_size=256
```

### 3 Conv Layer NN

- Val Loss: 0.2181
- Test Loss: 0.2269
- Val Accuracy: 0.9247
- Test Accuracy: 0.9174

```python
cnn_input_layer = Input((28, 28, 1), name='input')

cnn = Conv2D(64, 7, activation='relu', padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = MaxPool2D(2, name='pooling1')(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.2, name='dropout1')(cnn)

cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.2, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

Nadam(lr=0.001)
batch_size=256
```

### CNN: 3Conv + Droupout (0.4) + 512 batch_size

- Val Loss: 0.2172
- Test Loss: 0.21914228796958923
- Val Accuracy: 0.9273
- Test Accuracy: 0.9253000020980835

```python
cnn_input_layer = Input((28, 28, 1), name='input')

cnn = Conv2D(64, 7, activation='relu', padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = MaxPool2D(2, name='pooling1')(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.4, name='dropout1')(cnn)

cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.4, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=512
```

### CNN: 3Conv + Droupout (0.4) + 256 batch_size

- Val Loss: 0.2183
- Test Loss: 0.22551453113555908
- Val Accuracy: 0.9283
- Test Accuracy: 0.9208999872207642

```python
cnn_input_layer = Input((28, 28, 1), name='input')

cnn = Conv2D(64, 7, activation='relu', padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = MaxPool2D(2, name='pooling1')(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.4, name='dropout1')(cnn)

cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.4, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN: 3Conv + Droupout (0.5) + 256 batch_size

- Val Loss: 0.2310
- Test Loss: 0.22115148603916168
- Val Accuracy: 0.9293
- Test Accuracy: 0.9280999898910522

```python
cnn_input_layer = Input((28, 28, 1), name='input')

cnn = Conv2D(64, 7, activation='relu', padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = MaxPool2D(2, name='pooling1')(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN: 3Conv + LNR1 + Droupout (0.4) + 256 batch_size

- Val Loss: 0.2205
- Test Loss: 0.21933405101299286
- Val Accuracy: 0.9304
- Test Accuracy: 0.92330002784729

```python
cnn_input_layer = Input((28, 28, 1), name='input')

cnn = Conv2D(64, 7, activation='relu', padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = Lambda(lambda X: local_response_normalization(X), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.4, name='dropout1')(cnn)

cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.4, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN: 3Conv + LNR1(r=5) + Droupout (0.5) + 256 batch_size

- Val Loss: 0.2192
- Test Loss: 0.2200920581817627
- Val Accuracy: 0.9271
- Test Accuracy: 0.9259999990463257

```python
cnn_input_layer = Input((28, 28, 1), name='input')

cnn = Conv2D(64, 7, activation='relu', padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = Lambda(lambda X: local_response_normalization(X), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN: 3Conv + LNR1(r=2) + Droupout (0.5) + 256 batch_size

- Val Loss: 0.2276
- Test Loss: 0.23375444114208221
- Val Accuracy: 0.9292
- Test Accuracy: 0.9180999994277954

```python
cnn_input_layer = Input((28, 28, 1), name='input')

cnn = Conv2D(64, 7, activation='relu', padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=2), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN: 3Conv + LNR1(r=3) + Droupout (0.5) + 256 batch_size

- Val Loss: 0.2297
- Test Loss: 0.2283312976360321
- Val Accuracy: 0.9294
- Test Accuracy: 0.919700026512146

```python
cnn_input_layer = Input((28, 28, 1), name='input')

cnn = Conv2D(64, 7, activation='relu', padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN: 3Conv + LNR2(r=3) + Droupout (0.5) + 256 batch_size

- Val Loss: 0.2246
- Test Loss: 0.22063612937927246
- Val Accuracy: 0.9276
- Test Accuracy: 0.9229999780654907

```python
cnn_input_layer = Input((28, 28, 1), name='input')

cnn = Conv2D(64, 7, activation='relu', padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2_double')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn2')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN TBU

Best Val Loss: 0.2212
Best Val Accuracy: 0.9308
[0.22356010973453522, 0.92330002784729]

```python
cnn_input_layer = Input((28, 28, 1), name='input')

cnn = Conv2D(128, 7, activation='relu', padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv2_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2_double')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN: 3Conv with BatchNorm and LNR1 + 2 Dense Layers with Droupout (0.5); 256 batch_size

- Val Loss: 0.2360
- Test Loss: 0.22700156271457672
- Val Accuracy: 0.9298
- Test Accuracy: 0.9272000193595886

```python
cnn_input_layer = Input((28, 28, 1), name='input')

# layer 1
cnn = Conv2D(64, 7, padding='same', name='conv1_low_level')(cnn_input_layer)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn1')(cnn)

cnn = MaxPool2D(2, name='pooling1')(cnn)

# layer 2
cnn = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, padding='same', name='conv2_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)

cnn = MaxPool2D(2, name='pooling2')(cnn)

# layer 3
cnn = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, padding='same', name='conv3_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)

cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

# layer 4
cnn = Dense(128, activation='relu', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

# layer 5
cnn = Dense(64, activation='relu', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN: 3Conv with BatchNorm and LNR1 + 2 Dense Layers with Droupout (0.5); 256 batch_size (enabled ReLu activations for all layers with He inits)

- Test Loss: 0.23280414938926697
- Test Accur: 0.9211999773979187
- Val Loss: 0.2409
- Val Accuracy: 0.9253

```
cnn_input_layer = Input((28, 28, 1), name='input')

# layer 1
cnn = Conv2D(64, 7, padding='same', activation='relu', kernel_initializer='he_normal', name='conv1_low_level')(cnn_input_layer)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

# layer 2
cnn = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

# layer 3
cnn = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal', name='conv3_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

# layer 4
cnn = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

# layer 5
cnn = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN: ReduceLROnPlateau

Best Val Loss: 0.2286
Best Val Accuracy: 0.9320
[0.22308972477912903, 0.925599992275238]

```
cnn_input_layer = Input((28, 28, 1), name='input')

# layer 1
cnn = Conv2D(64, 7, padding='same', kernel_initializer='he_normal', name='conv1_low_level')(cnn_input_layer)
cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

# layer 2
cnn = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='conv2_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

# layer 3
cnn = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='conv3_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

# layer 4
cnn = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

# layer 5
cnn = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    verbose=1, 
    factor=0.5, 
    min_lr=5e-5
)
```

### CNN: CNN: 3Conv with BatchNorm and LNR1 + 2 Dense Layers with Droupout (0.5); 256 batch_size (dense layers with He inits)

Best Val Loss: 0.2304
Best Val Accuracy: 0.9275
[0.22775952517986298, 0.9211999773979187]

```
cnn_input_layer = Input((28, 28, 1), name='input')

# layer 1
cnn = Conv2D(64, 7, padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

# layer 2
cnn = Conv2D(128, 3, padding='same', activation='relu', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, padding='same', name='conv2_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

# layer 3
cnn = Conv2D(256, 3, padding='same', activation='relu', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, padding='same', name='conv3_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

# layer 4
cnn = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

# layer 5
cnn = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
```

### CNN: 3Conv with BatchNorm and LNR1 + 2 Dense Layers with Droupout (0.5); 256 batch_size (dense layers with He inits)

Best Val Loss: 0.2061
Best Val Accuracy: 0.9395
[0.22646933794021606, 0.9354000091552734]

```
cnn_input_layer = Input((28, 28, 1), name='input')

# layer 1
cnn = Conv2D(64, 7, padding='same', name='conv1_low_level')(cnn_input_layer)
cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

# layer 2
cnn = Conv2D(128, 3, padding='same', activation='relu', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, padding='same', name='conv2_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

# layer 3
cnn = Conv2D(256, 3, padding='same', activation='relu', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, padding='same', name='conv3_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

# layer 4
cnn = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

# layer 5
cnn = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    verbose=1, 
    factor=0.5, 
    min_lr=5e-5
)
```

### CNN: 3Conv with BatchNorm and 3LNR + 2 Dense Layers with Droupout (0.5); 256 batch_size (dense layers with He inits)

Best Val Loss: 0.2052
Best Val Accuracy: 0.9401
[0.20599116384983063, 0.9307000041007996]

```
cnn_input_layer = Input((28, 28, 1), name='input')

# layer 1
cnn = Conv2D(64, 7, padding='same', name='conv1_low_level')(cnn_input_layer)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

# layer 2
cnn = Conv2D(128, 3, padding='same', activation='relu', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, padding='same', name='conv2_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn2')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

# layer 3
cnn = Conv2D(256, 3, padding='same', activation='relu', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, padding='same', name='conv3_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn3')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

# layer 4
cnn = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

# layer 5
cnn = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-5,
    verbose=1, 
)
```

### CNN: 3Conv with BatchNorm and 3LNR + 2 Dense Layers with Droupout (0.5); 256 batch_size (dense layers with He inits); Data Augmentation

Best Val Loss: 0.2165
Best Val Accuracy: 0.9316
[0.2099497765302658, 0.9305999875068665]

```
cnn_input_layer = Input((28, 28, 1), name='input')

# layer 1
cnn = Conv2D(64, 7, padding='same', name='conv1_low_level')(cnn_input_layer)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn1')(cnn)
cnn = MaxPool2D(2, name='pooling1')(cnn)

# layer 2
cnn = Conv2D(128, 3, padding='same', activation='relu', name='conv2_1_double')(cnn)
cnn = Conv2D(128, 3, padding='same', name='conv2_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn2')(cnn)
cnn = MaxPool2D(2, name='pooling2')(cnn)

# layer 3
cnn = Conv2D(256, 3, padding='same', activation='relu', name='conv3_1_double')(cnn)
cnn = Conv2D(256, 3, padding='same', name='conv3_2_double')(cnn)

cnn = BatchNormalization()(cnn)
cnn = Activation('relu')(cnn)
cnn = Lambda(lambda X: local_response_normalization(X, depth_radius=3), name='lrn3')(cnn)
cnn = MaxPool2D(2, name='pooling3')(cnn)

cnn = Flatten()(cnn)

# layer 4
cnn = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1')(cnn)
cnn = Dropout(0.5, name='dropout1')(cnn)

# layer 5
cnn = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense2')(cnn)
cnn = Dropout(0.5, name='dropout2')(cnn)

cnn_output_layer = Dense(label_num, activation='softmax', name='dense_output')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

Nadam(lr=0.001)
batch_size=256
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-5,
    verbose=1, 
)
data_augmentator = ImageDataGenerator(
    rotation_range = 8,
    vertical_flip=True
)
```