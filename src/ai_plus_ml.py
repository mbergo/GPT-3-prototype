import tensorflow as tf
from tensorflow.keras import layers

# Input layer
inputs = layers.Input(shape=(None,))

# Embedding layer
embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

# Convolutional layer
conv = layers.Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(embedding)

# Max-pooling layer
pool = layers.GlobalMaxPooling1D()(conv)

# Attention layer
attention = layers.Dense(1, activation='tanh')(pool)
attention = layers.Flatten()(attention)
attention = layers.Activation('softmax')(attention)
attention = layers.RepeatVector(num_filters)(attention)
attention = layers.Permute([2, 1])(attention)

# Multiply the attention weights with the input feature map
sent_representation = layers.multiply([conv, attention])
sent_representation = layers.Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(num_filters,))(sent_representation)

# Supervised Learning
outputs = layers.Dense(num_classes, activation='softmax')(sent_representation)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Unsupervised Learning
encoder = tf.keras.Model(inputs=inputs, outputs=sent_representation)
encoder.compile(optimizer='adam', loss='mse')

# Transfer Learning
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False
x = base_model.output
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

