import tensorflow as tf
import numpy as np
import h5py

input_z = tf.keras.Input(shape=(140, 140, 3))
z = tf.keras.applications.mobilenet.MobileNet(include_top=False, input_shape=(140, 140, 3), weights=None)(input_z)
z = tf.keras.layers.AveragePooling2D((4, 4), strides=1)(z)
branch_z = tf.keras.Model(inputs=input_z, outputs=z)


def tile(embed1):
    embed = tf.keras.backend.tile(embed1, [1, 5, 5, 1])
    return embed


inputs_x = tf.keras.Input(shape=(256, 256, 3))
inputs_z_ = tf.keras.Input(shape=(1, 1, 1024))
z_ = tf.keras.layers.Lambda(tile)(inputs_z_)

x = tf.keras.applications.mobilenet.MobileNet(include_top=False, input_shape=(256, 256, 3), weights=None)(inputs_x)
x = tf.keras.layers.AveragePooling2D((4, 4), strides=1)(x)
x = tf.keras.layers.Multiply()([x, z_])
x = tf.keras.layers.GlobalAveragePooling2D()(x)

z_in = tf.keras.layers.Flatten()(inputs_z_)
x = tf.keras.layers.Concatenate()([x, z_in])
x = tf.keras.layers.Dropout(0.5)(x)
pred = tf.keras.layers.Dense(1, activation='sigmoid')(x)

branch_search = tf.keras.Model(inputs=[inputs_z_, inputs_x], outputs=pred)


inputs_1 = tf.keras.Input(shape=(140, 140, 3))
inputs_2 = tf.keras.Input(shape=(256, 256, 3))

output = branch_search([branch_z(inputs_1), inputs_2])

model = tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=output)


fdata = h5py.File('/home/masterbin-iiau/SPLT/Siam/Skim_data.h5','r')
search = fdata['search']
template = fdata['template']
labels = fdata['label']


model.compile(optimizer=tf.keras.optimizers.Adam(0.001, decay=1e-2),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([template, search], labels, epochs=20, batch_size=32, validation_split=0.1)

branch_search.save('./branch_search.h5')
branch_z.save('./branch_z.h5')

