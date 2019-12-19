# -*- coding: utf-8 -*-


from image_processing import *


"""1) Loading the set of images"""

traindir = DATA_ROOT_DIR + "training/"
imagedir = traindir + "images/"

# list with all the available images name
files = os.listdir(imagedir) 

n = NUMBER_TRAIN_IMG
imgs, gt_imgs = LoadTrainingData(n, traindir)

"""2) Data augmentation"""

# Augmenting the data
angles = [0, 30, 45, 60, 90, 120, 135, 150, 180]
aug_imgs_small, aug_gt_imgs_small = DataAugmentation(imgs, gt_imgs, angles, sym=True, printinfo=True)

X_tr = np.asarray([BuildExtendedImage(aug_imgs_small[i], PAD) for i in range(len(aug_imgs_small))])
Y_tr_raw = np.asarray([BuildExtendedImage(aug_gt_imgs_small[i], PAD) for i in range(len(aug_gt_imgs_small))])

del imgs, gt_imgs, aug_imgs_small, aug_gt_imgs_small
print("(Augmented) number of images (and groundtruth): ", X_tr.shape[0])

plt.imshow(X_tr[100], cmap='Greys_r')

"""3) CNN model definition"""

# processing the training data
X_tr = X_tr.astype('float32')
Y_tr = np.expand_dims(Y_tr_raw, axis=3).astype('float32')

print(" Shape of X_tr: ", X_tr.shape)
print(" Shape of Y_tr: ", Y_tr.shape)

# Build U-Net model

K.clear_session()


inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, IMG_CHANNELS))

s = tf.keras.layers.Lambda(lambda x: x )(inputs)
 
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
 
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c5)
 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])

c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])

c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])

c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)

c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint(CHECKPOINT_DIR + 'UNET_V2.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)

"""### 4) CNN model training and testing"""

history = model.fit(X_tr, Y_tr, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, validation_split = 0.20, callbacks=[checkpoint])

PlotHistory(history)

CreateSubmissionUNET(model)

VisualizeUNETPrediction(model, 18)

