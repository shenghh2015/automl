# EfficientDet-based image segmentation
# Author: shenghh
# Date: Aug 03, 2020
# ==============================================================================
"""A demo script to show to train a segmentation model."""

import os
from efficientdet_keras import EfficientDetNet
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model = EfficientDetNet('efficientdet-d0')
model.build((1, 512, 512, 3))

# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy'])
# 
# EPOCHS = 20
# VAL_SUBSPLITS = 5
# VALIDATION_STEPS = info.splits[
#     'test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS
# 
# model_history = model.fit(
#     train_dataset,
#     epochs=EPOCHS,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     validation_steps=VALIDATION_STEPS,
#     validation_data=test_dataset,
#     callbacks=[])
# 
# model.save_weights("./test/segmentation")
# 
# print(create_mask(model(tf.ones((1, 512, 512, 3)), False)))
