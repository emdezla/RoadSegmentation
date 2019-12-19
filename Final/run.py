# -*- coding: utf-8 -*-
"""Run script """


from image_processing import *

    K.clear_session()

    model = load_model(CHECKPOINT_DIR + 'UNET_V2.hdf5', custom_objects=None, compile=True)

    model..model.summary()

    CreateSubmissionUNET(model)

    VisualizeUNETPrediction(model, 18)

