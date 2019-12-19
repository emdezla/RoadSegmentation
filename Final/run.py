# -*- coding: utf-8 -*-
"""Run script """

ROOT = './'
CHECKPOINT_DIR = ROOT + 'Checkpoints/'

from image_processing import *

    K.clear_session()

    model = load_model(CHECKPOINT_DIR + 'modelUNET.h5', custom_objects=None, compile=True)

    model..model.summary()

    CreateSubmissionUNET(model)

    VisualizeUNETPrediction(model, 18)

