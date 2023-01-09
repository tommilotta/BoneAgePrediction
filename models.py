from inc_v4 import *
from simple_conv import *

def create_model(model_name, img_shape, metric, with_gender=True):
    if model_name == 'inceptionV4':
        model = Bone_Age_incV4(img_shape, with_gender)
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[metric])
    if model_name == 'simple_conv':
        model = Simple_Conv(img_shape)
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[metric])
    return model