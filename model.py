from __future__ import division
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras import backend as K
from keras.optimizers import RMSprop
import tensorflow as tf

# 定義一個名為one_obj的函數，帶有三個參數frame_l、joint_n和joint_d，並設定默認值分別為16、15和3
def one_obj(frame_l=16, joint_n=15, joint_d=3):
    
    # 定義一個名為input_joints的輸入層，命名為'joints'，形狀為(frame_l, joint_n, joint_d)
    # 定義一個名為input_joints_diff的輸入層，命名為'joints_diff'，形狀為(frame_l, joint_n, joint_d)
    input_joints = Input(name='joints', shape=(frame_l, joint_n, joint_d))
    input_joints_diff = Input(name='joints_diff', shape=(frame_l, joint_n, joint_d))
    
    # 在input_joints上應用一個Conv2D層
    # filters(決定特徵數)
    # kernel_size(決定捲基核覆蓋區域)
    # padding='same'(表示再輸入數據周圍添加足夠的零填充,確保輸出數據的大小與輸入數據相同)
    # BatchNormalization層 (用於加速訓練並提高模型性能)
    # LeakyReLU層 (非線性激活函數，用於引入非線性並幫助模型學習更複雜的表示)
    ##########branch 1##############
    x = Conv2D(filters = 32, kernel_size=(1,1),padding='same')(input_joints)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(filters = 16, kernel_size=(3,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Permute((1,3,2))(x)
    
    x = Conv2D(filters = 16, kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)   
    ##########branch 1##############
    
    ##########branch 2##############Temporal difference
    x_d = Conv2D(filters = 32, kernel_size=(1,1),padding='same')(input_joints_diff)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
    
    x_d = Conv2D(filters = 16, kernel_size=(3,1),padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)

    x_d = Permute((1,3,2))(x_d)
    
    x_d = Conv2D(filters = 16, kernel_size=(3,3),padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
    ##########branch 2##############
    
    # 用concatenate將x與x_d沿著最後一個維度連起來，已在後續一同處理
    # MaxPool2D層用於對輸入數據進行最大池化操作。是一種下采樣方法，在每個區域中選取最大值作為輸出。
    # MaxPool2D(pool_size=(2, 2))(x)將張量x劃分為若干個2x2大小的區域，並在每個區域中選取最大值作為輸出。
    # Dropout用於在訓練過程中隨機丟棄一些神經元，Dropout(0.1)(x)將張量x中約10%的元素設置為0。
    x = concatenate([x,x_d],axis=-1)
    
    x = Conv2D(filters = 32, kernel_size=(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x) 
    x = Dropout(0.1)(x)
       
    x = Conv2D(filters = 64, kernel_size=(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x) 
    x = Dropout(0.1)(x)
      
    model = Model([input_joints,input_joints_diff],x)

    return model

def multi_obj(frame_l=16, joint_n=15, joint_d=3):
    inp_j_0 = Input(name='inp_j_0', shape=(frame_l, joint_n, joint_d))
    inp_j_diff_0 = Input(name='inp_j_diff_0', shape=(frame_l, joint_n, joint_d))
    
    inp_j_1 = Input(name='inp_j_1', shape=(frame_l, joint_n, joint_d))
    inp_j_diff_1 = Input(name='inp_j_diff_1', shape=(frame_l, joint_n, joint_d))
    
    single = one_obj()
    x_0 = single([inp_j_0,inp_j_diff_0])
    x_1 = single([inp_j_1,inp_j_diff_1])
      
    x = Maximum()([x_0,x_1])
    
    x = Flatten()(x)
    x = Dropout(0.1)(x)
     
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    
    x = Dense(8, activation='sigmoid')(x)
      
    model = Model([inp_j_0,inp_j_diff_0,inp_j_1,inp_j_diff_1],x)
    
    return model