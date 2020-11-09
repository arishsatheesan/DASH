#!/usr/bin/env python
# coding: utf-8

# In[2]:


#This is the working version for both dense and convolution c2 layer. max possible c2=50
# Changes to be made while running: FILTERBATCH_LAST, Model name, Neurons, datawidth_d1

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
#import cv2

#FILTERBATCH_0=2
FILTERBATCH_LAST=16 #This is the filter channels preceding the first dense layer. 
NEURONS_D1=100 #number of neurons in the first dense layer
DATAWIDTH_D1=5 #dimension of input to D1 (refer line 168)
DATAHEIGHT_D1=5
model=Sequential()
model=load_model('lenet4.h5')

def weight_bias_dump():
    c_count, d_count, layer_num_c, layer_num_d = layer_details()
    for i in range(c_count):
        weights = model.layers[layer_num_c[i]].get_weights()
        conv_layer_weight_and_bias_dump(weights,layer_num_c[i])
    for j in range(d_count):
        weights = model.layers[layer_num_d[j]].get_weights()
        Dense_layer_weight_and_bias_dump(weights,layer_num_d[j])

def layer_details():
    n=len(model.layers)
    for i in range(n):
        c_count=0
        d_count=0
        layer_num_c=[]
        layer_num_d=[]
        for i in range(n):
            if ('convolutional') in str(model.layers[i]):
                c_count += 1
                layer_num_c.append(i)
            elif ('Dense') in str(model.layers[i]):
                d_count += 1
                layer_num_d.append(i)
    print ('Number of convolutional layers = '+str(c_count))
    print ('Convolutional layer numbers = '+ str(layer_num_c))
    print ('Number of dense layers = '+str(d_count))
    print ('Dense layer numbers = '+ str(layer_num_d))
    print ('----------------------------------------------------------------')
    return c_count, d_count, layer_num_c, layer_num_d

def conv_layer_weight_and_bias_dump(weights,layer_num):
    array_w = weights[0]
    array_b = weights[1]
    s=array_w.shape
    l=len(s)
    t=1
    for i in range(l):
        t*=s[i]
    nf=max((np.amax(abs(array_w))),(np.amax(abs(array_b)))) #normalizing factor
    array_w_norm = array_w/nf
    array_b_norm = array_b/nf
    #array_w_norm=array_w/(np.amax(abs(array_w)))  # normalizing the weights and biases
    #array_b_norm=array_b/(np.amax(abs(array_b)))
    
    SF = optimal_SF_layer_C(array_w,t)
    SF_b = optimal_SF_layer_bias(array_b)
    print('Scaling factor for weights in convolutional layer '+str(layer_num)+' = '+ str(SF) )
    print('Scaling factor for bias in convolutional layer '+str(layer_num)+' = '+ str(SF_b) )
    array_w_fix = convert_to_fix_point(array_w_norm,6)   #SF set constant 
    array_b_fix = convert_to_fix_point(array_b_norm,6)
    if(layer_num==0):
        weight_dump_conv(array_w_fix, 'c'+str(layer_num)+'.txt',SF)
    #elif((layer_num!=0) & (FILTERBATCH_0==1)):
        #weight_dump_conv(array_w_fix, 'c'+str(layer_num)+'.txt',SF)
    else:
        weight_dump_conv_2(array_w_fix, 'c'+str(layer_num)+'.txt',SF)
    bias_dump_conv(array_b_fix, 'b'+str(layer_num)+'.txt',SF_b)

def Dense_layer_weight_and_bias_dump(weights,layer_num):
    array_w = weights[0]
    array_b = weights[1]
    s=array_w.shape
    l=len(s)
    t=1
    for i in range(l):
        t*=s[i]
        
    array_w_norm=array_w/(np.amax(abs(array_w)))
    array_b_norm=array_b/(np.amax(abs(array_b)))
    
    SF = optimal_SF_layer_C(array_w_norm,t)
    SF_b = optimal_SF_layer_bias(array_b_norm)
    print('Scaling factor for weights in dense layer '+str(layer_num)+' = '+ str(SF) )
    print('Scaling factor for bias in dense layer '+str(layer_num)+' = '+ str(SF_b) )
    array_w_fix = convert_to_fix_point(array_w_norm,SF)
    array_b_fix = convert_to_fix_point(array_b_norm,SF_b)
    if(layer_num==0):
        weight_dump_dense(array_w_fix, 'd'+str(layer_num)+'.txt',SF)
    else:
        weight_dump_dense_2(array_w_fix, 'd'+str(layer_num)+'.txt',SF)
    bias_dump_dense(array_b_fix, 'b'+str(layer_num)+'.txt',SF_b)

def weight_dump_conv_2(array,filename,SF):
    print('Dumping convolutional layer parameter weights in file: {}'.format(filename[:-4]+'_mem.mem'))
    out=open(filename, "w")
    index=0
    n1=array.shape[0]
    n2=array.shape[1]
    n3=array.shape[2]
    n4=array.shape[3]
    array_T=array.transpose()
    arr_rs=array_T
    #arr_rs=array_T.reshape(n4,n1*n2,n3).transpose().reshape(n4,n3,n2*n1)
    
    for i in range(arr_rs.shape[0]):
        for j in range(arr_rs.shape[1]):
            for k in range(arr_rs.shape[2]):
                for l in range(arr_rs.shape[3]):
                    out.write(str(arr_rs[i,j,l,k])+'\n')
                    index += 1
    out.close()
    int_to_hex(filename,SF)
    int_to_hex_trun(filename[:-4]+'_mem.txt', 8)

def weight_dump_conv(array,filename,SF):
    print('Dumping convolutional layer parameter weights in file: {}'.format(filename[:-4]+'_mem.mem'))
    out=open(filename, "w")
    index=0
    n1=array.shape[0]
    n2=array.shape[1]
    n3=array.shape[2]
    n4=array.shape[3]
    array_T=array.transpose(3,0,1,2)
    arr_rs=array_T.reshape(n4,n3,n1,n2)
    for i in range(arr_rs.shape[0]):
        for j in range(arr_rs.shape[1]):
            for k in range(arr_rs.shape[2]):
                for l in range(arr_rs.shape[3]):
                    out.write(str(arr_rs[i,j,k,l])+'\n')
                    index += 1
    out.close()
    int_to_hex(filename,SF)
    int_to_hex_trun(filename[:-4]+'_mem.txt', 8)
    
def bias_dump_conv(array, filename,SF_b):
    print('Dumping parameter bias in file: {}'.format(filename[:-4]+'_mem.mem'))
    print ('----------------------------------------------------------------')
    out=open(filename, "w")
    for i in range(array.shape[0]):
        out.write(str(array[i])+'\n')
    out.close()
    int_to_hex(filename,SF_b)
    int_to_hex_trun(filename[:-4]+'_mem.txt', 8)
    
def weight_dump_dense(array,filename,SF):
    print('Dumping Dense layer parameter weights in file: {}'.format(filename[:-4]+'_mem.mem'))
    array_T = array.transpose(1,0)
    if(FILTERBATCH_LAST>1):
        if(FILTERBATCH_LAST==2):
            array_1=[]
            array_2=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][2*i])
                    array_2.append(array_T[j][2*i+1])
            array_new=np.append(array_1,array_2)
            np.array(array_new)
            out=open(filename, "w")
            for i in range(NEURONS_D1):
                for j in range(DATAWIDTH_D1*DATAHEIGHT_D1):
                    out.write(str(array_new[i*(DATAWIDTH_D1*DATAHEIGHT_D1)+j])+'\n') #25 is DATAWIDTH_D1*DATAHEIGHT_D1
                for k in range(DATAWIDTH_D1*DATAHEIGHT_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+k])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==3):
            array_1=[]
            array_2=[]
            array_3=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][3*i])
                    array_2.append(array_T[j][3*i+1])
                    array_3.append(array_T[j][3*i+2])
            array_new=array_1+array_2+array_3
            np.array(array_new)
            out=open(filename, "w")
            for i in range(NEURONS_D1):
                for j in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[i*(DATAWIDTH_D1*DATAHEIGHT_D1)+j])+'\n')
                for k in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+k])+'\n')
                for l in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+l])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==4):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][4*i])
                    array_2.append(array_T[j][4*i+1])
                    array_3.append(array_T[j][4*i+2])
                    array_4.append(array_T[j][4*i+3])
            array_new=array_1+array_2+array_3+array_4
            np.array(array_new)
            out=open(filename, "w")
            for i in range(NEURONS_D1):
                for j in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[i*(DATAWIDTH_D1*DATAHEIGHT_D1)+j])+'\n')
                for k in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+k])+'\n')
                for l in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+l])+'\n')
                for m in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+m])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==5):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][5*i])
                    array_2.append(array_T[j][5*i+1])
                    array_3.append(array_T[j][5*i+2])
                    array_4.append(array_T[j][5*i+3])
                    array_5.append(array_T[j][5*i+4])
            array_new=array_1+array_2+array_3+array_4+array_5
            np.array(array_new)
            out=open(filename, "w")
            for i in range(NEURONS_D1):
                for j in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[i*(DATAWIDTH_D1*DATAHEIGHT_D1)+j])+'\n')
                for k in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+k])+'\n')
                for l in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+l])+'\n')
                for m in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+m])+'\n')
                for n in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==6):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][6*i])
                    array_2.append(array_T[j][6*i+1])
                    array_3.append(array_T[j][6*i+2])
                    array_4.append(array_T[j][6*i+3])
                    array_5.append(array_T[j][6*i+4])
                    array_6.append(array_T[j][6*i+5])
            array_new=array_1+array_2+array_3+array_4+array_5+array_6
            np.array(array_new)
            out=open(filename, "w")
            for i in range(NEURONS_D1):
                for j in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[i*(DATAWIDTH_D1*DATAHEIGHT_D1)+j])+'\n')
                for k in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+k])+'\n')
                for l in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+l])+'\n')
                for m in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+m])+'\n')
                for n in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n])+'\n')
                for o in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+o])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==7):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][7*i])
                    array_2.append(array_T[j][7*i+1])
                    array_3.append(array_T[j][7*i+2])
                    array_4.append(array_T[j][7*i+3])
                    array_5.append(array_T[j][7*i+4])
                    array_6.append(array_T[j][7*i+5])
                    array_7.append(array_T[j][7*i+6])
            array_new=array_1+array_2+array_3+array_4+array_5+array_6+array_7
            np.array(array_new)
            out=open(filename, "w")
            for i in range(NEURONS_D1):
                for j in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[i*(DATAWIDTH_D1*DATAHEIGHT_D1)+j])+'\n')
                for k in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+k])+'\n')
                for l in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+l])+'\n')
                for m in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+m])+'\n')
                for n in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n])+'\n')
                for o in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+o])+'\n')
                for p in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+p])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==8):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][8*i])
                    array_2.append(array_T[j][8*i+1])
                    array_3.append(array_T[j][8*i+2])
                    array_4.append(array_T[j][8*i+3])
                    array_5.append(array_T[j][8*i+4])
                    array_6.append(array_T[j][8*i+5])
                    array_7.append(array_T[j][8*i+6])
                    array_8.append(array_T[j][8*i+7])
            array_new=array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8
            np.array(array_new)
            out=open(filename, "w")
            for i in range(NEURONS_D1):
                for j in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[i*(DATAWIDTH_D1*DATAHEIGHT_D1)+j])+'\n')
                for k in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+k])+'\n')
                for l in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+l])+'\n')
                for m in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+m])+'\n')
                for n in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n])+'\n')
                for o in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+o])+'\n')
                for p in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+p])+'\n')
                for q in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+q])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==9):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][9*i])
                    array_2.append(array_T[j][9*i+1])
                    array_3.append(array_T[j][9*i+2])
                    array_4.append(array_T[j][9*i+3])
                    array_5.append(array_T[j][9*i+4])
                    array_6.append(array_T[j][9*i+5])
                    array_7.append(array_T[j][9*i+6])
                    array_8.append(array_T[j][9*i+7])
                    array_9.append(array_T[j][9*i+8])
            array_new=array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9
            np.array(array_new)
            out=open(filename, "w")
            for i in range(NEURONS_D1):
                for j in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[i*(DATAWIDTH_D1*DATAHEIGHT_D1)+j])+'\n')
                for k in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+k])+'\n')
                for l in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+l])+'\n')
                for m in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+m])+'\n')
                for n in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n])+'\n')
                for o in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+o])+'\n')
                for p in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+p])+'\n')
                for q in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+q])+'\n')
                for r in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+r])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==10):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][10*i])
                    array_2.append(array_T[j][10*i+1])
                    array_3.append(array_T[j][10*i+2])
                    array_4.append(array_T[j][10*i+3])
                    array_5.append(array_T[j][10*i+4])
                    array_6.append(array_T[j][10*i+5])
                    array_7.append(array_T[j][10*i+6])
                    array_8.append(array_T[j][10*i+7])
                    array_9.append(array_T[j][10*i+8])
                    array_10.append(array_T[j][10*i+9])
            array_new=array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10
            np.array(array_new)
            out=open(filename, "w")
            for i in range(NEURONS_D1):
                for j in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[i*(DATAWIDTH_D1*DATAHEIGHT_D1)+j])+'\n')
                for k in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+k])+'\n')
                for l in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+l])+'\n')
                for m in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+m])+'\n')
                for n in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n])+'\n')
                for o in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+o])+'\n')
                for p in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+p])+'\n')
                for q in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+q])+'\n')
                for r in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+r])+'\n')
                for s in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+s])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==11):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][11*i+0])
                    array_2.append(array_T[j][11*i+1])
                    array_3.append(array_T[j][11*i+2])
                    array_4.append(array_T[j][11*i+3])
                    array_5.append(array_T[j][11*i+4])
                    array_6.append(array_T[j][11*i+5])
                    array_7.append(array_T[j][11*i+6])
                    array_8.append(array_T[j][11*i+7])
                    array_9.append(array_T[j][11*i+8])
                    array_10.append(array_T[j][11*i+9])
                    array_11.append(array_T[j][11*i+10])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==12):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][12*i+0])
                    array_2.append(array_T[j][12*i+1])
                    array_3.append(array_T[j][12*i+2])
                    array_4.append(array_T[j][12*i+3])
                    array_5.append(array_T[j][12*i+4])
                    array_6.append(array_T[j][12*i+5])
                    array_7.append(array_T[j][12*i+6])
                    array_8.append(array_T[j][12*i+7])
                    array_9.append(array_T[j][12*i+8])
                    array_10.append(array_T[j][12*i+9])
                    array_11.append(array_T[j][12*i+10])
                    array_12.append(array_T[j][12*i+11])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==13):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][13*i+0])
                    array_2.append(array_T[j][13*i+1])
                    array_3.append(array_T[j][13*i+2])
                    array_4.append(array_T[j][13*i+3])
                    array_5.append(array_T[j][13*i+4])
                    array_6.append(array_T[j][13*i+5])
                    array_7.append(array_T[j][13*i+6])
                    array_8.append(array_T[j][13*i+7])
                    array_9.append(array_T[j][13*i+8])
                    array_10.append(array_T[j][13*i+9])
                    array_11.append(array_T[j][13*i+10])
                    array_12.append(array_T[j][13*i+11])
                    array_13.append(array_T[j][13*i+12])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==14):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][14*i+0])
                    array_2.append(array_T[j][14*i+1])
                    array_3.append(array_T[j][14*i+2])
                    array_4.append(array_T[j][14*i+3])
                    array_5.append(array_T[j][14*i+4])
                    array_6.append(array_T[j][14*i+5])
                    array_7.append(array_T[j][14*i+6])
                    array_8.append(array_T[j][14*i+7])
                    array_9.append(array_T[j][14*i+8])
                    array_10.append(array_T[j][14*i+9])
                    array_11.append(array_T[j][14*i+10])
                    array_12.append(array_T[j][14*i+11])
                    array_13.append(array_T[j][14*i+12])
                    array_14.append(array_T[j][14*i+13])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==15):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][15*i+0])
                    array_2.append(array_T[j][15*i+1])
                    array_3.append(array_T[j][15*i+2])
                    array_4.append(array_T[j][15*i+3])
                    array_5.append(array_T[j][15*i+4])
                    array_6.append(array_T[j][15*i+5])
                    array_7.append(array_T[j][15*i+6])
                    array_8.append(array_T[j][15*i+7])
                    array_9.append(array_T[j][15*i+8])
                    array_10.append(array_T[j][15*i+9])
                    array_11.append(array_T[j][15*i+10])
                    array_12.append(array_T[j][15*i+11])
                    array_13.append(array_T[j][15*i+12])
                    array_14.append(array_T[j][15*i+13])
                    array_15.append(array_T[j][15*i+14])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==16):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][16*i+0])
                    array_2.append(array_T[j][16*i+1])
                    array_3.append(array_T[j][16*i+2])
                    array_4.append(array_T[j][16*i+3])
                    array_5.append(array_T[j][16*i+4])
                    array_6.append(array_T[j][16*i+5])
                    array_7.append(array_T[j][16*i+6])
                    array_8.append(array_T[j][16*i+7])
                    array_9.append(array_T[j][16*i+8])
                    array_10.append(array_T[j][16*i+9])
                    array_11.append(array_T[j][16*i+10])
                    array_12.append(array_T[j][16*i+11])
                    array_13.append(array_T[j][16*i+12])
                    array_14.append(array_T[j][16*i+13])
                    array_15.append(array_T[j][16*i+14])
                    array_16.append(array_T[j][16*i+15])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==17):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][17*i+0])
                    array_2.append(array_T[j][17*i+1])
                    array_3.append(array_T[j][17*i+2])
                    array_4.append(array_T[j][17*i+3])
                    array_5.append(array_T[j][17*i+4])
                    array_6.append(array_T[j][17*i+5])
                    array_7.append(array_T[j][17*i+6])
                    array_8.append(array_T[j][17*i+7])
                    array_9.append(array_T[j][17*i+8])
                    array_10.append(array_T[j][17*i+9])
                    array_11.append(array_T[j][17*i+10])
                    array_12.append(array_T[j][17*i+11])
                    array_13.append(array_T[j][17*i+12])
                    array_14.append(array_T[j][17*i+13])
                    array_15.append(array_T[j][17*i+14])
                    array_16.append(array_T[j][17*i+15])
                    array_17.append(array_T[j][17*i+16])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==18):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][18*i+0])
                    array_2.append(array_T[j][18*i+1])
                    array_3.append(array_T[j][18*i+2])
                    array_4.append(array_T[j][18*i+3])
                    array_5.append(array_T[j][18*i+4])
                    array_6.append(array_T[j][18*i+5])
                    array_7.append(array_T[j][18*i+6])
                    array_8.append(array_T[j][18*i+7])
                    array_9.append(array_T[j][18*i+8])
                    array_10.append(array_T[j][18*i+9])
                    array_11.append(array_T[j][18*i+10])
                    array_12.append(array_T[j][18*i+11])
                    array_13.append(array_T[j][18*i+12])
                    array_14.append(array_T[j][18*i+13])
                    array_15.append(array_T[j][18*i+14])
                    array_16.append(array_T[j][18*i+15])
                    array_17.append(array_T[j][18*i+16])
                    array_18.append(array_T[j][18*i+17])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==19):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][19*i+0])
                    array_2.append(array_T[j][19*i+1])
                    array_3.append(array_T[j][19*i+2])
                    array_4.append(array_T[j][19*i+3])
                    array_5.append(array_T[j][19*i+4])
                    array_6.append(array_T[j][19*i+5])
                    array_7.append(array_T[j][19*i+6])
                    array_8.append(array_T[j][19*i+7])
                    array_9.append(array_T[j][19*i+8])
                    array_10.append(array_T[j][19*i+9])
                    array_11.append(array_T[j][19*i+10])
                    array_12.append(array_T[j][19*i+11])
                    array_13.append(array_T[j][19*i+12])
                    array_14.append(array_T[j][19*i+13])
                    array_15.append(array_T[j][19*i+14])
                    array_16.append(array_T[j][19*i+15])
                    array_17.append(array_T[j][19*i+16])
                    array_18.append(array_T[j][19*i+17])
                    array_19.append(array_T[j][19*i+18])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==20):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][20*i+0])
                    array_2.append(array_T[j][20*i+1])
                    array_3.append(array_T[j][20*i+2])
                    array_4.append(array_T[j][20*i+3])
                    array_5.append(array_T[j][20*i+4])
                    array_6.append(array_T[j][20*i+5])
                    array_7.append(array_T[j][20*i+6])
                    array_8.append(array_T[j][20*i+7])
                    array_9.append(array_T[j][20*i+8])
                    array_10.append(array_T[j][20*i+9])
                    array_11.append(array_T[j][20*i+10])
                    array_12.append(array_T[j][20*i+11])
                    array_13.append(array_T[j][20*i+12])
                    array_14.append(array_T[j][20*i+13])
                    array_15.append(array_T[j][20*i+14])
                    array_16.append(array_T[j][20*i+15])
                    array_17.append(array_T[j][20*i+16])
                    array_18.append(array_T[j][20*i+17])
                    array_19.append(array_T[j][20*i+18])
                    array_20.append(array_T[j][20*i+19])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==21):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][21*i+0])
                    array_2.append(array_T[j][21*i+1])
                    array_3.append(array_T[j][21*i+2])
                    array_4.append(array_T[j][21*i+3])
                    array_5.append(array_T[j][21*i+4])
                    array_6.append(array_T[j][21*i+5])
                    array_7.append(array_T[j][21*i+6])
                    array_8.append(array_T[j][21*i+7])
                    array_9.append(array_T[j][21*i+8])
                    array_10.append(array_T[j][21*i+9])
                    array_11.append(array_T[j][21*i+10])
                    array_12.append(array_T[j][21*i+11])
                    array_13.append(array_T[j][21*i+12])
                    array_14.append(array_T[j][21*i+13])
                    array_15.append(array_T[j][21*i+14])
                    array_16.append(array_T[j][21*i+15])
                    array_17.append(array_T[j][21*i+16])
                    array_18.append(array_T[j][21*i+17])
                    array_19.append(array_T[j][21*i+18])
                    array_20.append(array_T[j][21*i+19])
                    array_21.append(array_T[j][21*i+20])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==22):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][22*i+0])
                    array_2.append(array_T[j][22*i+1])
                    array_3.append(array_T[j][22*i+2])
                    array_4.append(array_T[j][22*i+3])
                    array_5.append(array_T[j][22*i+4])
                    array_6.append(array_T[j][22*i+5])
                    array_7.append(array_T[j][22*i+6])
                    array_8.append(array_T[j][22*i+7])
                    array_9.append(array_T[j][22*i+8])
                    array_10.append(array_T[j][22*i+9])
                    array_11.append(array_T[j][22*i+10])
                    array_12.append(array_T[j][22*i+11])
                    array_13.append(array_T[j][22*i+12])
                    array_14.append(array_T[j][22*i+13])
                    array_15.append(array_T[j][22*i+14])
                    array_16.append(array_T[j][22*i+15])
                    array_17.append(array_T[j][22*i+16])
                    array_18.append(array_T[j][22*i+17])
                    array_19.append(array_T[j][22*i+18])
                    array_20.append(array_T[j][22*i+19])
                    array_21.append(array_T[j][22*i+20])
                    array_22.append(array_T[j][22*i+21])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==23):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][23*i+0])
                    array_2.append(array_T[j][23*i+1])
                    array_3.append(array_T[j][23*i+2])
                    array_4.append(array_T[j][23*i+3])
                    array_5.append(array_T[j][23*i+4])
                    array_6.append(array_T[j][23*i+5])
                    array_7.append(array_T[j][23*i+6])
                    array_8.append(array_T[j][23*i+7])
                    array_9.append(array_T[j][23*i+8])
                    array_10.append(array_T[j][23*i+9])
                    array_11.append(array_T[j][23*i+10])
                    array_12.append(array_T[j][23*i+11])
                    array_13.append(array_T[j][23*i+12])
                    array_14.append(array_T[j][23*i+13])
                    array_15.append(array_T[j][23*i+14])
                    array_16.append(array_T[j][23*i+15])
                    array_17.append(array_T[j][23*i+16])
                    array_18.append(array_T[j][23*i+17])
                    array_19.append(array_T[j][23*i+18])
                    array_20.append(array_T[j][23*i+19])
                    array_21.append(array_T[j][23*i+20])
                    array_22.append(array_T[j][23*i+21])
                    array_23.append(array_T[j][23*i+22])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==24):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][24*i+0])
                    array_2.append(array_T[j][24*i+1])
                    array_3.append(array_T[j][24*i+2])
                    array_4.append(array_T[j][24*i+3])
                    array_5.append(array_T[j][24*i+4])
                    array_6.append(array_T[j][24*i+5])
                    array_7.append(array_T[j][24*i+6])
                    array_8.append(array_T[j][24*i+7])
                    array_9.append(array_T[j][24*i+8])
                    array_10.append(array_T[j][24*i+9])
                    array_11.append(array_T[j][24*i+10])
                    array_12.append(array_T[j][24*i+11])
                    array_13.append(array_T[j][24*i+12])
                    array_14.append(array_T[j][24*i+13])
                    array_15.append(array_T[j][24*i+14])
                    array_16.append(array_T[j][24*i+15])
                    array_17.append(array_T[j][24*i+16])
                    array_18.append(array_T[j][24*i+17])
                    array_19.append(array_T[j][24*i+18])
                    array_20.append(array_T[j][24*i+19])
                    array_21.append(array_T[j][24*i+20])
                    array_22.append(array_T[j][24*i+21])
                    array_23.append(array_T[j][24*i+22])
                    array_24.append(array_T[j][24*i+23])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==25):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][25*i+0])
                    array_2.append(array_T[j][25*i+1])
                    array_3.append(array_T[j][25*i+2])
                    array_4.append(array_T[j][25*i+3])
                    array_5.append(array_T[j][25*i+4])
                    array_6.append(array_T[j][25*i+5])
                    array_7.append(array_T[j][25*i+6])
                    array_8.append(array_T[j][25*i+7])
                    array_9.append(array_T[j][25*i+8])
                    array_10.append(array_T[j][25*i+9])
                    array_11.append(array_T[j][25*i+10])
                    array_12.append(array_T[j][25*i+11])
                    array_13.append(array_T[j][25*i+12])
                    array_14.append(array_T[j][25*i+13])
                    array_15.append(array_T[j][25*i+14])
                    array_16.append(array_T[j][25*i+15])
                    array_17.append(array_T[j][25*i+16])
                    array_18.append(array_T[j][25*i+17])
                    array_19.append(array_T[j][25*i+18])
                    array_20.append(array_T[j][25*i+19])
                    array_21.append(array_T[j][25*i+20])
                    array_22.append(array_T[j][25*i+21])
                    array_23.append(array_T[j][25*i+22])
                    array_24.append(array_T[j][25*i+23])
                    array_25.append(array_T[j][25*i+24])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==26):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][26*i+0])
                    array_2.append(array_T[j][26*i+1])
                    array_3.append(array_T[j][26*i+2])
                    array_4.append(array_T[j][26*i+3])
                    array_5.append(array_T[j][26*i+4])
                    array_6.append(array_T[j][26*i+5])
                    array_7.append(array_T[j][26*i+6])
                    array_8.append(array_T[j][26*i+7])
                    array_9.append(array_T[j][26*i+8])
                    array_10.append(array_T[j][26*i+9])
                    array_11.append(array_T[j][26*i+10])
                    array_12.append(array_T[j][26*i+11])
                    array_13.append(array_T[j][26*i+12])
                    array_14.append(array_T[j][26*i+13])
                    array_15.append(array_T[j][26*i+14])
                    array_16.append(array_T[j][26*i+15])
                    array_17.append(array_T[j][26*i+16])
                    array_18.append(array_T[j][26*i+17])
                    array_19.append(array_T[j][26*i+18])
                    array_20.append(array_T[j][26*i+19])
                    array_21.append(array_T[j][26*i+20])
                    array_22.append(array_T[j][26*i+21])
                    array_23.append(array_T[j][26*i+22])
                    array_24.append(array_T[j][26*i+23])
                    array_25.append(array_T[j][26*i+24])
                    array_26.append(array_T[j][26*i+25])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==27):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][27*i+0])
                    array_2.append(array_T[j][27*i+1])
                    array_3.append(array_T[j][27*i+2])
                    array_4.append(array_T[j][27*i+3])
                    array_5.append(array_T[j][27*i+4])
                    array_6.append(array_T[j][27*i+5])
                    array_7.append(array_T[j][27*i+6])
                    array_8.append(array_T[j][27*i+7])
                    array_9.append(array_T[j][27*i+8])
                    array_10.append(array_T[j][27*i+9])
                    array_11.append(array_T[j][27*i+10])
                    array_12.append(array_T[j][27*i+11])
                    array_13.append(array_T[j][27*i+12])
                    array_14.append(array_T[j][27*i+13])
                    array_15.append(array_T[j][27*i+14])
                    array_16.append(array_T[j][27*i+15])
                    array_17.append(array_T[j][27*i+16])
                    array_18.append(array_T[j][27*i+17])
                    array_19.append(array_T[j][27*i+18])
                    array_20.append(array_T[j][27*i+19])
                    array_21.append(array_T[j][27*i+20])
                    array_22.append(array_T[j][27*i+21])
                    array_23.append(array_T[j][27*i+22])
                    array_24.append(array_T[j][27*i+23])
                    array_25.append(array_T[j][27*i+24])
                    array_26.append(array_T[j][27*i+25])
                    array_27.append(array_T[j][27*i+26])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==28):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][28*i+0])
                    array_2.append(array_T[j][28*i+1])
                    array_3.append(array_T[j][28*i+2])
                    array_4.append(array_T[j][28*i+3])
                    array_5.append(array_T[j][28*i+4])
                    array_6.append(array_T[j][28*i+5])
                    array_7.append(array_T[j][28*i+6])
                    array_8.append(array_T[j][28*i+7])
                    array_9.append(array_T[j][28*i+8])
                    array_10.append(array_T[j][28*i+9])
                    array_11.append(array_T[j][28*i+10])
                    array_12.append(array_T[j][28*i+11])
                    array_13.append(array_T[j][28*i+12])
                    array_14.append(array_T[j][28*i+13])
                    array_15.append(array_T[j][28*i+14])
                    array_16.append(array_T[j][28*i+15])
                    array_17.append(array_T[j][28*i+16])
                    array_18.append(array_T[j][28*i+17])
                    array_19.append(array_T[j][28*i+18])
                    array_20.append(array_T[j][28*i+19])
                    array_21.append(array_T[j][28*i+20])
                    array_22.append(array_T[j][28*i+21])
                    array_23.append(array_T[j][28*i+22])
                    array_24.append(array_T[j][28*i+23])
                    array_25.append(array_T[j][28*i+24])
                    array_26.append(array_T[j][28*i+25])
                    array_27.append(array_T[j][28*i+26])
                    array_28.append(array_T[j][28*i+27])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==29):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][29*i+0])
                    array_2.append(array_T[j][29*i+1])
                    array_3.append(array_T[j][29*i+2])
                    array_4.append(array_T[j][29*i+3])
                    array_5.append(array_T[j][29*i+4])
                    array_6.append(array_T[j][29*i+5])
                    array_7.append(array_T[j][29*i+6])
                    array_8.append(array_T[j][29*i+7])
                    array_9.append(array_T[j][29*i+8])
                    array_10.append(array_T[j][29*i+9])
                    array_11.append(array_T[j][29*i+10])
                    array_12.append(array_T[j][29*i+11])
                    array_13.append(array_T[j][29*i+12])
                    array_14.append(array_T[j][29*i+13])
                    array_15.append(array_T[j][29*i+14])
                    array_16.append(array_T[j][29*i+15])
                    array_17.append(array_T[j][29*i+16])
                    array_18.append(array_T[j][29*i+17])
                    array_19.append(array_T[j][29*i+18])
                    array_20.append(array_T[j][29*i+19])
                    array_21.append(array_T[j][29*i+20])
                    array_22.append(array_T[j][29*i+21])
                    array_23.append(array_T[j][29*i+22])
                    array_24.append(array_T[j][29*i+23])
                    array_25.append(array_T[j][29*i+24])
                    array_26.append(array_T[j][29*i+25])
                    array_27.append(array_T[j][29*i+26])
                    array_28.append(array_T[j][29*i+27])
                    array_29.append(array_T[j][29*i+28])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==30):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][30*i+0])
                    array_2.append(array_T[j][30*i+1])
                    array_3.append(array_T[j][30*i+2])
                    array_4.append(array_T[j][30*i+3])
                    array_5.append(array_T[j][30*i+4])
                    array_6.append(array_T[j][30*i+5])
                    array_7.append(array_T[j][30*i+6])
                    array_8.append(array_T[j][30*i+7])
                    array_9.append(array_T[j][30*i+8])
                    array_10.append(array_T[j][30*i+9])
                    array_11.append(array_T[j][30*i+10])
                    array_12.append(array_T[j][30*i+11])
                    array_13.append(array_T[j][30*i+12])
                    array_14.append(array_T[j][30*i+13])
                    array_15.append(array_T[j][30*i+14])
                    array_16.append(array_T[j][30*i+15])
                    array_17.append(array_T[j][30*i+16])
                    array_18.append(array_T[j][30*i+17])
                    array_19.append(array_T[j][30*i+18])
                    array_20.append(array_T[j][30*i+19])
                    array_21.append(array_T[j][30*i+20])
                    array_22.append(array_T[j][30*i+21])
                    array_23.append(array_T[j][30*i+22])
                    array_24.append(array_T[j][30*i+23])
                    array_25.append(array_T[j][30*i+24])
                    array_26.append(array_T[j][30*i+25])
                    array_27.append(array_T[j][30*i+26])
                    array_28.append(array_T[j][30*i+27])
                    array_29.append(array_T[j][30*i+28])
                    array_30.append(array_T[j][30*i+29])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==31):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][31*i+0])
                    array_2.append(array_T[j][31*i+1])
                    array_3.append(array_T[j][31*i+2])
                    array_4.append(array_T[j][31*i+3])
                    array_5.append(array_T[j][31*i+4])
                    array_6.append(array_T[j][31*i+5])
                    array_7.append(array_T[j][31*i+6])
                    array_8.append(array_T[j][31*i+7])
                    array_9.append(array_T[j][31*i+8])
                    array_10.append(array_T[j][31*i+9])
                    array_11.append(array_T[j][31*i+10])
                    array_12.append(array_T[j][31*i+11])
                    array_13.append(array_T[j][31*i+12])
                    array_14.append(array_T[j][31*i+13])
                    array_15.append(array_T[j][31*i+14])
                    array_16.append(array_T[j][31*i+15])
                    array_17.append(array_T[j][31*i+16])
                    array_18.append(array_T[j][31*i+17])
                    array_19.append(array_T[j][31*i+18])
                    array_20.append(array_T[j][31*i+19])
                    array_21.append(array_T[j][31*i+20])
                    array_22.append(array_T[j][31*i+21])
                    array_23.append(array_T[j][31*i+22])
                    array_24.append(array_T[j][31*i+23])
                    array_25.append(array_T[j][31*i+24])
                    array_26.append(array_T[j][31*i+25])
                    array_27.append(array_T[j][31*i+26])
                    array_28.append(array_T[j][31*i+27])
                    array_29.append(array_T[j][31*i+28])
                    array_30.append(array_T[j][31*i+29])
                    array_31.append(array_T[j][31*i+30])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==32):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][32*i+0])
                    array_2.append(array_T[j][32*i+1])
                    array_3.append(array_T[j][32*i+2])
                    array_4.append(array_T[j][32*i+3])
                    array_5.append(array_T[j][32*i+4])
                    array_6.append(array_T[j][32*i+5])
                    array_7.append(array_T[j][32*i+6])
                    array_8.append(array_T[j][32*i+7])
                    array_9.append(array_T[j][32*i+8])
                    array_10.append(array_T[j][32*i+9])
                    array_11.append(array_T[j][32*i+10])
                    array_12.append(array_T[j][32*i+11])
                    array_13.append(array_T[j][32*i+12])
                    array_14.append(array_T[j][32*i+13])
                    array_15.append(array_T[j][32*i+14])
                    array_16.append(array_T[j][32*i+15])
                    array_17.append(array_T[j][32*i+16])
                    array_18.append(array_T[j][32*i+17])
                    array_19.append(array_T[j][32*i+18])
                    array_20.append(array_T[j][32*i+19])
                    array_21.append(array_T[j][32*i+20])
                    array_22.append(array_T[j][32*i+21])
                    array_23.append(array_T[j][32*i+22])
                    array_24.append(array_T[j][32*i+23])
                    array_25.append(array_T[j][32*i+24])
                    array_26.append(array_T[j][32*i+25])
                    array_27.append(array_T[j][32*i+26])
                    array_28.append(array_T[j][32*i+27])
                    array_29.append(array_T[j][32*i+28])
                    array_30.append(array_T[j][32*i+29])
                    array_31.append(array_T[j][32*i+30])
                    array_32.append(array_T[j][32*i+31])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==33):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][33*i+0])
                    array_2.append(array_T[j][33*i+1])
                    array_3.append(array_T[j][33*i+2])
                    array_4.append(array_T[j][33*i+3])
                    array_5.append(array_T[j][33*i+4])
                    array_6.append(array_T[j][33*i+5])
                    array_7.append(array_T[j][33*i+6])
                    array_8.append(array_T[j][33*i+7])
                    array_9.append(array_T[j][33*i+8])
                    array_10.append(array_T[j][33*i+9])
                    array_11.append(array_T[j][33*i+10])
                    array_12.append(array_T[j][33*i+11])
                    array_13.append(array_T[j][33*i+12])
                    array_14.append(array_T[j][33*i+13])
                    array_15.append(array_T[j][33*i+14])
                    array_16.append(array_T[j][33*i+15])
                    array_17.append(array_T[j][33*i+16])
                    array_18.append(array_T[j][33*i+17])
                    array_19.append(array_T[j][33*i+18])
                    array_20.append(array_T[j][33*i+19])
                    array_21.append(array_T[j][33*i+20])
                    array_22.append(array_T[j][33*i+21])
                    array_23.append(array_T[j][33*i+22])
                    array_24.append(array_T[j][33*i+23])
                    array_25.append(array_T[j][33*i+24])
                    array_26.append(array_T[j][33*i+25])
                    array_27.append(array_T[j][33*i+26])
                    array_28.append(array_T[j][33*i+27])
                    array_29.append(array_T[j][33*i+28])
                    array_30.append(array_T[j][33*i+29])
                    array_31.append(array_T[j][33*i+30])
                    array_32.append(array_T[j][33*i+31])
                    array_33.append(array_T[j][33*i+32])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==34):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][34*i+0])
                    array_2.append(array_T[j][34*i+1])
                    array_3.append(array_T[j][34*i+2])
                    array_4.append(array_T[j][34*i+3])
                    array_5.append(array_T[j][34*i+4])
                    array_6.append(array_T[j][34*i+5])
                    array_7.append(array_T[j][34*i+6])
                    array_8.append(array_T[j][34*i+7])
                    array_9.append(array_T[j][34*i+8])
                    array_10.append(array_T[j][34*i+9])
                    array_11.append(array_T[j][34*i+10])
                    array_12.append(array_T[j][34*i+11])
                    array_13.append(array_T[j][34*i+12])
                    array_14.append(array_T[j][34*i+13])
                    array_15.append(array_T[j][34*i+14])
                    array_16.append(array_T[j][34*i+15])
                    array_17.append(array_T[j][34*i+16])
                    array_18.append(array_T[j][34*i+17])
                    array_19.append(array_T[j][34*i+18])
                    array_20.append(array_T[j][34*i+19])
                    array_21.append(array_T[j][34*i+20])
                    array_22.append(array_T[j][34*i+21])
                    array_23.append(array_T[j][34*i+22])
                    array_24.append(array_T[j][34*i+23])
                    array_25.append(array_T[j][34*i+24])
                    array_26.append(array_T[j][34*i+25])
                    array_27.append(array_T[j][34*i+26])
                    array_28.append(array_T[j][34*i+27])
                    array_29.append(array_T[j][34*i+28])
                    array_30.append(array_T[j][34*i+29])
                    array_31.append(array_T[j][34*i+30])
                    array_32.append(array_T[j][34*i+31])
                    array_33.append(array_T[j][34*i+32])
                    array_34.append(array_T[j][34*i+33])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==35):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][35*i+0])
                    array_2.append(array_T[j][35*i+1])
                    array_3.append(array_T[j][35*i+2])
                    array_4.append(array_T[j][35*i+3])
                    array_5.append(array_T[j][35*i+4])
                    array_6.append(array_T[j][35*i+5])
                    array_7.append(array_T[j][35*i+6])
                    array_8.append(array_T[j][35*i+7])
                    array_9.append(array_T[j][35*i+8])
                    array_10.append(array_T[j][35*i+9])
                    array_11.append(array_T[j][35*i+10])
                    array_12.append(array_T[j][35*i+11])
                    array_13.append(array_T[j][35*i+12])
                    array_14.append(array_T[j][35*i+13])
                    array_15.append(array_T[j][35*i+14])
                    array_16.append(array_T[j][35*i+15])
                    array_17.append(array_T[j][35*i+16])
                    array_18.append(array_T[j][35*i+17])
                    array_19.append(array_T[j][35*i+18])
                    array_20.append(array_T[j][35*i+19])
                    array_21.append(array_T[j][35*i+20])
                    array_22.append(array_T[j][35*i+21])
                    array_23.append(array_T[j][35*i+22])
                    array_24.append(array_T[j][35*i+23])
                    array_25.append(array_T[j][35*i+24])
                    array_26.append(array_T[j][35*i+25])
                    array_27.append(array_T[j][35*i+26])
                    array_28.append(array_T[j][35*i+27])
                    array_29.append(array_T[j][35*i+28])
                    array_30.append(array_T[j][35*i+29])
                    array_31.append(array_T[j][35*i+30])
                    array_32.append(array_T[j][35*i+31])
                    array_33.append(array_T[j][35*i+32])
                    array_34.append(array_T[j][35*i+33])
                    array_35.append(array_T[j][35*i+34])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==36):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][36*i+0])
                    array_2.append(array_T[j][36*i+1])
                    array_3.append(array_T[j][36*i+2])
                    array_4.append(array_T[j][36*i+3])
                    array_5.append(array_T[j][36*i+4])
                    array_6.append(array_T[j][36*i+5])
                    array_7.append(array_T[j][36*i+6])
                    array_8.append(array_T[j][36*i+7])
                    array_9.append(array_T[j][36*i+8])
                    array_10.append(array_T[j][36*i+9])
                    array_11.append(array_T[j][36*i+10])
                    array_12.append(array_T[j][36*i+11])
                    array_13.append(array_T[j][36*i+12])
                    array_14.append(array_T[j][36*i+13])
                    array_15.append(array_T[j][36*i+14])
                    array_16.append(array_T[j][36*i+15])
                    array_17.append(array_T[j][36*i+16])
                    array_18.append(array_T[j][36*i+17])
                    array_19.append(array_T[j][36*i+18])
                    array_20.append(array_T[j][36*i+19])
                    array_21.append(array_T[j][36*i+20])
                    array_22.append(array_T[j][36*i+21])
                    array_23.append(array_T[j][36*i+22])
                    array_24.append(array_T[j][36*i+23])
                    array_25.append(array_T[j][36*i+24])
                    array_26.append(array_T[j][36*i+25])
                    array_27.append(array_T[j][36*i+26])
                    array_28.append(array_T[j][36*i+27])
                    array_29.append(array_T[j][36*i+28])
                    array_30.append(array_T[j][36*i+29])
                    array_31.append(array_T[j][36*i+30])
                    array_32.append(array_T[j][36*i+31])
                    array_33.append(array_T[j][36*i+32])
                    array_34.append(array_T[j][36*i+33])
                    array_35.append(array_T[j][36*i+34])
                    array_36.append(array_T[j][36*i+35])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==37):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][37*i+0])
                    array_2.append(array_T[j][37*i+1])
                    array_3.append(array_T[j][37*i+2])
                    array_4.append(array_T[j][37*i+3])
                    array_5.append(array_T[j][37*i+4])
                    array_6.append(array_T[j][37*i+5])
                    array_7.append(array_T[j][37*i+6])
                    array_8.append(array_T[j][37*i+7])
                    array_9.append(array_T[j][37*i+8])
                    array_10.append(array_T[j][37*i+9])
                    array_11.append(array_T[j][37*i+10])
                    array_12.append(array_T[j][37*i+11])
                    array_13.append(array_T[j][37*i+12])
                    array_14.append(array_T[j][37*i+13])
                    array_15.append(array_T[j][37*i+14])
                    array_16.append(array_T[j][37*i+15])
                    array_17.append(array_T[j][37*i+16])
                    array_18.append(array_T[j][37*i+17])
                    array_19.append(array_T[j][37*i+18])
                    array_20.append(array_T[j][37*i+19])
                    array_21.append(array_T[j][37*i+20])
                    array_22.append(array_T[j][37*i+21])
                    array_23.append(array_T[j][37*i+22])
                    array_24.append(array_T[j][37*i+23])
                    array_25.append(array_T[j][37*i+24])
                    array_26.append(array_T[j][37*i+25])
                    array_27.append(array_T[j][37*i+26])
                    array_28.append(array_T[j][37*i+27])
                    array_29.append(array_T[j][37*i+28])
                    array_30.append(array_T[j][37*i+29])
                    array_31.append(array_T[j][37*i+30])
                    array_32.append(array_T[j][37*i+31])
                    array_33.append(array_T[j][37*i+32])
                    array_34.append(array_T[j][37*i+33])
                    array_35.append(array_T[j][37*i+34])
                    array_36.append(array_T[j][37*i+35])
                    array_37.append(array_T[j][37*i+36])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==38):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][38*i+0])
                    array_2.append(array_T[j][38*i+1])
                    array_3.append(array_T[j][38*i+2])
                    array_4.append(array_T[j][38*i+3])
                    array_5.append(array_T[j][38*i+4])
                    array_6.append(array_T[j][38*i+5])
                    array_7.append(array_T[j][38*i+6])
                    array_8.append(array_T[j][38*i+7])
                    array_9.append(array_T[j][38*i+8])
                    array_10.append(array_T[j][38*i+9])
                    array_11.append(array_T[j][38*i+10])
                    array_12.append(array_T[j][38*i+11])
                    array_13.append(array_T[j][38*i+12])
                    array_14.append(array_T[j][38*i+13])
                    array_15.append(array_T[j][38*i+14])
                    array_16.append(array_T[j][38*i+15])
                    array_17.append(array_T[j][38*i+16])
                    array_18.append(array_T[j][38*i+17])
                    array_19.append(array_T[j][38*i+18])
                    array_20.append(array_T[j][38*i+19])
                    array_21.append(array_T[j][38*i+20])
                    array_22.append(array_T[j][38*i+21])
                    array_23.append(array_T[j][38*i+22])
                    array_24.append(array_T[j][38*i+23])
                    array_25.append(array_T[j][38*i+24])
                    array_26.append(array_T[j][38*i+25])
                    array_27.append(array_T[j][38*i+26])
                    array_28.append(array_T[j][38*i+27])
                    array_29.append(array_T[j][38*i+28])
                    array_30.append(array_T[j][38*i+29])
                    array_31.append(array_T[j][38*i+30])
                    array_32.append(array_T[j][38*i+31])
                    array_33.append(array_T[j][38*i+32])
                    array_34.append(array_T[j][38*i+33])
                    array_35.append(array_T[j][38*i+34])
                    array_36.append(array_T[j][38*i+35])
                    array_37.append(array_T[j][38*i+36])
                    array_38.append(array_T[j][38*i+37])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==39):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][39*i+0])
                    array_2.append(array_T[j][39*i+1])
                    array_3.append(array_T[j][39*i+2])
                    array_4.append(array_T[j][39*i+3])
                    array_5.append(array_T[j][39*i+4])
                    array_6.append(array_T[j][39*i+5])
                    array_7.append(array_T[j][39*i+6])
                    array_8.append(array_T[j][39*i+7])
                    array_9.append(array_T[j][39*i+8])
                    array_10.append(array_T[j][39*i+9])
                    array_11.append(array_T[j][39*i+10])
                    array_12.append(array_T[j][39*i+11])
                    array_13.append(array_T[j][39*i+12])
                    array_14.append(array_T[j][39*i+13])
                    array_15.append(array_T[j][39*i+14])
                    array_16.append(array_T[j][39*i+15])
                    array_17.append(array_T[j][39*i+16])
                    array_18.append(array_T[j][39*i+17])
                    array_19.append(array_T[j][39*i+18])
                    array_20.append(array_T[j][39*i+19])
                    array_21.append(array_T[j][39*i+20])
                    array_22.append(array_T[j][39*i+21])
                    array_23.append(array_T[j][39*i+22])
                    array_24.append(array_T[j][39*i+23])
                    array_25.append(array_T[j][39*i+24])
                    array_26.append(array_T[j][39*i+25])
                    array_27.append(array_T[j][39*i+26])
                    array_28.append(array_T[j][39*i+27])
                    array_29.append(array_T[j][39*i+28])
                    array_30.append(array_T[j][39*i+29])
                    array_31.append(array_T[j][39*i+30])
                    array_32.append(array_T[j][39*i+31])
                    array_33.append(array_T[j][39*i+32])
                    array_34.append(array_T[j][39*i+33])
                    array_35.append(array_T[j][39*i+34])
                    array_36.append(array_T[j][39*i+35])
                    array_37.append(array_T[j][39*i+36])
                    array_38.append(array_T[j][39*i+37])
                    array_39.append(array_T[j][39*i+38])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==40):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            array_40=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][40*i+0])
                    array_2.append(array_T[j][40*i+1])
                    array_3.append(array_T[j][40*i+2])
                    array_4.append(array_T[j][40*i+3])
                    array_5.append(array_T[j][40*i+4])
                    array_6.append(array_T[j][40*i+5])
                    array_7.append(array_T[j][40*i+6])
                    array_8.append(array_T[j][40*i+7])
                    array_9.append(array_T[j][40*i+8])
                    array_10.append(array_T[j][40*i+9])
                    array_11.append(array_T[j][40*i+10])
                    array_12.append(array_T[j][40*i+11])
                    array_13.append(array_T[j][40*i+12])
                    array_14.append(array_T[j][40*i+13])
                    array_15.append(array_T[j][40*i+14])
                    array_16.append(array_T[j][40*i+15])
                    array_17.append(array_T[j][40*i+16])
                    array_18.append(array_T[j][40*i+17])
                    array_19.append(array_T[j][40*i+18])
                    array_20.append(array_T[j][40*i+19])
                    array_21.append(array_T[j][40*i+20])
                    array_22.append(array_T[j][40*i+21])
                    array_23.append(array_T[j][40*i+22])
                    array_24.append(array_T[j][40*i+23])
                    array_25.append(array_T[j][40*i+24])
                    array_26.append(array_T[j][40*i+25])
                    array_27.append(array_T[j][40*i+26])
                    array_28.append(array_T[j][40*i+27])
                    array_29.append(array_T[j][40*i+28])
                    array_30.append(array_T[j][40*i+29])
                    array_31.append(array_T[j][40*i+30])
                    array_32.append(array_T[j][40*i+31])
                    array_33.append(array_T[j][40*i+32])
                    array_34.append(array_T[j][40*i+33])
                    array_35.append(array_T[j][40*i+34])
                    array_36.append(array_T[j][40*i+35])
                    array_37.append(array_T[j][40*i+36])
                    array_38.append(array_T[j][40*i+37])
                    array_39.append(array_T[j][40*i+38])
                    array_40.append(array_T[j][40*i+39])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39+array_40
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
                for n40 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(39*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n40])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==41):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            array_40=[]
            array_41=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][41*i+0])
                    array_2.append(array_T[j][41*i+1])
                    array_3.append(array_T[j][41*i+2])
                    array_4.append(array_T[j][41*i+3])
                    array_5.append(array_T[j][41*i+4])
                    array_6.append(array_T[j][41*i+5])
                    array_7.append(array_T[j][41*i+6])
                    array_8.append(array_T[j][41*i+7])
                    array_9.append(array_T[j][41*i+8])
                    array_10.append(array_T[j][41*i+9])
                    array_11.append(array_T[j][41*i+10])
                    array_12.append(array_T[j][41*i+11])
                    array_13.append(array_T[j][41*i+12])
                    array_14.append(array_T[j][41*i+13])
                    array_15.append(array_T[j][41*i+14])
                    array_16.append(array_T[j][41*i+15])
                    array_17.append(array_T[j][41*i+16])
                    array_18.append(array_T[j][41*i+17])
                    array_19.append(array_T[j][41*i+18])
                    array_20.append(array_T[j][41*i+19])
                    array_21.append(array_T[j][41*i+20])
                    array_22.append(array_T[j][41*i+21])
                    array_23.append(array_T[j][41*i+22])
                    array_24.append(array_T[j][41*i+23])
                    array_25.append(array_T[j][41*i+24])
                    array_26.append(array_T[j][41*i+25])
                    array_27.append(array_T[j][41*i+26])
                    array_28.append(array_T[j][41*i+27])
                    array_29.append(array_T[j][41*i+28])
                    array_30.append(array_T[j][41*i+29])
                    array_31.append(array_T[j][41*i+30])
                    array_32.append(array_T[j][41*i+31])
                    array_33.append(array_T[j][41*i+32])
                    array_34.append(array_T[j][41*i+33])
                    array_35.append(array_T[j][41*i+34])
                    array_36.append(array_T[j][41*i+35])
                    array_37.append(array_T[j][41*i+36])
                    array_38.append(array_T[j][41*i+37])
                    array_39.append(array_T[j][41*i+38])
                    array_40.append(array_T[j][41*i+39])
                    array_41.append(array_T[j][41*i+40])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39+array_40+array_41
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
                for n40 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(39*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n40])+'\n')
                for n41 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(40*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n41])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==42):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            array_40=[]
            array_41=[]
            array_42=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][42*i+0])
                    array_2.append(array_T[j][42*i+1])
                    array_3.append(array_T[j][42*i+2])
                    array_4.append(array_T[j][42*i+3])
                    array_5.append(array_T[j][42*i+4])
                    array_6.append(array_T[j][42*i+5])
                    array_7.append(array_T[j][42*i+6])
                    array_8.append(array_T[j][42*i+7])
                    array_9.append(array_T[j][42*i+8])
                    array_10.append(array_T[j][42*i+9])
                    array_11.append(array_T[j][42*i+10])
                    array_12.append(array_T[j][42*i+11])
                    array_13.append(array_T[j][42*i+12])
                    array_14.append(array_T[j][42*i+13])
                    array_15.append(array_T[j][42*i+14])
                    array_16.append(array_T[j][42*i+15])
                    array_17.append(array_T[j][42*i+16])
                    array_18.append(array_T[j][42*i+17])
                    array_19.append(array_T[j][42*i+18])
                    array_20.append(array_T[j][42*i+19])
                    array_21.append(array_T[j][42*i+20])
                    array_22.append(array_T[j][42*i+21])
                    array_23.append(array_T[j][42*i+22])
                    array_24.append(array_T[j][42*i+23])
                    array_25.append(array_T[j][42*i+24])
                    array_26.append(array_T[j][42*i+25])
                    array_27.append(array_T[j][42*i+26])
                    array_28.append(array_T[j][42*i+27])
                    array_29.append(array_T[j][42*i+28])
                    array_30.append(array_T[j][42*i+29])
                    array_31.append(array_T[j][42*i+30])
                    array_32.append(array_T[j][42*i+31])
                    array_33.append(array_T[j][42*i+32])
                    array_34.append(array_T[j][42*i+33])
                    array_35.append(array_T[j][42*i+34])
                    array_36.append(array_T[j][42*i+35])
                    array_37.append(array_T[j][42*i+36])
                    array_38.append(array_T[j][42*i+37])
                    array_39.append(array_T[j][42*i+38])
                    array_40.append(array_T[j][42*i+39])
                    array_41.append(array_T[j][42*i+40])
                    array_42.append(array_T[j][42*i+41])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39+array_40+array_41+array_42
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
                for n40 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(39*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n40])+'\n')
                for n41 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(40*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n41])+'\n')
                for n42 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(41*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n42])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==43):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            array_40=[]
            array_41=[]
            array_42=[]
            array_43=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][43*i+0])
                    array_2.append(array_T[j][43*i+1])
                    array_3.append(array_T[j][43*i+2])
                    array_4.append(array_T[j][43*i+3])
                    array_5.append(array_T[j][43*i+4])
                    array_6.append(array_T[j][43*i+5])
                    array_7.append(array_T[j][43*i+6])
                    array_8.append(array_T[j][43*i+7])
                    array_9.append(array_T[j][43*i+8])
                    array_10.append(array_T[j][43*i+9])
                    array_11.append(array_T[j][43*i+10])
                    array_12.append(array_T[j][43*i+11])
                    array_13.append(array_T[j][43*i+12])
                    array_14.append(array_T[j][43*i+13])
                    array_15.append(array_T[j][43*i+14])
                    array_16.append(array_T[j][43*i+15])
                    array_17.append(array_T[j][43*i+16])
                    array_18.append(array_T[j][43*i+17])
                    array_19.append(array_T[j][43*i+18])
                    array_20.append(array_T[j][43*i+19])
                    array_21.append(array_T[j][43*i+20])
                    array_22.append(array_T[j][43*i+21])
                    array_23.append(array_T[j][43*i+22])
                    array_24.append(array_T[j][43*i+23])
                    array_25.append(array_T[j][43*i+24])
                    array_26.append(array_T[j][43*i+25])
                    array_27.append(array_T[j][43*i+26])
                    array_28.append(array_T[j][43*i+27])
                    array_29.append(array_T[j][43*i+28])
                    array_30.append(array_T[j][43*i+29])
                    array_31.append(array_T[j][43*i+30])
                    array_32.append(array_T[j][43*i+31])
                    array_33.append(array_T[j][43*i+32])
                    array_34.append(array_T[j][43*i+33])
                    array_35.append(array_T[j][43*i+34])
                    array_36.append(array_T[j][43*i+35])
                    array_37.append(array_T[j][43*i+36])
                    array_38.append(array_T[j][43*i+37])
                    array_39.append(array_T[j][43*i+38])
                    array_40.append(array_T[j][43*i+39])
                    array_41.append(array_T[j][43*i+40])
                    array_42.append(array_T[j][43*i+41])
                    array_43.append(array_T[j][43*i+42])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39+array_40+array_41+array_42+array_43
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
                for n40 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(39*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n40])+'\n')
                for n41 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(40*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n41])+'\n')
                for n42 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(41*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n42])+'\n')
                for n43 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(42*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n43])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==44):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            array_40=[]
            array_41=[]
            array_42=[]
            array_43=[]
            array_44=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][44*i+0])
                    array_2.append(array_T[j][44*i+1])
                    array_3.append(array_T[j][44*i+2])
                    array_4.append(array_T[j][44*i+3])
                    array_5.append(array_T[j][44*i+4])
                    array_6.append(array_T[j][44*i+5])
                    array_7.append(array_T[j][44*i+6])
                    array_8.append(array_T[j][44*i+7])
                    array_9.append(array_T[j][44*i+8])
                    array_10.append(array_T[j][44*i+9])
                    array_11.append(array_T[j][44*i+10])
                    array_12.append(array_T[j][44*i+11])
                    array_13.append(array_T[j][44*i+12])
                    array_14.append(array_T[j][44*i+13])
                    array_15.append(array_T[j][44*i+14])
                    array_16.append(array_T[j][44*i+15])
                    array_17.append(array_T[j][44*i+16])
                    array_18.append(array_T[j][44*i+17])
                    array_19.append(array_T[j][44*i+18])
                    array_20.append(array_T[j][44*i+19])
                    array_21.append(array_T[j][44*i+20])
                    array_22.append(array_T[j][44*i+21])
                    array_23.append(array_T[j][44*i+22])
                    array_24.append(array_T[j][44*i+23])
                    array_25.append(array_T[j][44*i+24])
                    array_26.append(array_T[j][44*i+25])
                    array_27.append(array_T[j][44*i+26])
                    array_28.append(array_T[j][44*i+27])
                    array_29.append(array_T[j][44*i+28])
                    array_30.append(array_T[j][44*i+29])
                    array_31.append(array_T[j][44*i+30])
                    array_32.append(array_T[j][44*i+31])
                    array_33.append(array_T[j][44*i+32])
                    array_34.append(array_T[j][44*i+33])
                    array_35.append(array_T[j][44*i+34])
                    array_36.append(array_T[j][44*i+35])
                    array_37.append(array_T[j][44*i+36])
                    array_38.append(array_T[j][44*i+37])
                    array_39.append(array_T[j][44*i+38])
                    array_40.append(array_T[j][44*i+39])
                    array_41.append(array_T[j][44*i+40])
                    array_42.append(array_T[j][44*i+41])
                    array_43.append(array_T[j][44*i+42])
                    array_44.append(array_T[j][44*i+43])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39+array_40+array_41+array_42+array_43+array_44
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
                for n40 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(39*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n40])+'\n')
                for n41 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(40*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n41])+'\n')
                for n42 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(41*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n42])+'\n')
                for n43 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(42*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n43])+'\n')
                for n44 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(43*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n44])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==45):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            array_40=[]
            array_41=[]
            array_42=[]
            array_43=[]
            array_44=[]
            array_45=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][45*i+0])
                    array_2.append(array_T[j][45*i+1])
                    array_3.append(array_T[j][45*i+2])
                    array_4.append(array_T[j][45*i+3])
                    array_5.append(array_T[j][45*i+4])
                    array_6.append(array_T[j][45*i+5])
                    array_7.append(array_T[j][45*i+6])
                    array_8.append(array_T[j][45*i+7])
                    array_9.append(array_T[j][45*i+8])
                    array_10.append(array_T[j][45*i+9])
                    array_11.append(array_T[j][45*i+10])
                    array_12.append(array_T[j][45*i+11])
                    array_13.append(array_T[j][45*i+12])
                    array_14.append(array_T[j][45*i+13])
                    array_15.append(array_T[j][45*i+14])
                    array_16.append(array_T[j][45*i+15])
                    array_17.append(array_T[j][45*i+16])
                    array_18.append(array_T[j][45*i+17])
                    array_19.append(array_T[j][45*i+18])
                    array_20.append(array_T[j][45*i+19])
                    array_21.append(array_T[j][45*i+20])
                    array_22.append(array_T[j][45*i+21])
                    array_23.append(array_T[j][45*i+22])
                    array_24.append(array_T[j][45*i+23])
                    array_25.append(array_T[j][45*i+24])
                    array_26.append(array_T[j][45*i+25])
                    array_27.append(array_T[j][45*i+26])
                    array_28.append(array_T[j][45*i+27])
                    array_29.append(array_T[j][45*i+28])
                    array_30.append(array_T[j][45*i+29])
                    array_31.append(array_T[j][45*i+30])
                    array_32.append(array_T[j][45*i+31])
                    array_33.append(array_T[j][45*i+32])
                    array_34.append(array_T[j][45*i+33])
                    array_35.append(array_T[j][45*i+34])
                    array_36.append(array_T[j][45*i+35])
                    array_37.append(array_T[j][45*i+36])
                    array_38.append(array_T[j][45*i+37])
                    array_39.append(array_T[j][45*i+38])
                    array_40.append(array_T[j][45*i+39])
                    array_41.append(array_T[j][45*i+40])
                    array_42.append(array_T[j][45*i+41])
                    array_43.append(array_T[j][45*i+42])
                    array_44.append(array_T[j][45*i+43])
                    array_45.append(array_T[j][45*i+44])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39+array_40+array_41+array_42+array_43+array_44+array_45
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
                for n40 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(39*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n40])+'\n')
                for n41 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(40*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n41])+'\n')
                for n42 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(41*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n42])+'\n')
                for n43 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(42*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n43])+'\n')
                for n44 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(43*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n44])+'\n')
                for n45 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(44*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n45])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==46):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            array_40=[]
            array_41=[]
            array_42=[]
            array_43=[]
            array_44=[]
            array_45=[]
            array_46=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][46*i+0])
                    array_2.append(array_T[j][46*i+1])
                    array_3.append(array_T[j][46*i+2])
                    array_4.append(array_T[j][46*i+3])
                    array_5.append(array_T[j][46*i+4])
                    array_6.append(array_T[j][46*i+5])
                    array_7.append(array_T[j][46*i+6])
                    array_8.append(array_T[j][46*i+7])
                    array_9.append(array_T[j][46*i+8])
                    array_10.append(array_T[j][46*i+9])
                    array_11.append(array_T[j][46*i+10])
                    array_12.append(array_T[j][46*i+11])
                    array_13.append(array_T[j][46*i+12])
                    array_14.append(array_T[j][46*i+13])
                    array_15.append(array_T[j][46*i+14])
                    array_16.append(array_T[j][46*i+15])
                    array_17.append(array_T[j][46*i+16])
                    array_18.append(array_T[j][46*i+17])
                    array_19.append(array_T[j][46*i+18])
                    array_20.append(array_T[j][46*i+19])
                    array_21.append(array_T[j][46*i+20])
                    array_22.append(array_T[j][46*i+21])
                    array_23.append(array_T[j][46*i+22])
                    array_24.append(array_T[j][46*i+23])
                    array_25.append(array_T[j][46*i+24])
                    array_26.append(array_T[j][46*i+25])
                    array_27.append(array_T[j][46*i+26])
                    array_28.append(array_T[j][46*i+27])
                    array_29.append(array_T[j][46*i+28])
                    array_30.append(array_T[j][46*i+29])
                    array_31.append(array_T[j][46*i+30])
                    array_32.append(array_T[j][46*i+31])
                    array_33.append(array_T[j][46*i+32])
                    array_34.append(array_T[j][46*i+33])
                    array_35.append(array_T[j][46*i+34])
                    array_36.append(array_T[j][46*i+35])
                    array_37.append(array_T[j][46*i+36])
                    array_38.append(array_T[j][46*i+37])
                    array_39.append(array_T[j][46*i+38])
                    array_40.append(array_T[j][46*i+39])
                    array_41.append(array_T[j][46*i+40])
                    array_42.append(array_T[j][46*i+41])
                    array_43.append(array_T[j][46*i+42])
                    array_44.append(array_T[j][46*i+43])
                    array_45.append(array_T[j][46*i+44])
                    array_46.append(array_T[j][46*i+45])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39+array_40+array_41+array_42+array_43+array_44+array_45+array_46
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
                for n40 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(39*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n40])+'\n')
                for n41 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(40*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n41])+'\n')
                for n42 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(41*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n42])+'\n')
                for n43 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(42*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n43])+'\n')
                for n44 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(43*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n44])+'\n')
                for n45 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(44*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n45])+'\n')
                for n46 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(45*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n46])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==47):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            array_40=[]
            array_41=[]
            array_42=[]
            array_43=[]
            array_44=[]
            array_45=[]
            array_46=[]
            array_47=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][47*i+0])
                    array_2.append(array_T[j][47*i+1])
                    array_3.append(array_T[j][47*i+2])
                    array_4.append(array_T[j][47*i+3])
                    array_5.append(array_T[j][47*i+4])
                    array_6.append(array_T[j][47*i+5])
                    array_7.append(array_T[j][47*i+6])
                    array_8.append(array_T[j][47*i+7])
                    array_9.append(array_T[j][47*i+8])
                    array_10.append(array_T[j][47*i+9])
                    array_11.append(array_T[j][47*i+10])
                    array_12.append(array_T[j][47*i+11])
                    array_13.append(array_T[j][47*i+12])
                    array_14.append(array_T[j][47*i+13])
                    array_15.append(array_T[j][47*i+14])
                    array_16.append(array_T[j][47*i+15])
                    array_17.append(array_T[j][47*i+16])
                    array_18.append(array_T[j][47*i+17])
                    array_19.append(array_T[j][47*i+18])
                    array_20.append(array_T[j][47*i+19])
                    array_21.append(array_T[j][47*i+20])
                    array_22.append(array_T[j][47*i+21])
                    array_23.append(array_T[j][47*i+22])
                    array_24.append(array_T[j][47*i+23])
                    array_25.append(array_T[j][47*i+24])
                    array_26.append(array_T[j][47*i+25])
                    array_27.append(array_T[j][47*i+26])
                    array_28.append(array_T[j][47*i+27])
                    array_29.append(array_T[j][47*i+28])
                    array_30.append(array_T[j][47*i+29])
                    array_31.append(array_T[j][47*i+30])
                    array_32.append(array_T[j][47*i+31])
                    array_33.append(array_T[j][47*i+32])
                    array_34.append(array_T[j][47*i+33])
                    array_35.append(array_T[j][47*i+34])
                    array_36.append(array_T[j][47*i+35])
                    array_37.append(array_T[j][47*i+36])
                    array_38.append(array_T[j][47*i+37])
                    array_39.append(array_T[j][47*i+38])
                    array_40.append(array_T[j][47*i+39])
                    array_41.append(array_T[j][47*i+40])
                    array_42.append(array_T[j][47*i+41])
                    array_43.append(array_T[j][47*i+42])
                    array_44.append(array_T[j][47*i+43])
                    array_45.append(array_T[j][47*i+44])
                    array_46.append(array_T[j][47*i+45])
                    array_47.append(array_T[j][47*i+46])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39+array_40+array_41+array_42+array_43+array_44+array_45+array_46+array_47
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
                for n40 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(39*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n40])+'\n')
                for n41 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(40*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n41])+'\n')
                for n42 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(41*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n42])+'\n')
                for n43 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(42*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n43])+'\n')
                for n44 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(43*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n44])+'\n')
                for n45 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(44*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n45])+'\n')
                for n46 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(45*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n46])+'\n')
                for n47 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(46*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n47])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==48):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            array_40=[]
            array_41=[]
            array_42=[]
            array_43=[]
            array_44=[]
            array_45=[]
            array_46=[]
            array_47=[]
            array_48=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][48*i+0])
                    array_2.append(array_T[j][48*i+1])
                    array_3.append(array_T[j][48*i+2])
                    array_4.append(array_T[j][48*i+3])
                    array_5.append(array_T[j][48*i+4])
                    array_6.append(array_T[j][48*i+5])
                    array_7.append(array_T[j][48*i+6])
                    array_8.append(array_T[j][48*i+7])
                    array_9.append(array_T[j][48*i+8])
                    array_10.append(array_T[j][48*i+9])
                    array_11.append(array_T[j][48*i+10])
                    array_12.append(array_T[j][48*i+11])
                    array_13.append(array_T[j][48*i+12])
                    array_14.append(array_T[j][48*i+13])
                    array_15.append(array_T[j][48*i+14])
                    array_16.append(array_T[j][48*i+15])
                    array_17.append(array_T[j][48*i+16])
                    array_18.append(array_T[j][48*i+17])
                    array_19.append(array_T[j][48*i+18])
                    array_20.append(array_T[j][48*i+19])
                    array_21.append(array_T[j][48*i+20])
                    array_22.append(array_T[j][48*i+21])
                    array_23.append(array_T[j][48*i+22])
                    array_24.append(array_T[j][48*i+23])
                    array_25.append(array_T[j][48*i+24])
                    array_26.append(array_T[j][48*i+25])
                    array_27.append(array_T[j][48*i+26])
                    array_28.append(array_T[j][48*i+27])
                    array_29.append(array_T[j][48*i+28])
                    array_30.append(array_T[j][48*i+29])
                    array_31.append(array_T[j][48*i+30])
                    array_32.append(array_T[j][48*i+31])
                    array_33.append(array_T[j][48*i+32])
                    array_34.append(array_T[j][48*i+33])
                    array_35.append(array_T[j][48*i+34])
                    array_36.append(array_T[j][48*i+35])
                    array_37.append(array_T[j][48*i+36])
                    array_38.append(array_T[j][48*i+37])
                    array_39.append(array_T[j][48*i+38])
                    array_40.append(array_T[j][48*i+39])
                    array_41.append(array_T[j][48*i+40])
                    array_42.append(array_T[j][48*i+41])
                    array_43.append(array_T[j][48*i+42])
                    array_44.append(array_T[j][48*i+43])
                    array_45.append(array_T[j][48*i+44])
                    array_46.append(array_T[j][48*i+45])
                    array_47.append(array_T[j][48*i+46])
                    array_48.append(array_T[j][48*i+47])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39+array_40+array_41+array_42+array_43+array_44+array_45+array_46+array_47+array_48
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
                for n40 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(39*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n40])+'\n')
                for n41 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(40*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n41])+'\n')
                for n42 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(41*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n42])+'\n')
                for n43 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(42*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n43])+'\n')
                for n44 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(43*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n44])+'\n')
                for n45 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(44*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n45])+'\n')
                for n46 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(45*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n46])+'\n')
                for n47 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(46*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n47])+'\n')
                for n48 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(47*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n48])+'\n')
            out.close()
        elif(FILTERBATCH_LAST==49):
            array_1=[]
            array_2=[]
            array_3=[]
            array_4=[]
            array_5=[]
            array_6=[]
            array_7=[]
            array_8=[]
            array_9=[]
            array_10=[]
            array_11=[]
            array_12=[]
            array_13=[]
            array_14=[]
            array_15=[]
            array_16=[]
            array_17=[]
            array_18=[]
            array_19=[]
            array_20=[]
            array_21=[]
            array_22=[]
            array_23=[]
            array_24=[]
            array_25=[]
            array_26=[]
            array_27=[]
            array_28=[]
            array_29=[]
            array_30=[]
            array_31=[]
            array_32=[]
            array_33=[]
            array_34=[]
            array_35=[]
            array_36=[]
            array_37=[]
            array_38=[]
            array_39=[]
            array_40=[]
            array_41=[]
            array_42=[]
            array_43=[]
            array_44=[]
            array_45=[]
            array_46=[]
            array_47=[]
            array_48=[]
            array_49=[]
            for j in range(NEURONS_D1):
                for i in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    array_1.append(array_T[j][49*i+0])
                    array_2.append(array_T[j][49*i+1])
                    array_3.append(array_T[j][49*i+2])
                    array_4.append(array_T[j][49*i+3])
                    array_5.append(array_T[j][49*i+4])
                    array_6.append(array_T[j][49*i+5])
                    array_7.append(array_T[j][49*i+6])
                    array_8.append(array_T[j][49*i+7])
                    array_9.append(array_T[j][49*i+8])
                    array_10.append(array_T[j][49*i+9])
                    array_11.append(array_T[j][49*i+10])
                    array_12.append(array_T[j][49*i+11])
                    array_13.append(array_T[j][49*i+12])
                    array_14.append(array_T[j][49*i+13])
                    array_15.append(array_T[j][49*i+14])
                    array_16.append(array_T[j][49*i+15])
                    array_17.append(array_T[j][49*i+16])
                    array_18.append(array_T[j][49*i+17])
                    array_19.append(array_T[j][49*i+18])
                    array_20.append(array_T[j][49*i+19])
                    array_21.append(array_T[j][49*i+20])
                    array_22.append(array_T[j][49*i+21])
                    array_23.append(array_T[j][49*i+22])
                    array_24.append(array_T[j][49*i+23])
                    array_25.append(array_T[j][49*i+24])
                    array_26.append(array_T[j][49*i+25])
                    array_27.append(array_T[j][49*i+26])
                    array_28.append(array_T[j][49*i+27])
                    array_29.append(array_T[j][49*i+28])
                    array_30.append(array_T[j][49*i+29])
                    array_31.append(array_T[j][49*i+30])
                    array_32.append(array_T[j][49*i+31])
                    array_33.append(array_T[j][49*i+32])
                    array_34.append(array_T[j][49*i+33])
                    array_35.append(array_T[j][49*i+34])
                    array_36.append(array_T[j][49*i+35])
                    array_37.append(array_T[j][49*i+36])
                    array_38.append(array_T[j][49*i+37])
                    array_39.append(array_T[j][49*i+38])
                    array_40.append(array_T[j][49*i+39])
                    array_41.append(array_T[j][49*i+40])
                    array_42.append(array_T[j][49*i+41])
                    array_43.append(array_T[j][49*i+42])
                    array_44.append(array_T[j][49*i+43])
                    array_45.append(array_T[j][49*i+44])
                    array_46.append(array_T[j][49*i+45])
                    array_47.append(array_T[j][49*i+46])
                    array_48.append(array_T[j][49*i+47])
                    array_49.append(array_T[j][49*i+48])
            array_new=array_1,+array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22+array_23+array_24+array_25+array_26+array_27+array_28+array_29+array_30+array_31+array_32+array_33+array_34+array_35+array_36+array_37+array_38+array_39+array_40+array_41+array_42+array_43+array_44+array_45+array_46+array_47+array_48+array_49
            out=open(filename, "w")
            np.array(array_new)
            for i in range(NEURONS_D1):
                for n1 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(0*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n1])+'\n')
                for n2 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(1*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n2])+'\n')
                for n3 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(2*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n3])+'\n')
                for n4 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(3*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n4])+'\n')
                for n5 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(4*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n5])+'\n')
                for n6 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(5*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n6])+'\n')
                for n7 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(6*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n7])+'\n')
                for n8 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(7*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n8])+'\n')
                for n9 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(8*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n9])+'\n')
                for n10 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(9*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n10])+'\n')
                for n11 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(10*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n11])+'\n')
                for n12 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(11*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n12])+'\n')
                for n13 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(12*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n13])+'\n')
                for n14 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(13*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n14])+'\n')
                for n15 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(14*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n15])+'\n')
                for n16 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(15*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n16])+'\n')
                for n17 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(16*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n17])+'\n')
                for n18 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(17*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n18])+'\n')
                for n19 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(18*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n19])+'\n')
                for n20 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(19*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n20])+'\n')
                for n21 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(20*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n21])+'\n')
                for n22 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(21*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n22])+'\n')
                for n23 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(22*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n23])+'\n')
                for n24 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(23*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n24])+'\n')
                for n25 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(24*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n25])+'\n')
                for n26 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(25*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n26])+'\n')
                for n27 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(26*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n27])+'\n')
                for n28 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(27*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n28])+'\n')
                for n29 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(28*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n29])+'\n')
                for n30 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(29*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n30])+'\n')
                for n31 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(30*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n31])+'\n')
                for n32 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(31*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n32])+'\n')
                for n33 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(32*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n33])+'\n')
                for n34 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(33*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n34])+'\n')
                for n35 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(34*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n35])+'\n')
                for n36 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(35*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n36])+'\n')
                for n37 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(36*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n37])+'\n')
                for n38 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(37*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n38])+'\n')
                for n39 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(38*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n39])+'\n')
                for n40 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(39*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n40])+'\n')
                for n41 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(40*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n41])+'\n')
                for n42 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(41*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n42])+'\n')
                for n43 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(42*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n43])+'\n')
                for n44 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(43*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n44])+'\n')
                for n45 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(44*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n45])+'\n')
                for n46 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(45*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n46])+'\n')
                for n47 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(46*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n47])+'\n')
                for n48 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(47*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n48])+'\n')
                for n49 in range(DATAWIDTH_D1*DATAWIDTH_D1):
                    out.write(str(array_new[(i*(DATAWIDTH_D1*DATAHEIGHT_D1))+(48*DATAWIDTH_D1*DATAHEIGHT_D1*NEURONS_D1)+n49])+'\n')
            out.close()
        int_to_hex(filename,SF)
        int_to_hex_trun(filename[:-4]+'_mem.txt', 8)
        """
        out=open(filename, "w")
        for i in range(array_new.shape[0]):
            out.write(str(array_new[i])+'\n')
        out.close()
        int_to_hex(filename,SF)
        int_to_hex_trun(filename[:-4]+'_mem.txt', 8)"""
    else:
        array_new=array_T
        out=open(filename, "w")
        for i in range(array_new.shape[0]):
            for j in range(array_new.shape[1]):
                out.write(str(array_new[i,j])+'\n')
        out.close()
        int_to_hex(filename,SF)
        int_to_hex_trun(filename[:-4]+'_mem.txt', 8)
    
def weight_dump_dense_2(array,filename,SF):
    print('Dumping Dense layer parameter weights in file: {}'.format(filename[:-4]+'_mem.mem'))
    array_T = array.transpose(1,0)
    array_new=array_T
    out=open(filename, "w")
    for i in range(array_new.shape[0]):
        for j in range(array_new.shape[1]):
            out.write(str(array_new[i,j])+'\n')
    out.close()
    int_to_hex(filename,SF)
    int_to_hex_trun(filename[:-4]+'_mem.txt', 8)
    
    
def bias_dump_dense(array, filename,SF_b):
    print('Dumping Dense layer parameter bias in file: {}'.format(filename[:-4]+'_mem.mem'))
    print ('----------------------------------------------------------------')
    out=open(filename, "w")
    for i in range(array.shape[0]):
        out.write(str(array[i])+'\n')
    out.close()
    int_to_hex(filename,SF_b)
    int_to_hex_trun(filename[:-4]+'_mem.txt', 8)

def convert_to_fix_point(arr1, SF):
    arr2 = arr1.copy().astype(np.float32) #cast to a specified datatype - here float32
    arr2[arr2 < 0] = 0.0   #elements <0 are made 0
    arr2 = np.round(np.abs(arr2) * (2 ** SF)) #multiply with a scaling factor
    arr3 = arr1.copy().astype(np.float32)
    arr3[arr3 > 0] = 0.0 #positive values are set to 0
    arr3 = -np.round(np.abs(-arr3) * (2 ** SF)) #multiply with a scaling factor
    arr4 = arr2 + arr3 
    return arr4.astype(np.int64)

def int_to_hex(filename,SF):
    file_int=open(filename,'r')
    file_hex=open(filename[:-4]+'_mem.txt','w')
    for line in file_int:
        file_hex.write(('{:b}'.format(int(line) & (2**(SF+2)-1))).zfill(SF+2)+'\n')
    file_hex.close()
    file_int.close()
    
    
def int_to_hex_trun(filename,SF):
    file_int=open(filename,'r')
    file_hex=open(filename[:-4]+'.mem','w')
    for line in file_int:
        line_new = line.rstrip()
        if len(line_new)>SF:
            file_hex.write(line_new[0:SF]+'\n')
        elif len(line_new)<SF:
            if line_new[0]=='1':
                file_hex.write( line_new.rjust(8,'1') + '\n')
            else:
                file_hex.write(line_new.zfill(SF) + '\n')
        else:
            file_hex.write(line_new + '\n')
    file_hex.close()
    file_int.close()

def optimal_SF_layer_bias(array):
    max_val=np.amax(abs(array))
    #global SF
    #min_val=np.amin(abs(array))

    if max_val>=0.1:
        SF=6
    elif max_val>=.01:
        SF=8
    elif max_val>=.001:
        SF=10
    elif max_val>=.0001:
        SF=14
    elif max_val>=.00001:
        SF=16
    else:
        SF=16
    return SF
def optimal_SF_layer_C(array,t):
    max_val=np.amax(abs(array))
    #global SF
    #min_val=np.amin(abs(array))
    
    if max_val>=0.1:
        SF=6
    elif max_val>=.01:
        SF=8
    elif max_val>=.001:
        SF=10
    elif max_val>=.0001:
        SF=14
    elif max_val>=.00001:
        SF=16
    else:
        SF=16
    return SF

def single_file_dump():
    c_count, d_count, layer_num_c, layer_num_d = layer_details()
    
    file_s=open('parameters_mem.txt','w')
    file_im=open('image.mem','r')
    for line_i in file_im:
        line = line_i.rstrip()
        file_s.write(line+'\n')
    for i in range(c_count):
        file_c=open('c'+str(layer_num_c[i])+'_mem.txt','r')
        for line_c in file_c:
            line = line_c.rstrip()
            file_s.write(line+'\n')
        file_c.close()
        file_b=open('b'+str(layer_num_c[i])+'_mem.txt','r')
        for line_b in file_b:
            line = line_b.rstrip()
            file_s.write(line+'\n')
        file_b.close()
        
    for j in range(d_count):
        file_d=open('d'+str(layer_num_d[j])+'_mem.txt','r')
        for line_d in file_d:
            line = line_d.rstrip()
            file_s.write(line+'\n')
        file_d.close()
        file_d_b=open('b'+str(layer_num_d[j])+'_mem.txt','r')
        for line_d_b in file_d_b:
            line = line_d_b.rstrip()
            file_s.write(line+'\n')
        file_d_b.close()
    print('Dumping parameters in a single file : parameters.mem')
    file_s.close()
    int_to_hex_trun('parameters_mem.txt', 8)    
       
weight_bias_dump()
single_file_dump()
## Make sure to change FILTERBATCH_LAST




