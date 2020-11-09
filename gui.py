from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from contextlib import redirect_stdout
import numpy as np

import sys
import os
from tkinter import * 
from tkinter import messagebox  
from tkinter import ttk 
top = Tk()  
top.title("Hardware Code Generation")     # Add a title
top.style=ttk.Style()
top.style.theme_use("xpnative")
import tkinter.simpledialog
#s = ttk.Style()
#s.theme_use('xpnative')
top.geometry("750x800") 

def train_net(*args):
        m=training_var.get()
        x=cnn_variable.get()
        y=conv_layer.get()
        z=dense_layer.get()
        t=train_button.get()
        p=pipeline_variable.get()
        if(m==1):
            cnn_select.config(state="disabled")
            cnn_dropdown.config(state="disabled")
            conv.config(state="disabled")
            convol.config(state="disabled")
            c1_kernel.config(state="disabled")
            c1_k_dropdown.config(state="disabled")
            c1_stride.config(state="disabled")
            c1stride_dropdown.config(state="disabled")
            filterbatch.config(state="disabled")
            c1_filter.config(state="disabled")
            c1_filter_dropdown.config(state="disabled")
            c2_filter.config(state="disabled")
            c2_filter_dropdown.config(state="disabled")
            maxpool.config(state="disabled")
            m1_kernel.config(state="disabled")
            m1_stride.config(state="disabled")
            m1_kernel_dropdown.config(state="disabled")
            m1_stride_dropdown.config(state="disabled")
            dense.config(state="disabled")
            e2.config(state="disabled")
            neurons.config(state="disabled")
            d1_neurons.config(state="disabled")
            d1_n_dropdown.config(state="disabled")
            d2_neurons.config(state="disabled")
            d2_n_dropdown.config(state="disabled")
            d3_neurons.config(state="disabled")
            d3_n_dropdown.config(state="disabled")
            train.config(state="disabled")
            upload.config(state="normal")
            #model_summary
            #summary
        elif(m==2):
            upload.config(state="disabled")
            cnn_select.config(state="normal")
            cnn_dropdown.config(state="normal")
            if(x=="default"):
                print(m)
                conv.config(state="disabled")
                convol.config(state="disabled")
                c1_kernel.config(state="disabled")
                c1_k_dropdown.config(state="disabled")
                c1_stride.config(state="disabled")
                c1stride_dropdown.config(state="disabled")
                filterbatch.config(state="disabled")
                c1_filter.config(state="disabled")
                c1_filter_dropdown.config(state="disabled")
                c2_filter.config(state="disabled")
                c2_filter_dropdown.config(state="disabled")
                maxpool.config(state="disabled")
                m1_kernel.config(state="disabled")
                m1_stride.config(state="disabled")
                m1_kernel_dropdown.config(state="disabled")
                m1_stride_dropdown.config(state="disabled")
                dense.config(state="disabled")
                e2.config(state="disabled")
                neurons.config(state="disabled")
                d1_neurons.config(state="disabled")
                d1_n_dropdown.config(state="disabled")
                d2_neurons.config(state="disabled")
                d2_n_dropdown.config(state="disabled")
                d3_neurons.config(state="disabled")
                d3_n_dropdown.config(state="disabled")
                train.config(state="normal")
            elif(x=="Custom"):
                print(x)
                conv.config(state="normal")
                convol.config(state="normal")
                c1_kernel.config(state="disabled")
                c1_k_dropdown.config(state="disabled")
                c1_stride.config(state="disabled")
                c1stride_dropdown.config(state="disabled")
                filterbatch.config(state="disabled")
                c1_filter.config(state="disabled")
                c1_filter_dropdown.config(state="disabled")
                c2_filter.config(state="disabled")
                c2_filter_dropdown.config(state="disabled")
                maxpool.config(state="disabled")
                m1_kernel.config(state="disabled")
                m1_stride.config(state="disabled")
                m1_kernel_dropdown.config(state="disabled")
                m1_stride_dropdown.config(state="disabled")
                dense.config(state="normal")
                e2.config(state="normal")
                neurons.config(state="disabled")
                d1_neurons.config(state="disabled")
                d1_n_dropdown.config(state="disabled")
                d2_neurons.config(state="disabled")
                d2_n_dropdown.config(state="disabled")
                d3_neurons.config(state="disabled")
                d3_n_dropdown.config(state="disabled")
                train.config(state="disabled")
                if(y=="default"):
                    conv.config(state="normal")
                    convol.config(state="normal")
                    c1_kernel.config(state="disabled")
                    c1_k_dropdown.config(state="disabled")
                    c1_stride.config(state="disabled")
                    c1stride_dropdown.config(state="disabled")
                    filterbatch.config(state="disabled")
                    c1_filter.config(state="disabled")
                    c1_filter_dropdown.config(state="disabled")
                    c2_filter.config(state="disabled")
                    c2_filter_dropdown.config(state="disabled")
                    maxpool.config(state="disabled")
                    m1_kernel.config(state="disabled")
                    m1_stride.config(state="disabled")
                    m1_kernel_dropdown.config(state="disabled")
                    m1_stride_dropdown.config(state="disabled")
                    dense.config(state="normal")
                    e2.config(state="normal")
                    neurons.config(state="disabled")
                    d1_neurons.config(state="disabled")
                    d1_n_dropdown.config(state="disabled")
                    d2_neurons.config(state="disabled")
                    d2_n_dropdown.config(state="disabled")
                    d3_neurons.config(state="disabled")
                    d3_n_dropdown.config(state="disabled")
                    train.config(state="normal")
                elif(y=="1"):
                    conv.config(state="normal")
                    convol.config(state="normal")
                    c1_kernel.config(state="normal")
                    c1_k_dropdown.config(state="normal")
                    c1_stride.config(state="normal")
                    c1stride_dropdown.config(state="normal")
                    filterbatch.config(state="normal")
                    c1_filter.config(state="normal")
                    c1_filter_dropdown.config(state="normal")
                    c2_filter.config(state="disabled")
                    c2_filter_dropdown.config(state="disabled")
                    maxpool.config(state="normal")
                    m1_kernel.config(state="normal")
                    m1_stride.config(state="normal")
                    m1_kernel_dropdown.config(state="normal")
                    m1_stride_dropdown.config(state="normal")
                    dense.config(state="normal")
                    e2.config(state="normal")
                    neurons.config(state="disabled")
                    d1_neurons.config(state="disabled")
                    d1_n_dropdown.config(state="disabled")
                    d2_neurons.config(state="disabled")
                    d2_n_dropdown.config(state="disabled")
                    d3_neurons.config(state="disabled")
                    d3_n_dropdown.config(state="disabled")
                    if(z=="default"):
                        train.config(state="normal")
                    else:
                        train.config(state="disabled")
                elif(y=="2"):
                    conv.config(state="normal")
                    convol.config(state="normal")
                    c1_kernel.config(state="normal")
                    c1_k_dropdown.config(state="normal")
                    c1_stride.config(state="normal")
                    c1stride_dropdown.config(state="normal")
                    filterbatch.config(state="normal")
                    c1_filter.config(state="normal")
                    c1_filter_dropdown.config(state="normal")
                    c2_filter.config(state="normal")
                    c2_filter_dropdown.config(state="normal")
                    maxpool.config(state="normal")
                    m1_kernel.config(state="normal")
                    m1_stride.config(state="normal")
                    m1_kernel_dropdown.config(state="normal")
                    m1_stride_dropdown.config(state="normal")
                    dense.config(state="normal")
                    e2.config(state="normal")
                    neurons.config(state="disabled")
                    d1_neurons.config(state="disabled")
                    d1_n_dropdown.config(state="disabled")
                    d2_neurons.config(state="disabled")
                    d2_n_dropdown.config(state="disabled")
                    d3_neurons.config(state="disabled")
                    d3_n_dropdown.config(state="disabled")
                    if(z=="default"):
                        train.config(state="normal")
                    else:
                        train.config(state="disabled")
            if(z=="1"):
#                conv.config(state="normal")
#                convol.config(state="normal")
#                c1_kernel.config(state="normal")
#                c1_k_dropdown.config(state="normal")
#                c1_stride.config(state="normal")
#                c1stride_dropdown.config(state="normal")
#                filterbatch.config(state="normal")
#                c1_filter.config(state="normal")
#                c1_filter_dropdown.config(state="normal")
#                c2_filter.config(state="normal")
#                c2_filter_dropdown.config(state="normal")
#                maxpool.config(state="normal")
#                m1_kernel.config(state="normal")
#                m1_stride.config(state="normal")
#                m1_kernel_dropdown.config(state="normal")
#                m1_stride_dropdown.config(state="normal")
                dense.config(state="normal")
                e2.config(state="normal")
                neurons.config(state="normal")
                d1_neurons.config(state="normal")
                d1_n_dropdown.config(state="normal")
                d2_neurons.config(state="disabled")
                d2_n_dropdown.config(state="disabled")
                d3_neurons.config(state="disabled")
                d3_n_dropdown.config(state="disabled")
                train.config(state="normal")
            elif(z=="2"):
#                conv.config(state="normal")
#                convol.config(state="normal")
#                c1_kernel.config(state="normal")
#                c1_k_dropdown.config(state="normal")
#                c1_stride.config(state="normal")
#                c1stride_dropdown.config(state="normal")
#                filterbatch.config(state="normal")
#                c1_filter.config(state="normal")
#                c1_filter_dropdown.config(state="normal")
#                c2_filter.config(state="normal")
#                c2_filter_dropdown.config(state="normal")
#                maxpool.config(state="normal")
#                m1_kernel.config(state="normal")
#                m1_stride.config(state="normal")
#                m1_kernel_dropdown.config(state="normal")
#                m1_stride_dropdown.config(state="normal")
                dense.config(state="normal")
                e2.config(state="normal")
                neurons.config(state="normal")
                d1_neurons.config(state="normal")
                d1_n_dropdown.config(state="normal")
                d2_neurons.config(state="normal")
                d2_n_dropdown.config(state="normal")
                d3_neurons.config(state="disabled")
                d3_n_dropdown.config(state="disabled")
                train.config(state="normal")
            elif(z=="3"):
#                conv.config(state="normal")
#                convol.config(state="normal")
#                c1_kernel.config(state="normal")
#                c1_k_dropdown.config(state="normal")
#                c1_stride.config(state="normal")
#                c1stride_dropdown.config(state="normal")
#                filterbatch.config(state="normal")
#                c1_filter.config(state="normal")
#                c1_filter_dropdown.config(state="normal")
#                c2_filter.config(state="normal")
#                c2_filter_dropdown.config(state="normal")
#                maxpool.config(state="normal")
#                m1_kernel.config(state="normal")
#                m1_stride.config(state="normal")
#                m1_kernel_dropdown.config(state="normal")
#                m1_stride_dropdown.config(state="normal")
                dense.config(state="normal")
                e2.config(state="normal")
                neurons.config(state="normal")
                d1_neurons.config(state="normal")
                d1_n_dropdown.config(state="normal")
                d2_neurons.config(state="normal")
                d2_n_dropdown.config(state="normal")
                d3_neurons.config(state="normal")
                d3_n_dropdown.config(state="normal")
                train.config(state="normal")          
                
def upload_h5():
    messagebox.showinfo("info", "This feature is not available at the moment!")
def train(*args):
    x=cnn_variable.get()
    if(x=="LeNet-5"):
        os.system('python LeNet_5_py.py')
        messagebox.showinfo("info", "Model trained!")
    elif(x=="LeNet-4"):
        os.system('python CNN_model_gen_LeNet4.py')
        messagebox.showinfo("info", "Model trained!")
    elif(x=="LeNet-1"):
        os.system('python CNN_model_gen_LeNet1.py')
        messagebox.showinfo("info", "Model trained!")
    elif(x=="default"):
        os.system('python CNN_model_gen_LeNet1.py')
        messagebox.showinfo("info", "Model trained!")
    elif(x=="Custom"):
        c=conv_layer.get()
        d=dense_layer.get()
        if(c1k_var.get()=="default"):
            ck=3
        else:
            ck=int(c1k_var.get())
            
        if(c1stride_var.get()=="default"):
            cs=1
        else:
            cs=int(c1stride_var.get())
            
        if(c1f_var.get()=="default"):
            c1f=4
        else:
            c1f=int(c1f_var.get())
            
        if(c2f_var.get()=="default"):
            c2f=4
        else:
            c2f=int(c2f_var.get())
            
        if(m1k_var.get()=="default"):
            mk=2
        else:
            mk=int(m1k_var.get())
            
        if(m1s_var.get()=="default"):
            ms=2
        else:
            ms=int(m1s_var.get())
            
        if(d1n_var.get()=="default"):
            d1=10
        else:
            d1=int(d1n_var.get())
        if(d2n_var.get()=="default"):
            d2=10
        else:
            d2=int(d2n_var.get())
        if(d3n_var.get()=="default"):
            d3=10
        else:
            d3=int(d3n_var.get())
        
        batch_size = 128
        num_classes = 10
        epochs = 2
        
        # input image dimensions
        img_rows, img_cols = 28, 28
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
            
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        model=Sequential()
        if(c=="2"):
            model.add(Conv2D(c1f, kernel_size=(ck, ck), activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(c2f, (ck, ck), activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
            model.add(Dropout(0.25))
            model.add(Flatten())
        else:
            model.add(Conv2D(c1f, kernel_size=(ck, ck), activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            #model.add(Conv2D(16, (3, 3), activation='relu'))
            #model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
            model.add(Dropout(0.25))
            model.add(Flatten())
        #model.add(Dense(20))
        if(d=="1"):
            model.add(Dense(d1))
            model.add(Activation('softmax'))
        elif(d=="2"):
            model.add(Dense(d1))
            model.add(Dense(d2))
            model.add(Activation('softmax'))
        elif(d=="3"):
            model.add(Dense(d1))
            model.add(Dense(d2))
            model.add(Dense(d3))
            model.add(Activation('softmax'))
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        model.save('lenet1.h5')
        
        with open('modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                model.layers
                model.summary()
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])
        messagebox.showinfo("info", "Model trained!")
#############################################################################
    else:
        os.system('python CNN_model_gen_LeNet1.py')
        messagebox.showinfo("info", "Model trained!")
def view_resource(*args):
    os.system('python display_resource.py')
    
def view_summary():
    os.system('python display.py')
    
def simulate_tcl():
    messagebox.showinfo("info", "This feature is not yet available!") 
    
def gen_ver(*args):
    g=generate_rad.get()
    p=pipeline_variable.get()
    x=cnn_variable.get()
    if(x=="LeNet-5"):
        if(p=="Fully pipelined"):
            if(g==1):
                os.system('python weight_bias_dump_lenet5.py')
                os.system('python Lenet5_pipelined_ver_gen.py')
                messagebox.showinfo("info", "Verilog code generated with filename 'top.v'!") 
            elif(g==2):
                messagebox.showinfo("info", "Bitstream generation is not yet available!") 
        elif(p=="Pipelined sequential"):
            if(g==1):
                os.system('python weight_bias_dump_lenet5.py')
                os.system('python Lenet5_ver_gen_seq.py')
                messagebox.showinfo("info", "Verilog code generated with filename 'top.v'!") 
            elif(g==2):
                messagebox.showinfo("info", "Bitstream generation is not yet available!") 
        else:
            messagebox.showinfo("info", "Sequential only version is not available!") 
    elif(x=="LeNet-4"):
        if(p=="Fully pipelined"):
            if(g==1):
                os.system('python weight_bias_dump_lenet4.py')
                os.system('python Lenet5_pipelined_ver_gen.py')
                messagebox.showinfo("info", "Verilog code generated with filename 'top.v'!") 
            elif(g==2):
                messagebox.showinfo("info", "Bitstream generation is not yet available!") 
        elif(p=="Pipelined sequential"):
            if(g==1):
                os.system('python weight_bias_dump_lenet4.py')
                os.system('python LeNet4_sequen_ver_gen.py')
                messagebox.showinfo("info", "Verilog code generated with filename 'top.v'!") 
            elif(g==2):
                messagebox.showinfo("info", "Bitstream generation is not yet available!") 
        else:
            messagebox.showinfo("info", "Sequential only version is not available!") 
    elif(x=="LeNet-1"):
        if(p=="Fully pipelined"):
            if(g==1):
                os.system('python weight_bias_dump_lenet1.py')
                os.system('python Lenet1_pipelined_ver_gen.py')
                messagebox.showinfo("info", "Verilog code generated with filename 'top.v'!") 
            elif(g==2):
                messagebox.showinfo("info", "Bitstream generation is not yet available!") 
        elif(p=="Pipelined sequential"):
            if(g==1):
                os.system('python weight_bias_dump_lenet1.py')
                os.system('python Lenet1_pipelined_ver_gen.py')
                messagebox.showinfo("info", "Verilog code generated with filename 'top.v'!") 
            elif(g==2):
                messagebox.showinfo("info", "Bitstream generation is not yet available!") 
        else:
            messagebox.showinfo("info", "Sequential only version is not available!") 
    else:
        messagebox.showinfo("info", "Only LeNet-5 generation is available!")

def disable_entry(*args):
    x=cnn_variable.get()
    y=conv_layer.get()
    z=dense_layer.get()
    t=train_button.get()
    p=pipeline_variable.get()

cnn = ["default", "LeNet-1", "LeNet-4","LeNet-5", "Custom"]
convolution=["default", "1", "2"]
dense_l=["default", "1", "2", "3"]
pipeline=["default", "Sequential", "Pipelined sequential", "Fully pipelined"]
c1_k=["default", "3", "5"]
c1_s=["default", "1"]
c1_f=["default", "1","2","3","4","5","6","7","8","9","10","11","12","13","14",
      "15","16","17","18","19","20"]
c2_f=["default", "1","2","3","4","5","6","7","8","9","10","11","12","13","14",
      "15","16","17","18","19","20"]
m1_k=["default", "2", "3"]
m1_s=["default", "2"]
c2_k=["default", "3", "5"]
c2_s=["default", "1"]
c2_f=["default", "1","2","3","4","5","6","7","8","9","10","11","12","13","14",
      "15","16","17","18","19","20"]
m2_k=["default", "2", "3"]
m2_s=["default", "2"]
d1_n=["default", "10", "20", "32", "84", "100", "120"]
d2_n=["default", "10", "20", "32", "84", "100", "120"]
d3_n=["default", "10", "20", "32", "84", "100", "120"]

cnn_variable=StringVar(top)
cnn_variable.set("default")
pipeline_variable=StringVar(top)
pipeline_variable.set("default")
conv_layer=StringVar(top)
conv_layer.set("default")
dense_layer=StringVar(top)
dense_layer.set("default")
train_button=StringVar(top)
model_summary=StringVar(top)
generate_rad=IntVar(top)
generate_b=IntVar(top)
training_var=IntVar(top)


c1k_var = StringVar(top)
c1k_var.set("default")
c1stride_var=StringVar(top)
c1stride_var.set("default")
c1f_var=StringVar(top)
c1f_var.set("default")
c2f_var=StringVar(top)
c2f_var.set("default")
m1k_var = StringVar(top)
m1k_var.set("default")
m1s_var = StringVar(top)
m1s_var.set("default")
d1n_var = StringVar(top)
d2n_var = StringVar(top)
d3n_var = StringVar(top)
d1n_var.set("default")
d2n_var.set("default")
d3n_var.set("default")

cnn_variable.trace("w",train_net)
conv_layer.trace("w",train_net)
dense_layer.trace("w",train_net)
pipeline_variable.trace("w",train_net)

cnn_select = Label(top, text = "Choose CNN Model")
conv = Label(top, text = "No. of Convolution layers")
c1_kernel=Label(top,text = "Kernel Size")
c1_stride=Label(top,text = "Stride Width")
filterbatch=Label(top,text = "Filter batch:")
c1_filter=Label(top,text = "C1")
c2_filter=Label(top,text = "C2")
maxpool=Label(top, text="Maxpool Layer Parameters:")
m1_kernel=Label(top,text = "Kernel Size")
m1_stride=Label(top,text = "Stride Width")
c2_kernel=Label(top,text = "Convolution layer 2 kernel size")
c2_stride=Label(top,text = "Convolution layer 2 Stride width")
c2_filterbatch=Label(top,text = "Convolution layer 2 Filter batch size")
m2_kernel=Label(top,text = "Maxpool layer 2 kernel size")
m2_stride=Label(top,text = "Maxpool layer 2 Stride width")
dense = Label(top, text = "No. of Dense layers")
neurons=Label(top,text = "Number of neurons:")
d1_neurons=Label(top,text = "Layer 1")
d2_neurons=Label(top,text = "Layer 2")
d3_neurons=Label(top,text = "Layer 3")
training_options=Label(top,text = "Model Training")
pipeline_select = Label(top, text = "Pipelining Selection")
resource = Label(top, text = "Calculated resource requirements")
res_or=Label(top, text="or")
code_gen=Label(top, text="Automatic Code Generation")
model_summary=Label(top, text="View Model Summary")


ttk.Separator(top).place(x=0, y=495, relwidth=1)

cnn_dropdown=OptionMenu(top,cnn_variable,*cnn)

convol = OptionMenu(top, conv_layer, *convolution)
c1_k_dropdown=OptionMenu(top,c1k_var,*c1_k)
c1stride_dropdown=OptionMenu(top,c1stride_var,*c1_s)
c1_filter_dropdown=OptionMenu(top,c1f_var,*c1_f)
c2_filter_dropdown=OptionMenu(top,c2f_var,*c2_f)
m1_kernel_dropdown=OptionMenu(top,m1k_var,*m1_k)
m1_stride_dropdown=OptionMenu(top,m1s_var,*m1_s)
d1_n_dropdown=OptionMenu(top,d1n_var,*d1_n)
d2_n_dropdown=OptionMenu(top,d2n_var,*d2_n)
d3_n_dropdown=OptionMenu(top,d3n_var,*d3_n)

e2 = OptionMenu(top, dense_layer, *dense_l)
train= Button(top, command=train, text=" Train ")
upload=Button(top, command=upload_h5, text="Upload")
summary= Button(top, command=view_summary, text="Summary")
pipeline_dropdown=OptionMenu(top,pipeline_variable, *pipeline)
resource_dwnld=Button(top, command=view_resource, text="  View  ")
resource_sim=Button(top, command=simulate_tcl, text="Simulate")
R1 = Radiobutton(top, text="Generate Verilog Code", variable=generate_rad, value=1)
R2 = Radiobutton(top, text="Generate Bitstream", variable=generate_rad, value=2)
generate_button=Button(top, command=gen_ver, text="Generate")
hdf5=Radiobutton(top, text="Upload HDF5 file", variable=training_var, command=train_net, value=1)
training_rad=Radiobutton(top, text="Train the network using given parameters", variable=training_var, 
                         command=train_net, value=2)

training_options.place(x= 30, y=50)
hdf5.place(x= 50, y=75)
training_rad.place(x= 50, y=100)

cnn_select.place(x=30, y=138)
cnn_dropdown.place(x=200, y=135)
conv.place(x = 30,y = 178)  
convol.place(x = 200, y = 175)
c1_kernel.place(x=325, y=178)
c1_k_dropdown.place(x=400, y=175)
c1_stride.place(x=500, y=178)
c1stride_dropdown.place(x=585, y=175)
filterbatch.place(x=325, y=218)
c1_filter.place(x=410, y=218)
c1_filter_dropdown.place(x=430, y=215)
c2_filter.place(x=530, y=218)
c2_filter_dropdown.place(x=550, y=215)
maxpool.place(x=30, y=263)
m1_kernel.place(x=225, y=263)
m1_stride.place(x=400, y=263)
m1_kernel_dropdown.place(x=300, y=260)
m1_stride_dropdown.place(x=485, y=260)
dense.place(x = 30, y = 305) 
e2.place(x = 200, y = 305)
neurons.place(x=30, y=350)
d1_neurons.place(x=200, y=353)
d1_n_dropdown.place(x=250, y=350)
d2_neurons.place(x=360, y=353)
d2_n_dropdown.place(x=410, y=350)
d3_neurons.place(x=520, y=353)
d3_n_dropdown.place(x=570, y=350)

upload.place(x=200, y=410)
train.place(x = 275, y = 410) 
model_summary.place(x=30, y=458)
summary.place(x = 200, y = 455)
pipeline_select.place(x=30, y=505)
pipeline_dropdown.place(x=250, y=505)
resource.place(x=30, y=680)
resource_dwnld.place(x=250, y=680)
resource_sim.place(x=350, y=680)
res_or.place(x=320, y=682)
code_gen.place(x=25, y=555)
R1.place(x= 50, y=580)
R2.place(x= 50, y=605)
generate_button.place(x=200, y=630)


top.mainloop()  