
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import pyscreenshot as ImageGrab
import time
import os
import cv2
import csv
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

window = tk.Tk()
window.title("project")
window.resizable(0, 0)

header = tk.Button(window, text="Handwritten Digit Recognization", font=("Algerian", 30))
header.place(x=30, y=50)

canvas1 = Canvas(window, width=490, height=240, bg="ivory")
canvas1.place(x=5, y=110)

l1 = tk.Label(canvas1, text="Folder", font=("Algerian", 20))
l1.place(x=5, y=5)

t1 = tk.Entry(canvas1, width=20, border=5)
t1.place(x=150, y=5)
s1 = t1.get()

def Screen_Capture():
    os.startfile("C:/Users/abhis/OneDrive/Desktop/Paint.lnk")
    s1 = t1.get()
    if not s1:
        messagebox.showerror("Error", "Please enter a valid directory name.")
        return
    
    images_folder = "C:/Users/abhis/OneDrive/Desktop/ML/Handwritten_recognization/Captured_images/" + s1 + "/"
    
    try:
        os.chdir("C:/Users/abhis/OneDrive/Desktop/ML/Handwritten_recognization")
        os.mkdir(images_folder)
    except FileExistsError:
        messagebox.showwarning("Warning", "Directory already exists. Capturing screen images will overwrite existing files.")
    
    time.sleep(10)
    for i in range(0, 4):
        time.sleep(8)
        im =ImageGrab.grab(bbox=(100,350,500,650))
        print("saved........",i)
        im.save(images_folder+str(i)+'.png')
        print("clear Screen now and redraw now.........")
    messagebox.showinfo("Result", "Capturing Screen is completed!!")


b1 = tk.Button(canvas1, text="1. Open paint and capture the screen", font=("Algerian", 15), bg="orange", fg="black", command=Screen_Capture)
b1.place(x=5, y=50)


def Live_Predict():
    model = tf.keras.models.load_model('handwritten.model')
    im =ImageGrab.grab(bbox=(100,510,500,800))
    im.save('Livephoto.png')
    print("saved........")
    load = Image.open("Livephoto.png").convert('L')
    load = load.resize((280,280))
    photo = ImageTk.PhotoImage(load)
    lbl = Label(canvas3, image=photo, width=280, height=280)
    lbl.image = photo
    lbl.place(x=2, y=2)
    
    try:
        img = cv2.imread('Livephoto.png')[:, :, 0]
        img = cv2.resize(img, (28, 28))
        img = np.invert(np.array([img]))
        predicted = model.predict(img)
        print(f"This image is probably a {np.argmax(predicted)}")
        a1 = tk.Label(canvas3, text="Prediction = ", font=("Algerian", 20))
        a1.place(x=5, y=300)
        b1 = tk.Label(canvas3, text=np.argmax(predicted), font=("Algerian", 20))
        b1.place(x=200, y=300)
    except Exception as e:
        print(f"Error : {e}")

b2 = tk.Button(canvas1, text="2. Live Prediction", font=("Algerian", 15), bg="pink", fg="blue", command=Live_Predict)
b2.place(x=5, y=100)


def Train_Model():
    model = models.Sequential()
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=3, batch_size=64, validation_data=(test_images, test_labels))
    model.save('handwritten.model')
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')

b3 = tk.Button(canvas1, text="3. train the model and calculate accuracy", font=("Algerian", 15), bg="green", fg="white", command=Train_Model)
b3.place(x=5, y=150)

def Predict_Image():
    model = tf.keras.models.load_model('handwritten.model')
    image_number = 0
    s1 = t1.get()
    while os.path.isfile(f"Captured_images/{s1}/{image_number}.png"):
        try:
            img = cv2.imread(f"Captured_images/{s1}/{image_number}.png")[:, :, 0]
            img = cv2.resize(img, (28, 28))
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This image is probably a {np.argmax(prediction)}")
            a1 = tk.Label(canvas3, text="Prediction = ", font=("Algerian", 20))
            a1.place(x=5, y=300)
            b1 = tk.Label(canvas3, text=np.argmax(prediction), font=("Algerian", 20))
            b1.place(x=200, y=300)
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except Exception as e:
            print(f"Error : {e}")
        finally:
            image_number += 1

b3 = tk.Button(canvas1, text="4. Predict the Images", font=("Algerian", 15), bg="white", fg="red", command=Predict_Image)
b3.place(x=5, y=200)

canvas2 = Canvas(window, width=490, height=240, bg="black")
canvas2.place(x=5, y=370)

def activate_paint(e) :
    global lastx, lasty
    canvas2.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y

def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    canvas2.create_line((lastx, lasty, x, y), width=5, fill='white', capstyle='round', smooth=True)
    lastx, lasty = x, y

canvas2.bind('<1>', activate_paint)

def clear():
    canvas2.delete('all')

btn = tk.Button(canvas2, text='clear', fg='white', bg='green', command=clear)
btn.place(x=2, y=2)

canvas3 = Canvas(window, width=390, height=500, bg="green")
canvas3.place(x=500, y=110)

window.geometry("800x650")
window.mainloop()

