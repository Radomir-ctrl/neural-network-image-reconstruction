import tkinter as tk
import math
from pathlib import Path
from PIL import Image
from random import uniform

window = tk.Tk()
window.title("Generation")
canvas = tk.Canvas(window, width=32*2*5, height=32*5, bg='#000', highlightthickness=0)
canvas.pack()

image = Image.open(Path(__file__).resolve().parent / "ex3.png")
width = 32
height = 32
pixel = 5

def rgb(r, g, b):
    r = math.floor(min(max(r, 0), 255))
    g = math.floor(min(max(g, 0), 255))
    b = math.floor(min(max(b, 0), 255))
    return f'#{r:02x}{g:02x}{b:02x}'

def array1D(w):
    temp = []
    for i in range(w):
        temp.append(0)
    return temp

def array2D(w, h, random=False):
    temp = []
    for i in range(w):
        temp.append([])
        for j in range(h):
            if random:
                temp[i].append(uniform(-1,1))
            else:
                temp[i].append(0)
    return temp

def create_pixels2D(w, h, fill='#000', shift_x=0, shift_y=0):
    temp = array2D(w, h)
    for i in range(w):
        for j in range(h):
            start_x = i * pixel + shift_x * pixel
            start_y = j * pixel + shift_y * pixel
            end_x = start_x + pixel
            end_y = start_y + pixel
            temp[i][j] = canvas.create_rectangle(start_x, start_y, end_x, end_y, fill=fill, width=0)
            
    return temp

def lrelu(x):
    if x > 1:
        return 1 + 0.01 * (x - 1)
    elif x < 0:
        return 0.01 * x
    return x

def feed(x,y):

    # передача нормализованных координат во входные нейроны
    inputs_layer[0] = x
    inputs_layer[1] = y

    # рассчет первого скрытого слоя
    for i in range(len(hidden1_layer) - 1):
        sum = 0
        for j in range(len(inputs_layer)):
            sum += inputs_layer[j] * weight_inp_h1[j][i]
        hidden1_layer[i] = lrelu(sum)
    hidden1_layer[-1] = 1

    # рассчет второго скрытого слоя
    for i in range(len(hidden2_layer) - 1):
        sum = 0
        for j in range(len(hidden1_layer)):
            sum += hidden1_layer[j] * weight_h1_h2[j][i]
        hidden2_layer[i] = lrelu(sum)
    hidden2_layer[-1] = 1

    # рассчет выходного слоя
    for i in range(len(output_layer)):
        sum = 0
        for j in range(len(hidden2_layer)):
            sum += hidden2_layer[j] * weight_h2_out[j][i]
        output_layer[i] = lrelu(sum)

    # возврат списка для цветовых каналов
    return output_layer


original_image = create_pixels2D(width, height, 'blue')
nn_image = create_pixels2D(width, height, 'red', width)

# входные два нейрона для координат пикселя
inputs_layer = [0, 0, 1]
# первый скрытый слой: 32 нейрона и bias
hidden1_layer = array1D(32+1)
# второй скрытый слой: 32 нейрона и bias
hidden2_layer = array1D(32+1)
# выходные три нейрона для цветовых каналов
output_layer = [0,0,0]

lr = 0.03

# в начале мы указваем значение весов в виде случайных значений от -1 до 1
weight_inp_h1 = array2D(len(inputs_layer), len(hidden1_layer), random=True)
weight_h1_h2 = array2D(len(hidden1_layer), len(hidden2_layer), random=True)
weight_h2_out = array2D(len(hidden2_layer), len(output_layer), random=True)


for i in range(width):
    for j in range(height):
		    # извлекает цветовое значение пикселя из исходного изображения 
		    # по координатам (i, j).
        value = image.getpixel((i, j))
        color = rgb(value[0], value[1], value[2])
        canvas.itemconfig(original_image[i][j], fill=color)

def dlrelu(x):
    if x > 1:
        return 0.01
    elif x < 0:
        return 0.01
    return 1

def back(predicted, real):
    output_errors = array1D(3)
    output_errors[0] = predicted[0] - real[0]
    output_errors[1] = predicted[1] - real[1]
    output_errors[2] = predicted[2] - real[2]

    # дельты-весов выходного слоя
    output_deltas = array1D(3)
    output_deltas[0] = output_errors[0] * dlrelu(predicted[0])
    output_deltas[1] = output_errors[1] * dlrelu(predicted[1])
    output_deltas[2] = output_errors[2] * dlrelu(predicted[2])

    hidden2_errors = array1D(len(hidden2_layer))
    for i in range(3):
        for j in range(len(hidden2_layer) - 1):
            hidden2_errors[j] += output_deltas[i] * weight_h2_out[j][i]

    # дельты-весов скрытого слоя 2
    hidden2_deltas = array1D(len(hidden2_layer))
    for i in range(len(hidden2_layer) - 1):
        hidden2_deltas[i] = hidden2_errors[i] * dlrelu(hidden2_layer[i])

    # ошибки скрытого слоя 1
    hidden1_errors = array1D(len(hidden1_layer))
    for i in range(len(hidden2_layer) - 1):
        for j in range(len(hidden1_layer)):
            hidden1_errors[j] += hidden2_deltas[i] * weight_h1_h2[j][i]

    # дельты-весов скрытого слоя 1
    hidden1_deltas = array1D(len(hidden1_layer))
    for i in range(len(hidden1_layer) - 1):
        hidden1_deltas[i] = hidden1_errors[i] * dlrelu(hidden1_layer[i])


    for i in range(len(output_layer)):
        for j in range(len(hidden2_layer)):
            weight_h2_out[j][i] -= output_deltas[i] * hidden2_layer[j] * lr

    for i in range(len(hidden2_layer) - 1):
        for j in range(len(hidden1_layer)):
            weight_h1_h2[j][i] -= hidden2_deltas[i] * hidden1_layer[j] * lr

    for i in range(len(hidden1_layer) - 1):
        for j in range(len(inputs_layer)):
            weight_inp_h1[j][i] -= hidden1_deltas[i] * inputs_layer[j] * lr

def main():
    
    # визуализация
    for i in range(width):
        for j in range(height):
            x = i/(width/2)-1
            y = j/(height/2)-1
            out = feed(x,y)

            
            color = rgb(out[0] * 255, out[1] * 255, out[2] * 255)
            canvas.itemconfig(nn_image[i][j], fill=color)
    
    # обучение
    for i in range(width):
        for j in range(height):
            x = i/(width/2)-1
            y = j/(height/2)-1
            out = feed(x,y)
            pixel = image.getpixel((i, j))
            real = [pixel[0]/255,pixel[1]/255,pixel[2]/255]

            # обучаем сеть на основе этих значений
            back(out,real)
    window.after(10, main)

main()
window.mainloop()
