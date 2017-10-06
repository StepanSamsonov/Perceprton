from tkinter.filedialog import *
from perceptron import Perceptron
from matplotlib import pyplot as plt
import numpy as np


def open_from_main_window():
    open = askopenfilename()
    global net
    if open != '':
        net = Perceptron(link=open)
        main_window.destroy()
        work_with_net()


def open_from_work_with_net():
    open = askopenfilename()
    global net
    if open != '':
        net = Perceptron(link=open)
        workwindow.destroy()
        work_with_net()


def open_from_new_open():
    open = askopenfilename()
    global net
    if open != '':
        net = Perceptron(link=open)
        create_window.destroy()
        work_with_net()


def open_new_from_work():
    workwindow.destroy()
    open_new()


def open_new_from_main():
    main_window.destroy()
    open_new()


def open_save():
    save = asksaveasfilename()
    net.save(save)


def open_new():
    def gotowork():
        global net
        try:
            net = Perceptron(int(inputnet.get()), int(middlenet.get()), int(outnet.get()), float(speed.get()))
        except BaseException:
            return None
        create_window.destroy()
        work_with_net()

    global create_window
    create_window = Tk()
    create_window.title("Create new network")
    create_window.geometry("400x300")
    create_window.title("Neural network")
    create_window.resizable(0, 0)
    menu = Menu(create_window)
    create_window.config(menu=menu)
    fm = Menu(menu, tearoff=False)
    menu.add_cascade(label="File", menu=fm)
    fm.add_command(label="Open", command=open_from_new_open)

    inputnet = Entry(width=10)
    middlenet = Entry(width=10)
    outnet = Entry(width=10)
    speed = Entry(width=10)
    speed.insert(END, '1')
    create = Button(create_window, text="Create", command=gotowork, width=5, height=2)
    neurons = Label(create_window, text="Enter the number of neurons in each layer")
    front = Label(create_window, text="Front")
    middle = Label(create_window, text="Middle")
    back = Label(create_window, text="Back")
    speedstudy = Label(create_window, text="Enter the speed of networks learning")
    inputnet.place(x=65, y=100)
    middlenet.place(x=165, y=100)
    outnet.place(x=265, y=100)
    speed.place(x=165, y=185)
    create.place(x=175, y=225)
    neurons.place(x=80, y=40)
    front.place(x=80, y=75)
    middle.place(x=180, y=75)
    back.place(x=280, y=75)
    speedstudy.place(x=100, y=155)
    create_window.mainloop()


def work_with_net():
    def studynet():
        open = askopenfilename()
        try:
            epoch = int(epoch_enrty.get())
            error = float(error_entry.get())
        except BaseException:
            return None
        if open != '':
            try:
                wrong = net.train(link=open, epoch=epoch, error=error)
            except BaseException:
                return None
            plt.plot(np.arange(len(wrong)), wrong)
            plt.grid()
            plt.xlabel('Epochs')
            plt.ylabel('Error')
            plt.show()

    completed = None
    def changenet():
        nonlocal completed
        try:
            completed.place_forget()
        except BaseException:
            pass
        try:
            if (layer.get()):
                net._weight_input[int(inputlayer.get())][int(outputlayer.get())] = float(value.get())
            else:
                net._weight_output[int(inputlayer.get())][int(outputlayer.get())] = float(value.get())
        except BaseException:
            return None
        completed = Label(workwindow, text="Completed!")
        completed.place(x=385, y=235)

    def work_net():
        linkopen = askopenfilename()
        if linkopen != '':
            answertest = net.work(link=linkopen)
        else:
            return None
        finalline = ""
        for elem in answertest:
            finalline = finalline + elem + '\n'
        endtext = open(linkopen, 'a')
        endtext.write(finalline)
        endtext.close()

    def view():
        if layer.get():
            inputwindow = Toplevel(workwindow)
            inputwindow.title("See connection on the input level")
            tx = Text(inputwindow, width=70, height=20)
            scr = Scrollbar(inputwindow, command=tx.yview)
            tx.configure(yscrollcommand=scr.set)
            inputneurons = ""
            for i in range(len(net._weight_input)):
                inputneurons += ("Neurons which connected with the next layer neuron with index " + str(i) + "\n")
                for j in range(len(net._weight_input[i])):
                    inputneurons += (str(j) + ": " + str(net._weight_input[i][j]) + "\n")
            tx.insert(1.0, inputneurons)
            tx.grid(row=0, column=0)
            scr.grid(row=0, column=1)
            tx.configure(state=DISABLED)
            inputwindow.mainloop()
        else:
            outputwindow = Toplevel(workwindow)
            outputwindow.title("See connection on the output level")
            tx = Text(outputwindow, width=70, height=20)
            scr = Scrollbar(outputwindow, command=tx.yview)
            tx.configure(yscrollcommand=scr.set)
            outputneurons = ""
            for i in range(len(net._weight_output)):
                outputneurons += ("Neurons which connected with the next layer neuron with index " + str(i) + "\n")
                for j in range(len(net._weight_output[i])):
                    outputneurons += (str(j) + ": " + str(net._weight_output[i][j]) + "\n")
            tx.insert(1.0, outputneurons)
            tx.grid(row=0, column=0)
            scr.grid(row=0, column=1)
            tx.configure(state=DISABLED)
            outputwindow.mainloop()

    global workwindow
    workwindow = Tk()
    workwindow.title("Neural network")
    workwindow.geometry("500x400")
    workwindow.resizable(0, 0)
    menu = Menu(workwindow)
    workwindow.config(menu=menu)
    fm = Menu(menu, tearoff=False)
    menu.add_cascade(label="File", menu=fm)
    fm.add_command(label="Open", command=open_from_work_with_net)
    fm.add_command(label="New", command=open_new_from_work)
    fm.add_command(label="Save", command=open_save)

    study = Button(workwindow, text="Study", command=studynet, width=10, height=5)
    change = Button(workwindow, text="Change", command=changenet, width=7, height=2)
    work = Button(workwindow, text="Work", command=work_net, width=10, height=5)
    inputlayer = Entry()
    outputlayer = Entry()
    value = Entry()
    layer = IntVar()
    frontlayer = Radiobutton(workwindow, text="Front layer", variable=layer, value=True)
    backlayer = Radiobutton(workwindow, text="Back layer", variable=layer, value=False)
    textvalue = Label(workwindow, text="Enter the value which you want to change:")
    textneuron = Label(workwindow, text="Enter the number of neurons to change the value:")
    texthand = Label(workwindow, text="Select the layer and number of neurons to change the weight manually")
    textstudy = Label(workwindow, text="Click the 'Study' for training and 'Work' for working")
    aboutnet = "The network has: " + str(net._size_of_input) + ", " + str(net._size_of_middle) + ", " + str(
        net._size_of_output) + " neuros in layers;"
    aboutnet = aboutnet + " Speed training is " + str(net._coefficient)
    textaboutnet = Label(workwindow, text=aboutnet)

    epoch_enrty = Entry(width=10)
    epoch_enrty.insert(END, '100')
    error_entry = Entry(width=10)
    error_entry.insert(END, '0.01')
    Label(workwindow, text='Epoch number:').place(x=197, y=80)
    epoch_enrty.place(x=210, y=100)
    Label(workwindow, text='Error value:').place(x=210, y=120)
    error_entry.place(x=210, y=140)

    frontlayer.place(x=100, y=220)
    backlayer.place(x=100, y=250)
    inputlayer.place(x=100, y=300)
    outputlayer.place(x=300, y=300)
    value.place(x=200, y=350)
    study.place(x=100, y=80)
    work.place(x=305, y=80)
    change.place(x=300, y=225)
    textvalue.place(x=150, y=325)
    textneuron.place(x=130, y=275)
    texthand.place(x=65, y=200)
    textstudy.place(x=110, y=50)
    textaboutnet.place(x=10, y=15)
    Button(workwindow, text="View", command=view).place(x=50, y=230)
    workwindow.mainloop()


main_window = Tk()
main_window.title("Neural network")
main_window.geometry("400x300")
main_window.resizable(0, 0)
menu = Menu(main_window)
main_window.config(menu=menu)

fm = Menu(menu, tearoff=False)
menu.add_cascade(label="File", menu=fm)
fm.add_command(label="New", command=open_new_from_main)
fm.add_command(label="Open", command=open_from_main_window)
main_label = Label(main_window,
                  text="Welcome to the program which work with artificial neural networks.\n\nAll networks have one hidden layer.\n\nYou can create new networks and save them, download old,\n educate and manually change values of synaptic weights.\n\nGood luck!")
main_label.place(x=18, y=20)
main_window.mainloop()
