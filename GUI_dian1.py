"""
This file contains functions for solving the Lazor game and giving the solution
board from a given board game information. Besides, a class specialized for
different types of block are defined, whose methods are used to judge whether a
lazor would hit the block and determine the next state (position and velocity
direction) of that lazor if hitting.

The functions include lazor_solver, _get_path, _next_block_pos,
and _is_out_of_board. The classes include block and its three subclasses:
reflect_block, refract_block, and refract_block, with two methods: is_Hit
and next_state.

Please do not change resolution and scale of your laptop when you use this code. If you change it , please restart your editor.
"""
# import library
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import ImageGrab
from tensorflow.keras.models import load_model
import os
from screeninfo import get_monitors


class digit_solver:
    def __init__(self, root, model):
        # create the root window and set some basic features
        self.root = root
        self.root.title("Handwritten Digit Recognition Device")
        rtWidth, rtHeight = 2200, 1650
        # rtWidth = int(rtWidth/2)
        # rtHeight = int(rtHeight/2)
        self.root.geometry(str(rtWidth)+'x'+str(rtHeight))
        self.root.resizable(True, True)

        # configure the grid
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=19)
        self.root.rowconfigure(0, weight=6)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=1)
        self.root.rowconfigure(4, weight=6)

        # model
        self.model = model

        # Initialization
        self.lastx, self.lasty = None, None
        # variables for pointer and Eraser
        self.pointer = "black"
        self.erase = "white"

        # image file storage
        global image_number
        image_number = 0
        # specify the folder name for storage of result img and table
        self.folder_name = "Results"
        # join the current working directory with the folder name
        self.folder_path = os.path.join(os.getcwd(), self.folder_name)
        # delete all the files under the specified folder
        self.__clear_folder()

        # define widgets for the tk GUI
        # create a canvas for drawing
        self.cv = tk.Canvas(root,
                            bg='white',
                            cursor="dotbox")
        self.cv.grid(row=0, column=1, sticky='nsew', rowspan=5)
        self.cv.bind('<Button-1>', self.activate_event)

        # create the erase button
        self.savebutton = tk.Button(text='Erase',
                                    bd=5,
                                    font=("Arial", 24, "bold"),
                                    command=self.eraser)
        self.savebutton.grid(row=1, column=0, sticky='nsew')

        # create the save button
        self.savebutton = tk.Button(text='Recognize Digit(s)',
                                    bd=5,
                                    font=("Arial", 24, "bold"),
                                    command=self.Recognize_Digit)
        self.savebutton.grid(row=2, column=0, sticky='nsew')

        # create the clear button
        self.clearbutton = tk.Button(text='Clear All',
                                     bd=5,
                                     font=("Arial", 24, "bold"),
                                     command=self.clear_canvas)
        self.clearbutton.grid(row=3, column=0, sticky='nsew')

        # create a color pannel (a LabelFrame of Buttons), that is we can pick a color for drawing from color pannel
        self.pick_color = tk.LabelFrame(self.root,
                                        bd=5,
                                        relief=tk.RIDGE,
                                        bg="white")
        self.pick_color.grid(row=0, column=0, sticky='nsew')

        self.pick_color.rowconfigure(0, weight=1)
        self.pick_color.rowconfigure(1, weight=1)
        self.pick_color.rowconfigure(2, weight=1)
        self.pick_color.rowconfigure(3, weight=1)
        self.pick_color.rowconfigure(4, weight=1)
        self.pick_color.rowconfigure(5, weight=1)
        self.pick_color.rowconfigure(6, weight=1)
        self.pick_color.columnconfigure(0, weight=1)
        self.pick_color.columnconfigure(1, weight=1)
        self.label = tk.Label(self.pick_color,
                              bd=2,
                              text="Color Options",
                              font=("Arial", 24, "bold"))
        self.label.grid(row=0, column=0, sticky='nsew', columnspan=2)
        colors = ['black', 'red', 'green', 'brown', 'skyblue', '#e9c46a',
                  'indigo', 'purple', 'blue', '#856ff8', '#e76f51', '#2a9d8f']
        i, j = 1, 0
        for color in colors:
            tk.Button(self.pick_color,
                      bg=color,
                      bd=2,
                      relief=tk.RIDGE,
                      command=lambda col=color: self.select_color(col)).\
                grid(row=i, column=j, sticky='nsew')
            i += 1
            if i == 7:
                i, j = 1, 1

        # create a scale bar (a LabelFrame of a Scale) for adjustment of pointer and eraser's size
        self.pointer_frame = tk.LabelFrame(self.root,
                                           bd=5,
                                           bg='white',
                                           relief=tk.RIDGE)
        self.pointer_frame.grid(row=4, column=0, sticky='nsew')

        self.pointer_frame.rowconfigure(0, weight=1)
        self.pointer_frame.rowconfigure(1, weight=9)
        self.pointer_frame.columnconfigure(0, weight=1)
        self.label2 = tk.Label(self.pointer_frame,
                               bd=2,
                               text="Size Bar",
                               font=("Arial", 24, "bold"))
        self.label2.grid(row=0, sticky='nsew')
        self.pointer_size = tk.Scale(self.pointer_frame,
                                     bd=6,
                                     orient=tk.VERTICAL,
                                     from_=20,
                                     to=0)
        self.pointer_size.set(12)
        self.pointer_size.grid(row=1, sticky='nsew')

    def __clear_folder(self):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        else:
            # Get a list of all files in the folder
            files = os.listdir(self.folder_name)
            # Iterate over the files and delete each one
            for file_name in files:
                file_path = os.path.join(self.folder_name, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        # print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    def __delete_file(self, filename):
        # Construct the full path to the file
        file_path = os.path.join(self.folder_path, filename)
        try:
            # Attempt to remove the file
            os.remove(file_path)
            # print(f"File '{filename}' in directory '{file_path}' has been deleted.")
        except FileNotFoundError:
            print(f"File '{filename}' not found in directory '{file_path}'.")
        except Exception as e:
            print(f"An error occurred while deleting the file: {e}")

    def clear_canvas(self):
        '''
        clear all mark in canvas.

        **Returns**

            None
        '''
        self.cv.delete('all')

    def activate_event(self, event):
        '''
        activate event and init params when pressing mouse left in canvas.

        **Parameters**

            event:
                A class object that describe the event.

        **Returns**

            None
        '''
        self.cv.bind('<B1-Motion>', self.drawing)
        self.lastx, self.lasty = event.x, event.y

    def drawing(self, event):
        '''
        drawing while keep mouse left pressing and moving in canvas.

        **Parameters**

            event:
                A class object that describe the event.

        **Returns**

            None
        '''
        x, y = event.x, event.y
        # live drawing a smooth curve in canvas
        self.cv.create_line((self.lastx, self.lasty, x, y),
                            width=self.pointer_size.get(),
                            fill=self.pointer,
                            capstyle=tk.ROUND,
                            smooth=True,
                            splinesteps=12)
        self.lastx, self.lasty = x, y

    def select_color(self, col):
        self.pointer = col

    def eraser(self):
        self.pointer = self.erase

    def Recognize_Digit(self):
        '''
        separate and recognize digit(s) draw in canvas.

        **Returns**

            None
        '''
        # name setting for saved canvas screenshot
        filename = f'Original_digits.png'
        # use another name for cv to avoid mixing with cv2
        my_cv = self.cv

        # get the coordiantes of the canvas
        x = self.root.winfo_rootx() + my_cv.winfo_x()
        y = self.root.winfo_rooty() + my_cv.winfo_y()
        x1 = x + my_cv.winfo_width()
        y1 = y + my_cv.winfo_height()

        # extract the horizontal/verticle resolution of the PC screen
        pc_resol_width = get_monitors()[0].width
        pc_resol_height = get_monitors()[0].height

        # trim the part of root window outside your PC monitor
        x = digit_solver.nneg(x)
        y = digit_solver.nneg(y)
        x1 = min(x1, pc_resol_width)
        y1 = min(y1, pc_resol_height)

        # extract the horizontal resolution and pixel number/height of the PC
        # screen to calculate the PC scale
        # pc_resol_width = get_monitors()[0].width
        # pc_pixel_width = self.root.winfo_screenwidth()
        # pc_scale = int(pc_resol_width/pc_pixel_width)
        # print(pc_scale)

        # # scale the coordiantes according to pc scale
        # x *= pc_scale
        # y *= pc_scale
        # x1 *= pc_scale
        # y1 *= pc_scale

        # get the screenshot for canvas and save it in a png format
        # buffer to avoid screenshot the region outside the canvas (esp, the slight separation of two edges of windows).
        bf = 5
        ImageGrab.grab().crop((x+bf, y+bf, x1-bf, y1-bf)).save(self.folder_path + '/' + filename)

        image = cv2.imread(self.folder_path + '/' + filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)

        # black-white reverse to separate each digit in the image
        ret, th = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        # print(ret2)
        # cv2.imshow('th image', th)

        contours = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]

        # variable for storage of digit recognization result
        rgcrlt, ind = [[], []], 0

        for cnt in contours:
            ind += 1

            x, y, w, h = cv2.boundingRect(cnt)
            # make a rectangle box around each curve
            cv2.rectangle(image,
                          (x, y),
                          (x+w, y+h),
                          (0, 0, 255),
                          2)

            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = th[y:y + h, x:x + w]

            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit, (18, 18))

            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit,
                                  ((5, 5), (5, 5)),
                                  "constant",
                                  constant_values=0)

            digit = padded_digit.reshape(1, 28, 28, 1)
            digit = digit / 255.0

            pred = self.model.predict([digit])[0]
            final_pred = np.argmax(pred)

            # data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'
            data = '#' + str(ind) + '  ' + 'Prediction: ' + str(final_pred)

            # store the digit recognition result into rgcrlt
            rgcrlt[0].append(str(final_pred))
            rgcrlt[1].append(str(int(max(pred) * 100)))

            # show the digit recognition result in the figure
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 0, 255)  # Note: BGR
            thickness = 1
            cv2.putText(image,
                        data,
                        (x, y - 5),
                        font,
                        fontScale,
                        color,
                        thickness)

        # save the final image with recorgnized digits
        filename1 = f'Recognized_digits.png'
        imagedir = self.folder_path + '/' + filename1
        cv2.imwrite(imagedir, image)

        crtfld = self.folder_path

        # display the result summary of digit(s) recognition in a toplevel window
        top1 = tk.Toplevel()
        result_panel(top1, imagedir, rgcrlt, crtfld)
        top1.mainloop()

        # delete the files for and from recognition
        self.__delete_file(filename)
        self.__delete_file(filename1)

    @staticmethod
    def nneg(x): return (x > 0) * x


class result_panel:
    def __init__(self, top1, imagedir, rgcrlt, crtfld):
        # create the root window and set some basic features
        self.top1 = top1
        self.top1.title("Summary of Digit Recognition")
        tp1Width, tp1Height = 2200, 1650
        # tp1Width = int(tp1Width/2)
        # tp1Height = int(tp1Height/2)
        self.top1.geometry(str(tp1Width)+'x'+str(tp1Height))
        self.top1.resizable(True, True)

        # dir of temporary unsaved result image
        self.imagedir = imagedir
        # list of recognition result
        self.rgcrlt = rgcrlt
        # current number of saved result
        # current folder path
        self.crtfld = crtfld

        global isSave
        isSave = False

        # configure the grid
        self.top1.columnconfigure(0, weight=4)
        self.top1.columnconfigure(1, weight=1)
        self.top1.rowconfigure(0, weight=1)
        self.top1.rowconfigure(1, weight=8)

        # label wedgit for figure title
        self.btnfg = tk.Button(self.top1,
                               bd=5,
                               text="Result Figure with Recognized Digit(s) (Click to Save)",
                               font=("Arial", 24, "bold"),
                               relief=tk.RIDGE,
                               command=self.savefigure)
        self.btnfg.grid(row=0, column=0, sticky='nsew')

        # label wedgit for showing figure
        self.image = tk.PhotoImage(file=self.imagedir)
        self.rgcimg = tk.Label(self.top1, bd=2, image=self.image)
        self.rgcimg.image = self.image
        self.rgcimg.grid(row=1, column=0, sticky='nsew')

        # label wedgit for table title
        self.btntb = tk.Button(self.top1,
                               bd=5,
                               text="Result Table (Click to Save)",
                               font=("Arial", 24, "bold"),
                               relief=tk.RIDGE,
                               command=self.savetable)
        self.btntb.grid(row=0, column=1, sticky='nsew')

        # treeview for showing the result data of recognized digit
        s = ttk.Style()
        s.theme_use('clam')
        # Add the rowheight
        s.configure('Treeview', rowheight=30, foreground='black', font=("Arial", 16))
        s.configure("Treeview.Heading", foreground='black', font=("Arial", 16, "bold"))

        self.rlttable = ttk.Treeview(self.top1,
                                     columns=('#', 'regdig', '%'),
                                     show='headings')

        self.tablehead = ['#', 'Predicted Digit', 'Probability (%)']
        self.rlttable.heading('#', text=self.tablehead[0])
        self.rlttable.column('#', anchor=tk.CENTER, stretch=tk.NO, width=80)
        self.rlttable.heading('regdig', text=self.tablehead[1])
        self.rlttable.column('regdig', anchor=tk.CENTER)
        self.rlttable.heading('%', text=self.tablehead[2])
        self.rlttable.column('%', anchor=tk.CENTER)
        self.rlttable.grid(row=1, column=1, sticky='nsew')
        self.__rlt2table()

    def __rlt2table(self):
        for i in range(len(self.rgcrlt[0])):
            i_rev = len(self.rgcrlt[0])-1-i
            data = (str(i_rev+1), self.rgcrlt[0][i_rev], self.rgcrlt[1][i_rev])
            self.rlttable.insert(parent='', index=0, values=data)

    def savefigure(self):
        global image_number, isSave
        if not isSave:
            image_number += 1
        isSave = True
        filename = f'Result_Figure_{image_number}.png'
        figuredir = self.crtfld + '/' + filename
        self.image.write(figuredir)

        tk.messagebox.showinfo("Save to png file", "Figure was saved!")

    def savetable(self):
        global image_number, isSave
        if not isSave:
            image_number += 1
        isSave = True
        filename = f'Result_Table_{image_number}.txt'
        tabledir = self.crtfld + '/' + filename
        with open(tabledir, "w") as file:
            file.write('\t\t'.join(self.tablehead)+'\n')
            for i in range(len(self.rgcrlt[0])):
                file.write(str(i+1)+'\t\t'+self.rgcrlt[0][i]+'\t\t\t\t\t'+self.rgcrlt[1][i]+'\n')

            tk.messagebox.showinfo("Save to txt file", "Table was saved!")


if __name__ == '__main__':
    # load the CNN model for handwritten digit recognition
    model = load_model(r'C:\Users\dianl\Desktop\fall 2023\software\final project\keras_mnist.h5')
    print("CNN model is loaded successfully. Please go to the Tk window...")
    print("------------------------------------------------------------------")
    print("Please make sure when hitting Recognize Digit(s) button, the canvas wedgit in the root Tk window is not hidden by any object/pattern in your PC monitor window. Maximizing the Tk root window is recommanded!")
    print("------------------------------------------------------------------")

    # create the root window and set some basic features
    root = tk.Tk()

    # class the digit solver class
    app = digit_solver(root, model)

    # execute GUI
    root.mainloop()
