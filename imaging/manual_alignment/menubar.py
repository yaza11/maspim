import os
import tkinter as tk
from tkinter import filedialog, simpledialog

from imaging.manual_alignment.objects import LoadedImage


class MenuBar:

    def __init__(self, app):
        self.app = app

        # create a menubar
        self.menubar = tk.Menu(self.app)
        self.app.config(menu=self.menubar)

        # Add file menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Save workspace", command=self.app.save)
        self.file_menu.add_command(label="Load workspace", command=self.app.load)
        # Add 'Open' to the file menu
        self.file_menu.add_command(label="Add images", command=self.add_images)
        # Add 'Exit' to the file menu
        self.file_menu.add_command(label="Quit", command=self.quit)

        # Add 'View' menu
        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=self.view_menu)
        # add 'Hide Teaching Points View' to the view menu
        self.view_menu.add_command(label="Toggle TP View", command=self.tg_tp_view)

        # Add 'Calc' menu
        self.calc_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Calc", menu=self.calc_menu)
        # Add 'Cm/Px' to the calc menu
        self.calc_menu.add_command(label="cm/Px", command=self.calc_cm_per_px)

    def add_images(self):
        """Load the images from the file paths"""
        file_paths = filedialog.askopenfilenames()
        for file_path in file_paths:
            for k, v in self.app.items.items():
                if v.path == file_path:
                    raise ValueError(f"{file_path} has already been loaded")
            loaded_image = LoadedImage.from_path(file_path)
            loaded_image.create_im_on_canvas(self.app)
            self.app.items[loaded_image.tag] = loaded_image

    def quit(self):
        self.app.quit()

    def tg_tp_view(self):
        """Toggle the visibility of the teaching points view"""
        if self.app.tree_visible:
            self.app.tree_frame.pack_forget()
            self.app.tree_visible = False
        else:
            self.app.tree_frame.pack(side=tk.RIGHT, fill=tk.Y)
            self.app.tree_visible = True

    def calc_cm_per_px(self):
        # get the two vertical scale lines
        scale_lines = self.app.canvas.find_withtag("scale_line")
        if len(scale_lines) < 2:
            raise ValueError("You need to draw two vertical lines to calculate the scale")
        elif len(scale_lines) > 2:
            raise ValueError("You have drawn more than two vertical lines")
        else:
            # calculate the distance between the two scale lines
            pixel_distance = abs(
                self.app.canvas.coords(scale_lines[1])[0] - self.app.canvas.coords(scale_lines[0])[0])
            # calculate the distance in real world
            real_distance = simpledialog.askfloat("Real Distance", "Real Distance (cm):")
            # calculate the scale
            self.app.cm_per_pixel = real_distance / pixel_distance
            # create a text on the canvas to display the scale
            self.app.canvas.create_text(10, 10, text=f"1cm = {pixel_distance / real_distance} pixel", anchor="nw",
                                        tags="scale")
