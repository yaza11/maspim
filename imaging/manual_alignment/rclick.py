"""implement right click functionality for the application"""
import logging
import tkinter as tk
from tkinter import simpledialog

import numpy as np


class RightClickMenu:
    """ this is the superclass for the right-click menu"""

    def __init__(self, app):
        self.app = app
        self.menu = tk.Menu(self.app, tearoff=0)
        self._add_menu_item()
        self.clicked_item = None
        self.clicked_event = None

    def _add_menu_item(self):
        pass

    def show_menu(self, event, item=None):
        pass


class RightClickOnLine(RightClickMenu):
    """this is the right-click menu for the line"""

    def _add_menu_item(self):
        self.menu.add_command(label="Set Scale Line",
                              command=lambda: self.set_scale_line(self.clicked_item))
        self.menu.add_command(label="Set Sediment Start",
                              command=lambda: self.set_sediment_start(self.clicked_item))
        self.menu.add_command(label="Delete",
                              command=lambda: self.delete_line(self.clicked_item))

    def show_menu(self, event, item=None):
        # update the item to be right-clicked
        self.clicked_item = item
        self.clicked_event = event
        # show the menu
        self.menu.post(event.x_root, event.y_root)

    def set_scale_line(self, item):
        """tag the scale line with 'scale_line'"""
        # change the color of the scale line to blue
        self.app.canvas.itemconfig(item, fill="blue")
        self.app.items[item].label = "scale_line"
        self.app.scale_line.append(item)

    def set_sediment_start(self, item):
        """tag the vertical line with 'sediment_start'"""
        self.app.canvas.itemconfig(item, fill="green")
        # label the line with 'sediment_start'
        self.app.items[item].label = "sediment_start_line"
        self.app.sediment_start = item

    def delete_line(self, item):
        """delete the vertical line"""
        self.app.items[item].rm(self.app)


class RightClickOnImage(RightClickMenu):
    """this is the right-click menu for the image"""

    def _add_menu_item(self):
        self.menu.add_command(label="Rotate",
                              command=lambda: self.rotate_image(self.clicked_item))

        self.menu.add_command(label="Add Label",
                              command=lambda: self.add_label(self.clicked_item))

        chg_size = tk.Menu(self.menu, tearoff=0)
        chg_size.add_command(label="x0.5", command=lambda: self.enlarge_image(self.clicked_item, 0.5))
        chg_size.add_command(label="x1.5", command=lambda: self.enlarge_image(self.clicked_item, 1.5))
        chg_size.add_command(label="x2", command=lambda: self.enlarge_image(self.clicked_item, 2))
        self.menu.add_cascade(label="Resize", menu=chg_size)

        self.menu.add_command(label="Unlock",
                              command=lambda: self.unlock_image(self.clicked_item))

        self.menu.add_command(label="Lock",
                              command=lambda: self.lock_image(self.clicked_item))
        self.menu.add_command(label="Delete",
                              command=lambda: self.app.items[self.clicked_item].rm(self.app))

    def add_label(self, item):
        """ add label to the item, unlike tag, label is not unique and can be changed and easy to understand"""
        label = simpledialog.askstring("Input", "Enter the label")
        if label:
            self.app.items[item].label = label
            # update the label column in the tree view
            self.app.tree.set(self.app.items[item].tree_master, "label", label)

    def show_menu(self, event, item=None):
        # update the item to be right-clicked
        self.clicked_item = item
        self.clicked_event = event

        if self.app.items[item].locked:
            self.menu.entryconfig("Unlock", state="normal")
            self.menu.entryconfig("Lock", state="disabled")
        else:
            self.menu.entryconfig("Unlock", state="disabled")
            self.menu.entryconfig("Lock", state="normal")
        # show the menu
        self.menu.post(event.x_root, event.y_root)

    def rotate_image(self, item):
        """rotate the image by 90 degrees"""
        # rotate the image by 90 degrees
        self.app.items[item].rotate()
        self.app.canvas.itemconfig(item, image=self.app.items[item].tk_img)

    def enlarge_image(self, item, scale_factor):
        """enlarge/shrink the image"""
        self.app.items[item].enlarge(scale_factor)
        self.app.canvas.itemconfig(item, image=self.app.items[item].tk_img)

    def lock_image(self, item):
        """lock the image"""
        self.app.canvas.tag_unbind(item, "<Button-1>")
        # on the left top corner, display 'locked'
        x1, y1, x2, y2 = self.app.canvas.bbox(item)
        # create a text on the canvas
        text = tk.Text(self.app.canvas, width=6, height=1)
        text.insert(tk.END, "Locked")
        text.config(state="disabled")
        self.app.canvas.create_window(x1, y1, window=text, tags=f"Locked{item}")
        self.app.items[item].lock()

    def unlock_image(self, item):
        """unlock the image"""
        self.app.canvas.tag_bind(item,
                                 "<Button-1>",
                                 lambda event: self.app.on_drag_start(item, event))
        self.app.canvas.delete(f"Locked{item}")
        self.app.items[item].unlock()


class RightClickOnTeachingPoint(RightClickMenu):
    """this is the right-click menu for the teaching point"""

    def _add_menu_item(self):
        self.menu.add_command(label="Delete",
                              command=lambda: self.delete_teaching_point(self.clicked_event, self.clicked_item))

    def show_menu(self, event, item=None):
        logging.debug(f"Teaching point {item} is right-clicked, the coordinates are {self.app.canvas.coords(item)}")
        # update the item to be right-clicked
        self.clicked_item = item
        self.clicked_event = event
        # show the menu
        self.menu.post(event.x_root, event.y_root)

    def delete_teaching_point(self, event, item):
        """delete the teaching point"""
        # find the clicked image
        clicked_image = self.app.find_clicked_image(event)
        self.app.items[clicked_image.tag].teaching_points.pop(item)
        self.app.canvas.delete(item)
        logging.debug(f"Teaching point {item} is deleted")
