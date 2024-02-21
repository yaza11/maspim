import json
import os.path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog, simpledialog
from .objects import LoadedImage, TeachingPoint, VerticalLine
from .menubar import MenuBar
from .rclick import RightClickOnLine, RightClickOnImage, RightClickOnTeachingPoint


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.tree = None
        self.tree_frame = None
        self.canvas = None
        self.canvas_frame = None
        self.right_click_on_tp = None
        self.right_click_on_image = None
        self.right_click_on_line = None
        self._drag_offset_y = None
        self._drag_offset_x = None
        self.title('Transformer')
        self.items = {}
        self.create_canvas()

        self.scale_line = []
        self.sediment_start_line = None

        self.create_tp_tree()

        self.cm_per_pixel = None
        # add menubar
        self.menu = MenuBar(self)

        self.resizing = False
        self.tree_visible = True

        self.create_right_click_op()

    def create_right_click_op(self):
        self.right_click_on_line = RightClickOnLine(self)
        self.right_click_on_image = RightClickOnImage(self)
        self.right_click_on_tp = RightClickOnTeachingPoint(self)

    def create_canvas(self):
        canvas_width = 1000
        canvas_height = 1000
        # Create a frame for the canvas and scrollbars
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, width=canvas_width, height=canvas_height, bg='white',
                                scrollregion=(0, 0, 5000, 5000))
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Create horizontal and vertical scrollbars
        h_scroll = tk.Scrollbar(self.canvas, orient='horizontal', command=self.canvas.xview)
        h_scroll.pack(side=tk.BOTTOM, fill='x')
        v_scroll = tk.Scrollbar(self.canvas_frame, orient='vertical', command=self.canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill='y')
        # Configure the canvas to use the scrollbars
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

    def create_tp_tree(self):
        # create a treeview to display teaching points
        self.tree_frame = tk.Frame(self)
        self.tree_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree = ttk.Treeview(self.tree_frame,
                                 columns=('label', 'px', 'py', 'mx', 'my', 'd'),
                                 selectmode='browse')
        self.tree.heading('#0', text='img')
        self.tree.heading('label', text='label')
        self.tree.heading('px', text='pX')
        self.tree.heading('py', text='pY')
        self.tree.heading('mx', text='mX')
        self.tree.heading('my', text='mY')
        self.tree.heading('d', text='d')
        self.tree.column('#0', width=100)
        self.tree.column('label', width=50)
        self.tree.column('px', width=50)
        self.tree.column('py', width=50)
        self.tree.column('mx', width=50)
        self.tree.column('my', width=50)
        self.tree.column('d', width=50)
        self.tree.pack(side=tk.LEFT, fill=tk.Y)

    def on_drag_start(self, item, event):
        """Function to handle dragging"""
        # Get the coordinates of the image
        x1, y1, x2, y2 = self.canvas.bbox(item)

        # calculate the offset
        self._drag_offset_x = self.canvas.canvasx(event.x) - x1
        self._drag_offset_y = self.canvas.canvasy(event.y) - y1
        # Create a rectangle around the image
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", tags="rect")

        # Calculate the corner threshold as a percentage of the image's width and height
        corner_threshold_x = (x2 - x1) * 0.1
        corner_threshold_y = (y2 - y1) * 0.1

        # if the mouse is near the bottom-right corner, start resizing
        if (abs(x2 - self.canvas.canvasx(event.x)) < corner_threshold_x
                and abs(y2 - self.canvas.canvasy(event.y)) < corner_threshold_y):
            self.canvas.bind("<B1-Motion>", lambda e: self.on_resize(e, item))
            self.canvas.bind("<ButtonRelease-1>", lambda e: self.on_resize_stop(e))

        else:
            self.canvas.bind("<B1-Motion>", lambda e: self.on_drag_move(e, item))
            self.canvas.bind("<ButtonRelease-1>", lambda e: self.on_drag_stop(e, item))

    def on_drag_move(self, event, item):
        """move the item to the new position"""
        x, y = self.canvas.canvasx(event.x) - self._drag_offset_x, self.canvas.canvasy(event.y) - self._drag_offset_y
        self.canvas.coords(item, x, y)

    def on_drag_stop(self, event, item):
        """Stop dragging the image"""
        # record the new position of the image
        x, y = self.canvas.coords(item)
        self.items[item].position = (x, y)
        # remove the rectangle from the canvas
        self.canvas.delete('rect')
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

    def on_resize(self, event, item):
        """Resize the image based on mouse position"""
        x1, y1, x2, y2 = self.canvas.bbox(item)
        new_width = self.canvas.canvasx(event.x) - x1
        # Prevent the image from being too small
        new_width = max(new_width, 100)
        # get the original PIL image
        image = self.items[item].thumbnail
        # calculate the aspect ratio of the original image
        aspect_ratio = image.width / image.height
        # calculate the new height based on the aspect ratio
        new_height = new_width / aspect_ratio
        # resize the image by creating a new image with the new dimensions
        self.items[item].resize((new_width, new_height))
        # update the canvas item with the new image
        self.canvas.itemconfig(item, image=self.items[item].tk_img)

    def on_resize_stop(self, event):
        """Stop resizing the image"""
        self.resizing = False
        # remove the rectangle from the canvas
        self.canvas.delete('rect')
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

    def add_vertical_line(self, event):
        """draw a ruler on the canvas when ctrl-left-click is pressed, and calculate the scale"""
        vl = VerticalLine((self.canvas.canvasx(event.x), 0))
        self.items[vl.tag] = vl
        vl.create_on_canvas(self)

        # calculate the pixel distance between this ruler and sediment_start
        if self.sediment_start_line is not None and self.cm_per_pixel is not None:
            pixel_distance = self.canvas.coords(self.sediment_start_line)[0] - self.canvas.canvasx(event.x)
            real_distance = pixel_distance * self.cm_per_pixel
            vl.add_depth_text(self, f"{abs(real_distance):.2f}")

    def add_teaching_point(self, event):
        """mark and record the teaching point with a dot on shift-left-click"""
        # record the teaching point
        tp = TeachingPoint([self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)])
        tp.create_on_canvas(self)
        self.items[tp.tag] = tp
        # find the vertical lines that are close to the teaching point
        min_vl_idx = None
        for k, v in self.items.items():
            if v.type == "VerticalLine" and v.tag != "sediment_start_line" and v.tag != "scale_line":
                if min_vl_idx is None:
                    min_vl_idx = k
                else:
                    if abs(v.x - tp.x) < abs(self.items[min_vl_idx].x - tp.x):
                        min_vl_idx = k
        if min_vl_idx is not None:
            tp.depth = self.items[min_vl_idx].depth

        # find the image that contains the teaching point
        for k, v in self.items.items():
            if v.type == "LoadedImage":
                x1, y1, x2, y2 = self.canvas.bbox(v.tag)
                if x1 <= self.canvas.canvasx(event.x) <= x2 and y1 <= self.canvas.canvasy(event.y) <= y2:
                    # calculate the scale factor of the image on the canvas
                    original_width, original_height = v.orig_img.width, v.orig_img.height
                    scale_x = original_width / (x2 - x1)
                    scale_y = original_height / (y2 - y1)
                    # calculate the coordinates of the teaching point in the original image
                    image_x = (self.canvas.canvasx(event.x) - x1) * scale_x
                    image_y = (self.canvas.canvasy(event.y) - y1) * scale_y
                    # adjust the coordinates to the rotated image
                    rotation_angle = v.rotation
                    # calculate the rotation angles if there are multiple 90-degree rotations
                    rotation_angle = rotation_angle % 360
                    if rotation_angle == 90:
                        image_x, image_y = image_y, original_width - image_x
                    elif rotation_angle == 180:
                        image_x, image_y = original_width - image_x, original_height - image_y
                    elif rotation_angle == 270:
                        image_x, image_y = original_height - image_y, image_x

                    tp.linked_im = v.tag
                    tp.image_coords = (image_x, image_y)

                    msi_x, msi_y = None, None
                    # try adding the msi coordinates to the teaching point
                    if v.msi_rect is not None and v.px_rect is not None:
                        msi_x, msi_y = tp.get_msi_coords_from_px(v.msi_rect, v.px_rect)

                    # record the teaching point in the treeview, under the image name parent
                    tp.linked_tree_item = self.tree.insert(v.tree_master,
                                                           'end',
                                                           text="",
                                                           values=(tp.tag,
                                                                   round(image_x, 0),
                                                                   round(image_y, 0),
                                                                   msi_x,msi_y, tp.depth))
                    break

    def bind_events_to_loaded_images(self, loaded_image):
        """Bind events to the loaded images"""
        self.canvas.tag_bind(f"{loaded_image.tag}", "<Button-1>",
                             lambda e, item=f"{loaded_image.tag}": self.on_drag_start(item, e))
        # bind ctrl-left-click to add a ruler
        self.canvas.tag_bind(f"{loaded_image.tag}", "<Control-Button-1>", self.add_vertical_line)
        # bind shift-left-click to add a teaching point
        self.canvas.tag_bind(f"{loaded_image.tag}", "<Shift-Button-1>", self.add_teaching_point)
        # bind right-click event to the image
        self.canvas.tag_bind(f"{loaded_image.tag}",
                             "<Button-2>",
                             lambda e, item=f"{loaded_image.tag}": self.right_click_on_image.show_menu(e, item))

    def bind_events_to_vertical_lines(self, vertical_line):
        """Bind events to the vertical lines"""
        self.canvas.tag_bind(f"{vertical_line.tag}",
                             "<Button-2>",
                             lambda e, item=f"{vertical_line.tag}": self.right_click_on_line.show_menu(e, item))
        # bind shift-left-click to add a teaching point
        self.canvas.tag_bind(f"{vertical_line.tag}", "<Shift-Button-1>", self.add_teaching_point)

    def add_metadata(self):
        """Add metadata to the app"""
        file_path = filedialog.askopenfilename()
        if file_path:
            # connect to the sqlite database
            import sqlite3
            conn = sqlite3.connect(file_path)
            c = conn.cursor()
            # get the image name, px_rect, and msi_rect
            c.execute('SELECT * FROM mis')
            data = c.fetchall()
            for row in data:
                im_name, px_rect, msi_rect = row
                im_name = 'im_' + im_name
                # attach the metadata to the corresponding image
                try:
                    self.items[im_name].px_rect = eval(px_rect)
                    self.items[im_name].msi_rect = eval(msi_rect)
                    print(f"px_rect: {self.items[im_name].px_rect}, msi_rect: {self.items[im_name].msi_rect}")
                except KeyError:
                    pass
            conn.close()
        else:
            print("No file path is given")

    def save(self):
        """Save the current state of the canvas"""
        # get the file path to save the state
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        data_to_save = {"cm_per_pixel": self.cm_per_pixel, "items": []}
        try:
            data_to_save["scale_line0"] = self.scale_line[0]
            data_to_save["scale_line1"] = self.scale_line[1]
        except IndexError:
            pass
        data_to_save["sediment_start_line"] = self.sediment_start_line

        # save the treeview in json format
        for k, v in self.items.items():
            data_to_save["items"].append(v.to_json())
        with open(file_path, "w") as f:
            json.dump(data_to_save, f)

    def load(self):
        """Load the state of the canvas"""
        file_path = filedialog.askopenfilename(defaultextension=".json")
        with open(file_path, "r") as f:
            data = json.load(f)
            self.cm_per_pixel = data["cm_per_pixel"]
            try:
                self.scale_line.append(data["scale_line0"])
                self.scale_line.append(data["scale_line1"])
            except KeyError:
                pass
            self.sediment_start_line = data["sediment_start_line"]

            for item in data["items"]:
                if item["type"] == "LoadedImage":
                    loaded_image = LoadedImage.from_json(item, self)
                    self.items[loaded_image.tag] = loaded_image
                elif item["type"] == "TeachingPoint":
                    teaching_point = TeachingPoint.from_json(item, self)
                    self.items[teaching_point.tag] = teaching_point
                elif item["type"] == "VerticalLine":
                    vertical_line = VerticalLine.from_json(item, self)
                    self.items[vertical_line.tag] = vertical_line

            # restore the treeview
            for item in self.items.values():
                if item.type == "TeachingPoint":
                    item.linked_tree_item = self.tree.insert(self.items[item.linked_im].tree_master,
                                                             'end',
                                                             text="",
                                                             values=(item.tag,
                                                                     item.image_coords[0],
                                                                     item.image_coords[1],
                                                                     "", "",
                                                                     item.depth))

    def export_tps(self, file_path):
        """Export the teaching points to a json file"""
        data_to_save = "img;label;px;py;mx;my;d\n"
        for k, v in self.items.items():
            if v.type == "TeachingPoint":
                data_to_save += f"{v.linked_im};{self.items[v.linked_im].label};{v.image_coords[0]};{v.image_coords[1]};{v.msi_coords[0]};{v.msi_coords[1]};{v.depth}\n"
        with open(file_path, "w") as f:
            f.write(data_to_save)

    def main(self):
        self.mainloop()


if __name__ == "__main__":
    app = MainApplication()
    app.main()
