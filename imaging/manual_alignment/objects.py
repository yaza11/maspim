import os
import tkinter as tk
from PIL import Image, ImageTk
Image.MAX_IMAGE_PIXELS = None


class LoadedImage:
    def __init__(self):
        self.path = None
        self.orig_img = None
        self.thumbnail = None
        self.tk_img = None
        self._tag = None
        self.rotation = 0
        self.locked = False
        self.position = (0, 0)
        self.label = None
        self.thumbnail_size = (500, 500)
        self.tree_master = None
        self.type = "LoadedImage"
        self.img_type = None
        self.msi_rect = None
        self.px_rect = None

    @property
    def tag(self):
        return self._tag

    def __str__(self):
        return self.path

    def __repr__(self):
        return self.path

    @classmethod
    def from_path(cls, path):
        self = cls()
        self.path = path
        self._tag = 'im_' + os.path.basename(path)
        self.orig_img = Image.open(path)
        self.thumbnail = self.orig_img.copy()
        self.thumbnail.thumbnail(self.thumbnail_size)
        self.tk_img = ImageTk.PhotoImage(self.thumbnail)
        self.tree_master = None
        return self

    def create_im_on_canvas(self, app):
        # create the image on the canvas
        app.canvas.create_image(
            self.position[0],
            self.position[1],
            anchor="nw",
            image=self.tk_img,
            tags=f"{self.tag}"
        )
        # insert the tag into the tree
        self.tree_master = app.tree.insert(
            "",
            'end',
            text=self.tag,
            values=("", "", "", "", "", "")
        )

        # bind events to the image
        app.bind_events_to_loaded_images(self)

    def resize(self, size):
        self.thumbnail_size = size
        self.thumbnail = self.orig_img.copy()
        self.thumbnail.thumbnail(size)
        self.tk_img = ImageTk.PhotoImage(self.thumbnail)

    def rotate(self):
        self.rotation += 90
        self.thumbnail = self.thumbnail.rotate(90, expand=True)
        self.tk_img = ImageTk.PhotoImage(self.thumbnail)

    def enlarge(self, scale_factor):
        self.thumbnail_size = (
            self.thumbnail.width * scale_factor,
            self.thumbnail.height * scale_factor
        )
        self.thumbnail = self.orig_img.copy()
        self.thumbnail.thumbnail(self.thumbnail_size)
        self.tk_img = ImageTk.PhotoImage(self.thumbnail)

    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False

    def to_json(self):
        return {
            "type": "LoadedImage",
            "label": self.tag,
            "path": self.path,
            "rotation": self.rotation,
            "locked": self.locked,
            "thumbnail_size": self.thumbnail_size,
            "position": self.position,
            "tree_master": self.tree_master,
            "msi_rect": self.msi_rect,
            "px_rect": self.px_rect
        }

    @classmethod
    def from_json(cls, json_data, app):
        self = cls()
        self.path = json_data['path']
        self._tag = json_data['label']
        self.rotation = json_data['rotation']
        self.px_rect = json_data['px_rect']
        self.msi_rect = json_data['msi_rect']
        self.locked = json_data['locked']
        self.thumbnail_size = json_data['thumbnail_size']
        self.position = json_data['position']
        self.orig_img = Image.open(self.path)
        self.thumbnail = self.orig_img.copy()
        self.thumbnail.thumbnail(self.thumbnail_size)
        self.thumbnail = self.thumbnail.rotate(self.rotation, expand=True)
        self.tk_img = ImageTk.PhotoImage(self.thumbnail)
        self.tree_master = json_data['tree_master']
        self.create_im_on_canvas(app)
        return self

    def rm(self, app):
        # remove the image from the canvas
        app.canvas.delete(self.tag)
        # remove the image from the tree
        app.tree.delete(self.tree_master)
        # remove from the items dictionary
        del app.items[self.tag]


class TeachingPoint:
    def __init__(self, position):
        self.position = position
        self.linked_tree_item = None
        self.linked_im = None
        self.path_to_image = None
        self.image_coords = None
        self.depth = None
        self._tag = f"tp_{position[0]}_{position[1]}"
        self.msi_coords = None
        self.type = "TeachingPoint"

    def get_msi_coords_from_px(self, msi_rect, px_rect):
        print(f"msi_rect: {msi_rect}")
        print(f"px_rect: {px_rect}")
        x_min, y_min, x_max, y_max = msi_rect
        x_min_px, y_min_px, x_max_px, y_max_px = px_rect
        msi_x = (self.image_coords[0] - x_min_px) / (x_max_px - x_min_px) * (x_max - x_min) + x_min
        msi_y = (self.image_coords[1] - y_min_px) / (y_max_px - y_min_px) * (y_max - y_min) + y_min
        self.msi_coords = (msi_x, msi_y)
        return msi_x, msi_y

    @property
    def tag(self):
        return self._tag

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    def __str__(self):
        return self.tag

    def __repr__(self):
        return self.tag

    def rm(self, app):
        # remove the teaching point from the canvas
        app.canvas.delete(self.tag)
        # remove the teaching point from the tree
        app.tree.delete(self.linked_tree_item)
        # remove from the items dictionary
        del app.items[self.tag]

    def create_on_canvas(self, app):
        app.canvas.create_oval(
            self.position[0] - 5,
            self.position[1] - 5,
            self.position[0] + 5,
            self.position[1] + 5,
            fill="red",
            tags=self.tag
        )
        app.canvas.tag_bind(self.tag,
                            "<Button-2>",
                            lambda e: app.right_click_on_tp.show_menu(e, self.tag))

    @classmethod
    def from_json(cls, json_data, app):
        self = cls(json_data['position'])
        self._tag = json_data['tag']
        self.path_to_image = json_data['path_to_image']
        self.msi_coords = json_data['msi_coords']
        self.image_coords = json_data['image_coords']
        self.depth = json_data['depth']
        self.linked_im = json_data['linked_im']
        self.create_on_canvas(app)
        return self

    def to_json(self):
        return {
            "type": "TeachingPoint",
            "tag": self.tag,
            "position": self.position,
            "path_to_image": self.path_to_image,
            "image_coords": self.image_coords,
            "depth": self.depth,
            'linked_im': self.linked_im,
            'msi_coords': self.msi_coords
        }


class VerticalLine:
    color_map = {
        "scale_line": "blue",
        "sediment_start_line": "green"
    }

    def __init__(self, position):
        self.position = position
        self.tree_master = None
        self.depth = None
        self.path_to_image = None
        self.label = None
        self._tag = f"vl_{position[0]}"
        self.type = "VerticalLine"

    @property
    def tag(self):
        return self._tag

    def create_on_canvas(self, app):
        app.canvas.create_line(
            self.position[0],
            0,
            self.position[0],
            5000,
            fill=self.color,
            tags=self.tag,
            width=1,
            dash=(4, 4)
        )
        app.bind_events_to_vertical_lines(self)

    def add_depth_text(self, app, depth):
        self.depth = depth
        text = tk.Text(app.canvas, height=1, width=8)
        text.insert(tk.END, f"{depth}cm")
        text.config(state=tk.DISABLED)
        app.canvas.create_window(
            self.position[0],
            10,
            window=text,
            anchor="nw",
            tags=self.tag
        )

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def color(self):
        if self.label in self.color_map:
            return self.color_map[self.label]
        else:
            return "red"

    def __str__(self):
        return self.tag

    def __repr__(self):
        return self.tag

    def rm(self, app):
        # remove the teaching point from the canvas
        app.canvas.delete(self.tag)
        # remove from the items dictionary
        del app.items[self.tag]

    @classmethod
    def from_json(cls, json_data, app):
        self = cls(json_data['position'])
        self._tag = json_data['tag']
        self.label = json_data['label']
        self.path_to_image = json_data['path_to_image']
        self.depth = json_data['depth']
        self.create_on_canvas(app)
        return self

    def to_json(self):
        return {
            "type": "VerticalLine",
            "tag": self.tag,
            "position": self.position,
            "path_to_image": self.path_to_image,
            "depth": self.depth,
            "label": self.label
        }


if __name__ == "__main__":
    pass
