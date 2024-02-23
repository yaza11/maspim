import logging
import os
import tkinter as tk
from PIL import Image, ImageTk

Image.MAX_IMAGE_PIXELS = None


class LoadedImage:
    """A superclass for the loaded images"""

    def __init__(self):
        self.img_path = None
        self.img = None
        self.thumbnail = None
        self.thumbnail_size = (500, 500)  # the size of the thumbnail on the canvas
        self.tk_img = None
        self._tag = None
        self.locked = False
        self.origin = (0, 0)  # the origin of the image on the canvas

    @property
    def x(self):
        return self.origin[0]

    @property
    def y(self):
        return self.origin[1]

    @property
    def tag(self):
        return self._tag

    def __str__(self):
        return self.img_path

    def __repr__(self):
        return self.img_path

    @classmethod
    def from_path(cls, img_path):
        self = cls()
        self.img_path = img_path
        self._tag = os.path.basename(img_path)
        self.img = Image.open(img_path)
        self.thumbnail = self.img.copy()
        self.thumbnail.thumbnail(self.thumbnail_size)
        self.tk_img = ImageTk.PhotoImage(self.thumbnail)
        return self

    def create_im_on_canvas(self, app):
        # create the image on the canvas
        app.canvas.create_image(
            self.origin[0],
            self.origin[1],
            anchor="nw",
            image=self.tk_img,
            tags=f"{self.tag}"
        )
        # bind events to the image
        app.bind_events_to_loaded_images(self)

    def resize(self, size):
        self.thumbnail_size = size
        self.thumbnail = self.img.copy()
        self.thumbnail.thumbnail(size)
        self.tk_img = ImageTk.PhotoImage(self.thumbnail)

    def enlarge(self, scale_factor):
        self.thumbnail_size = (
            self.thumbnail.width * scale_factor,
            self.thumbnail.height * scale_factor
        )
        self.thumbnail = self.img.copy()
        self.thumbnail.thumbnail(self.thumbnail_size)
        self.tk_img = ImageTk.PhotoImage(self.thumbnail)

    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False

    def to_json(self):
        logging.debug(f"Saving {self.__class__.__name__} to json")
        json_data = {
            "type": self.__class__.__name__,
            "img_path": self.img_path,
            "thumbnail_size": self.thumbnail_size,
            "origin": self.origin,
            "locked": self.locked,
        }
        return json_data

    @classmethod
    def from_json(cls, json_data, app):
        self = cls()
        self.img_path = json_data['img_path']
        self.thumbnail_size = json_data['thumbnail_size']
        self.origin = json_data['origin']
        self.locked = json_data['locked']
        self.img = Image.open(self.img_path)
        self.thumbnail = self.img.copy()
        self.thumbnail.thumbnail(self.thumbnail_size)
        self.tk_img = ImageTk.PhotoImage(self.thumbnail)
        self._tag = os.path.basename(self.img_path)
        self.create_im_on_canvas(app)
        return self

    def rm(self, app):
        pass


def draw_teaching_points(x, y, app):
    # mark the teaching point on the canvas
    app.canvas.create_oval(
        x - 5,
        y - 5,
        x + 5,
        y + 5,
        fill="red",
        tags=f"tp_{int(x)}_{int(y)}"
    )
    # bind events to the teaching point
    app.canvas.tag_bind(f"tp_{int(x)}_{int(y)}",
                        "<Button-2>",
                        lambda e: app.right_click_on_tp.show_menu(e, f"tp_{int(x)}_{int(y)}"))


class TeachableImage(LoadedImage):
    """A subclass of LoadedImage that holds the teachable images"""

    def __init__(self):
        super().__init__()
        self.teaching_points = None  # a list of teaching points

    def add_teaching_point(self, event, app):
        canvas_x, canvas_y = app.canvas.canvasx(event.x), app.canvas.canvasy(event.y)
        logging.debug(f"teaching point added canvas_x: {canvas_x}, canvas_y: {canvas_y}")
        # draw the teaching point on the canvas
        draw_teaching_points(canvas_x, canvas_y, app)
        # try to find the approximate depth of the teaching point
        if app.sediment_start is not None and app.cm_per_pixel is not None:
            depth = abs(app.canvas.coords(app.sediment_start)[0] - canvas_x) * app.cm_per_pixel
        else:
            depth = None

        original_width, original_height = self.img.size
        scale_x = original_width / self.thumbnail.width
        scale_y = original_height / self.thumbnail.height

        # calculate the coordinates of the teaching point in the original image
        img_x = (canvas_x - self.x) * scale_x
        img_y = (canvas_y - self.y) * scale_y

        if self.teaching_points is None:
            self.teaching_points = {}
        teaching_point_key = (canvas_x, canvas_y)
        self.teaching_points[teaching_point_key] = (img_x, img_y, depth)

        # once a teaching point is added, lock the image
        if len(self.teaching_points) > 0 and not self.locked:
            self.lock()

    def to_json(self):
        json_data = super().to_json()
        json_data["teaching_points"] = self.teaching_points
        # json data key cannot be a tuple, convert the key to a string
        json_data["teaching_points"] = {str(k): v for k, v in json_data["teaching_points"].items()}
        logging.debug(f"json data to write: {json_data}")
        return json_data

    @classmethod
    def from_json(cls, json_data, app):
        self = super().from_json(json_data, app)
        # convert the key back to a tuple
        json_data["teaching_points"] = {eval(k): v for k, v in json_data["teaching_points"].items()}
        self.teaching_points = json_data['teaching_points']
        logging.debug(f"teaching points: {self.teaching_points}")
        # draw the teaching points on the canvas if they exist
        try:
            if self.teaching_points is not None:
                for tp, _ in self.teaching_points.items():
                    draw_teaching_points(tp[0], tp[1], app)
        except Exception as e:
            logging.error(e)
            pass
        return self


class MsiImage(TeachableImage):
    """A subclass of LoadedImage that holds the MSI image"""

    def __init__(self):
        super().__init__()
        self.msi_rect = None  # the coordinates of the MSI image rectangle in R00X?Y? format
        self.px_rect = None  # the coordinates of the MSI image rectangle in pixel
        self.teaching_points_updated = False  # a flag to indicate if the teaching points have been updated

    def update_tp_coords(self):
        """ replace the coordinates of the teaching points with the MSI coordinates"""
        assert self.teaching_points_updated is False, "The teaching points have already been updated"
        assert self.msi_rect is not None and self.px_rect is not None, "You need to set the MSI and pixel rectangle first"
        assert self.teaching_points is not None, "You need to add teaching points first"
        logging.debug(f"msi_rect: {self.msi_rect}")
        logging.debug(f"px_rect: {self.px_rect}")
        x_min, y_min, x_max, y_max = self.msi_rect
        x_min_px, y_min_px, x_max_px, y_max_px = self.px_rect
        for k, v in self.teaching_points.items():
            msi_x = (v[0] - x_min_px) / (x_max_px - x_min_px) * (x_max - x_min) + x_min
            msi_y = (v[1] - y_min_px) / (y_max_px - y_min_px) * (y_max - y_min) + y_min
            self.teaching_points[k] = (msi_x, msi_y, v[2])
        self.teaching_points_updated = True

    def to_json(self):
        json_data = super().to_json()
        json_data["msi_rect"] = self.msi_rect
        json_data["px_rect"] = self.px_rect
        json_data["teaching_points_updated"] = self.teaching_points_updated
        return json_data

    @classmethod
    def from_json(cls, json_data, app):
        self = super().from_json(json_data, app)
        self.msi_rect = json_data['msi_rect']
        self.px_rect = json_data['px_rect']
        self.teaching_points_updated = json_data['teaching_points_updated']
        return self

    def rm(self, app):
        # remove the image from the canvas
        app.canvas.delete(self.tag)
        # remove from the items dictionary
        del app.items[self.tag]


class LinescanImage(LoadedImage):
    """A subclass of LoadedImage that holds the linescan image"""

    def __init__(self):
        super().__init__()

    def rm(self, app):
        # remove the image from the canvas
        app.canvas.delete(self.tag)
        # remove from the items dictionary
        del app.items[self.tag]
        app.n_linescan -= 1


class XrayImage(TeachableImage):
    """A subclass of LoadedImage that holds the xray image"""

    def __init__(self):
        super().__init__()

    def rm(self, app):
        # remove the image from the canvas
        app.canvas.delete(self.tag)
        # remove from the items dictionary
        del app.items[self.tag]
        app.n_xray -= 1


class VerticalLine:
    color_map = {
        "scale_line": "blue",
        "sediment_start_line": "green"
    }

    def __init__(self, position):
        self.position = position
        self.depth = None
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
        if self.tag in self.color_map:
            return self.color_map[self.tag]
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
        self.depth = json_data['depth']
        self.create_on_canvas(app)
        return self

    def to_json(self):
        return {
            "type": "VerticalLine",
            "tag": self.tag,
            "position": self.position,
            "depth": self.depth,
        }


if __name__ == "__main__":
    pass
