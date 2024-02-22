import json
import re
import tkinter as tk
from tkinter import filedialog

import numpy as np
import tqdm

from objects import LoadedImage, VerticalLine, MsiImage, XrayImage, LinescanImage
from menubar import MenuBar
from rclick import RightClickOnLine, RightClickOnImage, RightClickOnTeachingPoint
from func import CorSolver, sort_points_clockwise
import logging


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.n_xray = 0
        self.n_linescan = 0
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
        self.sediment_start = None

        self.cm_per_pixel = None
        # add menubar
        self.menu = MenuBar(self)

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
        self.items[item].origin = (x, y)
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
        if self.sediment_start is not None and self.cm_per_pixel is not None:
            pixel_distance = self.canvas.coords(self.sediment_start)[0] - self.canvas.canvasx(event.x)
            real_distance = pixel_distance * self.cm_per_pixel
            vl.add_depth_text(self, f"{abs(real_distance):.2f}")

    def find_clicked_image(self, event):
        for k, v in self.items.items():
            if isinstance(v, LoadedImage):
                logging.debug(f"v.tag: {v.tag}")
                x1, y1, x2, y2 = self.canvas.bbox(v.tag)
                if x1 <= self.canvas.canvasx(event.x) <= x2 and y1 <= self.canvas.canvasy(event.y) <= y2:
                    return v
        return None

    def bind_events_to_loaded_images(self, loaded_image):
        """Bind events to the loaded images"""
        self.canvas.tag_bind(f"{loaded_image.tag}", "<Button-1>",
                             lambda e, item=f"{loaded_image.tag}": self.on_drag_start(item, e))
        # bind ctrl-left-click to add a ruler
        self.canvas.tag_bind(f"{loaded_image.tag}", "<Control-Button-1>", self.add_vertical_line)
        # bind shift-left-click to add a teaching point
        self.canvas.tag_bind(f"{loaded_image.tag}", "<Shift-Button-1>",
                             lambda e: self.add_teaching_point(e))
        # bind right-click event to the image
        self.canvas.tag_bind(f"{loaded_image.tag}",
                             "<Button-2>",
                             lambda e, item=f"{loaded_image.tag}": self.right_click_on_image.show_menu(e, item))

    def calc_transformation_matrix(self):
        """ solve the transformation among MSi coordinates, xray pixel coordinates, and line scan depth"""
        # get all fixed points from xray teaching points
        xray_tps = []
        linescan_tps = []
        for k, v in self.items.items():
            if isinstance(v, XrayImage):
                logging.debug(f"v.tag: {v.tag} \n v.teaching_points: {v.teaching_points}")
                xray_tps.extend(v.teaching_points.values())
        # get all fixed points from xray teaching points
        xray_ds = []
        for i, v in enumerate(xray_tps):
            xray_ds.append(v[2])
            linescan_tps.append([v[2], v[1]])
            xray_tps[i] = v[:2]

        # sort the xray_tps by the x coordinate, and cut them to groups, each with 3 points
        xray_tps = sorted(xray_tps, key=lambda x: x[0])
        xray_tps = [xray_tps[i:i + 3] for i in range(0, len(xray_tps), 3)]
        # sort the xray_ds, and cut them to groups, each with 3 points, and get the average depth
        xray_ds = sorted(xray_ds)
        xray_ds = [xray_ds[i:i + 3] for i in range(0, len(xray_ds), 3)]
        xray_ds = [np.mean(d) for d in xray_ds]

        logging.debug(f"xray_tps: {xray_tps}")
        # in each group, sort the points clockwise
        for i, group in enumerate(xray_tps):
            xray_tps[i] = sort_points_clockwise(np.array(group))
            logging.debug(f"group: {group}")

        # sort the linescan_tps by the x coordinate, and cut them to groups, each with 3 points
        linescan_tps = sorted(linescan_tps, key=lambda x: x[0])
        linescan_tps = [linescan_tps[i:i + 3] for i in range(0, len(linescan_tps), 3)]
        # in each group, sort the points clockwise
        for i, group in enumerate(linescan_tps):
            linescan_tps[i] = sort_points_clockwise(np.array(group))
            logging.debug(f"group: {group}")

        msi_tps = {}
        msi_ds = {}

        for k, v in self.items.items():
            if isinstance(v, MsiImage):
                msi_tps[k] = v.teaching_points
        for k, v in msi_tps.items():
            msi_ds[k] = np.array(list(v.values()))[:, 2].mean()
            msi_tps[k] = np.array(list(v.values()))[:, :2]
            msi_tps[k] = sort_points_clockwise(np.array(msi_tps[k]))

        # solve the affine transformation of how to transform from msi_tps to xray_tps
        self.solvers_xray = {}
        self.solvers_depth = {}
        for k, v in msi_tps.items():
            msi_d = msi_ds[k]
            # find the index of the xray teaching points that are closest to the msi teaching points
            diff = np.abs(np.array(xray_ds) - msi_d)
            idx = np.argmin(diff)

            assert diff.min() < 0.5, f"vertical_distance: {diff.min()} is too large"

            # get the xray teaching points that are closest to the msi teaching points
            self.solvers_xray[k] = CorSolver()
            self.solvers_xray[k].fit(v, xray_tps[idx])
            # solve the transformation of how to transform from msi_tps to line_scan_tps
            self.solvers_depth[k] = CorSolver()
            self.solvers_depth[k].fit(v, linescan_tps[idx])

    def machine_to_real_world(self):
        """apply the transformation to the msi teaching points"""
        # ask for the sqlite file to read the metadata
        file_path = filedialog.askopenfilename()
        if file_path:
            # connect to the sqlite database
            import sqlite3
            conn = sqlite3.connect(file_path)
            c = conn.cursor()
            try:
                c.execute('SELECT msi_img_file_name, spot_array FROM transformation')
                data = c.fetchall()
            except sqlite3.OperationalError:
                logging.debug("The transformation table does not exist yet, creating one")
                c.execute('CREATE TABLE transformation (msi_img_file_name TEXT, spot_array BLOB)')
                c.execute('SELECT msi_img_file_name, spot_name FROM metadata')
                data = c.fetchall()
                logging.debug(f"reading all the spotname from metadata table and convert them to array")
                for row in tqdm.tqdm(data):
                    im_name, spot_name = row
                    spot_name = eval(spot_name)
                    # apply the transformation to the spot_name
                    spot_name = [re.findall(r'X(\d+)Y(\d+)', s) for s in spot_name]
                    # flatten the list
                    spot_name = [item for sublist in spot_name for item in sublist]
                    # convert to an array
                    spot_name = np.array(spot_name)
                    # conver the spotname to int
                    spot_name = spot_name.astype(int)
                    # write the spotnames to the transformation table as a blob
                    # insert the transformed spot_name to the transformation table
                    c.execute('INSERT INTO transformation VALUES (?, ?)', (im_name,spot_name.tobytes()))
                conn.commit()
                c.execute('SELECT msi_img_file_name, spot_array FROM transformation')
                data = c.fetchall()
            for row in data:
                im_name, spot_array = row
                spot_array = np.frombuffer(spot_array, dtype=int).reshape(-1, 2)
                logging.debug(f"im_name: {im_name}, spot_array: {spot_array}")
                # apply the transformation to the spot_array
                if im_name in self.solvers_xray.keys():
                    xray_array = self.solvers_xray[im_name].transform(spot_array)
                    linescan_array = self.solvers_depth[im_name].transform(spot_array)
                    # insert the xray array to a new column in the transformation table
                    c.execute('ALTER TABLE transformation ADD COLUMN xray_array BLOB')
                    c.execute('UPDATE transformation SET xray_array = ? WHERE msi_img_file_name = ?',
                              (xray_array.tobytes(), im_name))
                    # insert the linescan array to a new column in the transformation table
                    c.execute('ALTER TABLE transformation ADD COLUMN linescan_array BLOB')
                    c.execute('UPDATE transformation SET linescan_array = ? WHERE msi_img_file_name = ?',
                              (linescan_array.tobytes(), im_name))
                else:
                    logging.debug(f"{im_name} is not in the solvers_xray.keys()")
            conn.commit()
            conn.close()
        else:
            logging.debug("No file path is given")



    def bind_events_to_vertical_lines(self, vertical_line):
        """Bind events to the vertical lines"""
        self.canvas.tag_bind(f"{vertical_line.tag}",
                             "<Button-2>",
                             lambda e, item=f"{vertical_line.tag}": self.right_click_on_line.show_menu(e, item))
        # bind shift-left-click to add a teaching point
        self.canvas.tag_bind(f"{vertical_line.tag}", "<Shift-Button-1>", self.add_teaching_point)

    def add_teaching_point(self, event):
        """Add a teaching point to the canvas"""
        clicked_image = self.find_clicked_image(event)
        assert isinstance(clicked_image, XrayImage) or isinstance(clicked_image, MsiImage), (
            "You need to click on an xray image"
            "or a MSI image to add a "
            "teaching point")
        if clicked_image is not None:
            clicked_image.add_teaching_point(event, self)

    def add_metadata(self):
        """Add metadata to the app"""
        file_path = filedialog.askopenfilename()
        if file_path:
            # connect to the sqlite database
            import sqlite3
            conn = sqlite3.connect(file_path)
            c = conn.cursor()
            # get the image name, px_rect, and msi_rect
            c.execute('SELECT msi_img_file_name, px_rect, msi_rect FROM metadata')
            data = c.fetchall()
            for row in data:
                im_name, px_rect, msi_rect = row
                im_name = im_name
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
        data_to_save["sediment_start"] = self.sediment_start

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
            self.sediment_start = data["sediment_start"]

            for item in data["items"]:
                if "MsiImage" in item["type"]:
                    loaded_image = MsiImage.from_json(item, self)
                    self.items[loaded_image.tag] = loaded_image
                elif "XrayImage" in item["type"]:
                    loaded_image = XrayImage.from_json(item, self)
                    self.items[loaded_image.tag] = loaded_image
                elif "LinescanImage" in item["type"]:
                    loaded_image = LinescanImage.from_json(item, self)
                    self.items[loaded_image.tag] = loaded_image
                elif item["type"] == "VerticalLine":
                    vertical_line = VerticalLine.from_json(item, self)
                    self.items[vertical_line.tag] = vertical_line
                    self.bind_events_to_vertical_lines(vertical_line)

    def export_tps(self, file_path):
        """Export the teaching points to a json file"""
        data_to_save = "img;x;y;d\n"
        for k, v in self.items.items():
            if hasattr(v, "teaching_points"):
                for k, tp in v.teaching_points.items():
                    data_to_save += f"{v.tag};{tp[0]};{tp[1]};{tp[2]}\n"
        with open(file_path, "w") as f:
            f.write(data_to_save)

    def main(self):
        self.mainloop()


if __name__ == "__main__":
    app = MainApplication()
    app.main()
