import json
import re
import sqlite3
import sys
import tkinter as tk
from tkinter import ttk, simpledialog
from tkinter import filedialog
import logging

import numpy as np
import tqdm

logging.basicConfig(level=logging.DEBUG)

from objects import LoadedImage, VerticalLine, MsiImage, XrayImage, LinescanImage, TeachableImage
from menubar import MenuBar
from rclick import RightClickOnLine, RightClickOnImage, RightClickOnTeachingPoint
from func import CorSolver, sort_points_clockwise


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("1200x800")
        self.n_xray = 0
        self.n_linescan = 0
        self.canvas = None
        self.right_click_on_tp = None
        self.right_click_on_image = None
        self.right_click_on_line = None
        self.title('CorelDraw Imposter')
        self.items = {}
        self.create_canvas()

        self.database_path = None

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
        canvas_frame = tk.Frame(self)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, width=canvas_width, height=canvas_height, bg='white',
                                scrollregion=(0, 0, 5000, 5000))
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Create horizontal and vertical scrollbars
        h_scroll = tk.Scrollbar(self.canvas, orient='horizontal', command=self.canvas.xview)
        h_scroll.pack(side=tk.BOTTOM, fill='x')
        v_scroll = tk.Scrollbar(canvas_frame, orient='vertical', command=self.canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill='y')
        # bind the mousewheel event to the canvas
        # Configure the canvas to use the scrollbars
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self.horizontal_mousewheel)

    def on_mousewheel(self, event):
        try:
            # For windows and MacOS
            logging.debug(f"event.delta: {event.delta}")
            self.canvas.yview_scroll(event.delta, "units")
        except AttributeError:
            raise AttributeError("The mousewheel event is not supported on this platform")

    def horizontal_mousewheel(self, event):
        try:
            # For windows and MacOS
            logging.debug(f"event.delta: {event.delta}")
            self.canvas.xview_scroll(event.delta, "units")
        except AttributeError:
            raise AttributeError("The mousewheel event is not supported on this platform")

    def on_drag_start(self, item, event):
        """Function to handle dragging"""
        # Get the coordinates of the image
        x1, y1, x2, y2 = self.canvas.bbox(item)

        # calculate the offset
        _drag_offset_x = self.canvas.canvasx(event.x) - x1
        _drag_offset_y = self.canvas.canvasy(event.y) - y1
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
            self.canvas.bind("<B1-Motion>", lambda e: self.on_drag_move(e, item, _drag_offset_x, _drag_offset_y))
            self.canvas.bind("<ButtonRelease-1>", lambda e: self.on_drag_stop(e, item))

    def on_drag_move(self, event, item, _drag_offset_x, _drag_offset_y):
        """move the item to the new position"""
        x, y = self.canvas.canvasx(event.x) - _drag_offset_x, self.canvas.canvasy(event.y) - _drag_offset_y
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

    def view_blob_data(self):
        # get a popup window to choose SPEC_ID and COLUMN_NAME
        popup = tk.Toplevel()
        popup.title("View BLOB Data")
        popup.geometry("300x200")
        # create a label to display the SPEC_ID
        spec_id_label = tk.Label(popup, text="SPEC_ID:")
        spec_id_label.pack()
        spec_id_entry = tk.Entry(popup)
        spec_id_entry.pack()
        # create a label to display the COLUMN_NAME
        column_name_label = tk.Label(popup, text="COLUMN_NAME:")
        column_name_label.pack()
        column_name_entry = tk.Entry(popup)
        column_name_entry.pack()
        # create a label to display the Table Name
        table_name_label = tk.Label(popup, text="Table Name:")
        table_name_label.pack()
        table_name_entry = tk.Entry(popup)
        table_name_entry.pack()
        # create a button to submit the SPEC_ID and COLUMN_NAME
        submit_button = tk.Button(popup, text="Submit", command=lambda: self.get_blob_data(spec_id_entry.get(),
                                                                                           table_name_entry.get(),
                                                                                           column_name_entry.get()))
        submit_button.pack()

    def get_blob_data(self, spec_id, table_name, column_name):
        if self.database_path is None:
            file_path = filedialog.askopenfilename()
            if file_path:
                database_path = file_path
            else:
                raise ValueError("You need to select a database file")
        else:
            database_path = self.database_path
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT {column_name} FROM {table_name} WHERE spec_id={spec_id};")
        blob = cursor.fetchone()[0]
        conn.close()

        if 'spot' in column_name:
            array = np.frombuffer(blob, dtype=np.int64).reshape(-1, 2)
        else:
            array = np.frombuffer(blob, dtype=np.float64).reshape(-1, 2)
        logging.debug(f"array: {array}")

        # popup a window to display the blob data and add a scrollbar
        popup = tk.Toplevel()
        popup.title("BLOB Data")
        popup.geometry("800x600")
        container = tk.Frame(popup)
        container.pack(fill=tk.BOTH, expand=True)
        # create a treeview to display the blob data
        tree = ttk.Treeview(container, show="headings")
        # Initialize Treeview columns on first load
        tree['columns'] = ['x', 'y']
        tree.heading('x', text='x')
        tree.heading('y', text='y')
        # add a scrollbar to the treeview
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        # Pack the Treeview and Scrollbar in the container frame
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side="right", fill="y")
        # insert the blob data to the treeview
        for i, row in enumerate(array):
            tree.insert("", "end", values=(row[0], row[1]))
        conn.close()

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
        # TODO: could be optimized by using addtag_withtag when creating the images and the items
        clicked_images = []
        for k, v in self.items.items():
            if isinstance(v, LoadedImage):
                logging.debug(f"v.tag: {v.tag}")
                x1, y1, x2, y2 = self.canvas.bbox(v.tag)
                if x1 <= self.canvas.canvasx(event.x) <= x2 and y1 <= self.canvas.canvasy(event.y) <= y2:
                    clicked_images.append(v)

        if len(clicked_images) == 1:
            return clicked_images[0]
        elif len(clicked_images) > 1:
            # when images are overlapping, find the front image
            # first call find_all to get the current order of the images
            current_order = self.canvas.find_all()
            # sort the clicked images based on the current order
            clicked_images = sorted(clicked_images,
                                    key=lambda x: current_order.index(self.canvas.find_withtag(x.tag)[0]))
            logging.debug(f'the front image is {clicked_images[-1]}')
            return clicked_images[-1]
        else:
            return None

    def lock_all(self):
        # invoke lock_image method for all the images
        for k, v in self.items.items():
            if isinstance(v, LoadedImage):
                self.right_click_on_image.lock_image(k)
        logging.debug("All the images are locked")

    def move_all_tps_to_top(self):
        # move all the teaching points (with tag 'tp_*') in canvas to the top
        tps = self.find_wildcard('tp_')
        for tp in tps:
            self.canvas.tag_raise(tp)
        logging.debug("All the teaching points are moved to the top")

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
        if sys.platform == "darwin":
            self.canvas.tag_bind(f"{loaded_image.tag}",
                                 "<Button-2>",
                                 lambda e, item=f"{loaded_image.tag}": self.right_click_on_image.show_menu(e, item))
        elif sys.platform == "win32":
            self.canvas.tag_bind(f"{loaded_image.tag}",
                                 "<Button-3>",
                                 lambda e, item=f"{loaded_image.tag}": self.right_click_on_image.show_menu(e, item))
        else:
            raise ValueError("The platform is not supported")

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
        if self.database_path is None:
            file_path = filedialog.askopenfilename()
            if file_path:
                self.database_path = file_path
            else:
                raise ValueError("You need to select a database file")
        # connect to the sqlite database
        file_path = self.database_path
        if file_path:
            # connect to the sqlite database
            conn = sqlite3.connect(file_path)
            c = conn.cursor()
            try:
                # check if the transformation table exists
                c.execute('SELECT * FROM transformation')
            except sqlite3.OperationalError:
                logging.debug("The transformation table does not exist yet, creating one")
                # create a transformation table with metadata(spec_id) as the reference key
                c.execute(
                    'CREATE TABLE transformation (spec_id INTEGER, msi_img_file_name TEXT, spot_array BLOB, xray_array BLOB, linescan_array BLOB, FOREIGN KEY(spec_id) REFERENCES metadata(spec_id))')
                conn.commit()
                # read all the spotname from metadata table and convert them to array
                c.execute('SELECT spec_id, msi_img_file_name, spot_name FROM metadata')
                data = c.fetchall()
                assert len(data) > 0, "No data is found in the metadata table"
                for row in tqdm.tqdm(data):
                    spec_id, im_name, spot_name = row
                    spec_id = int(spec_id)
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
                    c.execute('INSERT INTO transformation (spec_id, msi_img_file_name, spot_array) VALUES (?, ?, ?)',
                              (spec_id, im_name, spot_name.tobytes()))
                    conn.commit()
                    # store_blob_info(conn, 'spot_array', spot_name.dtype, spot_name.shape)

            c.execute('SELECT spec_id, msi_img_file_name, spot_array FROM transformation')
            data = c.fetchall()
            for row in data:
                spec_id, im_name, spot_array = row
                spec_id = int(spec_id)
                spot_array = np.frombuffer(spot_array, dtype=int).reshape(-1, 2)
                logging.debug(f"spec_id: {spec_id}, im_name: {im_name}, spot_array: {spot_array}")
                # apply the transformation to the spot_array
                if im_name in self.solvers_xray.keys():
                    xray_array = self.solvers_xray[im_name].transform(spot_array)
                    xray_array_dtype = xray_array.dtype
                    logging.debug(f"xray_array_dtype: {xray_array_dtype}")
                    # store_blob_info(conn, 'xray_array', xray_array_dtype, xray_array.shape)
                    xray_array_shape = xray_array.shape
                    logging.debug(f"xray_array_shape: {xray_array_shape}")
                    linescan_array = self.solvers_depth[im_name].transform(spot_array)
                    # line_scan_dtype = linescan_array.dtype
                    # linescan_array_shape = linescan_array.shape
                    # store_blob_info(conn, 'linescan_array', line_scan_dtype, linescan_array_shape)
                    c.execute('UPDATE transformation SET xray_array = ? WHERE spec_id = ?',
                              (xray_array.tobytes(), spec_id))
                    c.execute('UPDATE transformation SET linescan_array = ? WHERE spec_id = ?',
                              (linescan_array.tobytes(), spec_id))
                else:
                    logging.debug(f"{im_name} is not in the solvers_xray.keys()")
            # store the blob info to the blob_info table

            conn.commit()

            conn.close()
        else:
            logging.debug("No file path is given")

    def set_tp_size(self):
        """set the size of the teaching points"""
        size = simpledialog.askinteger("Input", "Enter the size of the teaching points", initialvalue=5)
        size = size if size is not None else 5
        for k, v in self.items.items():
            if isinstance(v, TeachableImage):
                v.tp_size = size

    def bind_events_to_vertical_lines(self, vertical_line):
        """Bind events to the vertical lines"""
        if sys.platform == "darwin":
            self.canvas.tag_bind(f"{vertical_line.tag}",
                                 "<Button-2>",
                                 lambda e, item=f"{vertical_line.tag}": self.right_click_on_line.show_menu(e, item))
        elif sys.platform == "win32":
            self.canvas.tag_bind(f"{vertical_line.tag}",
                                 "<Button-3>",
                                 lambda e, item=f"{vertical_line.tag}": self.right_click_on_line.show_menu(e, item))
        else:
            raise ValueError("The platform is not supported")
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
        if self.database_path is None:
            file_path = filedialog.askopenfilename()
            if file_path:
                self.database_path = file_path
            else:
                raise ValueError("You need to select a database file")
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

    def use_as_ref_to_resize(self, item):
        """use the selected image as the reference to resize other images"""
        ref_width = self.items[item].thumbnail.width
        for k, v in self.items.items():
            logging.debug(f"{k} has the class of {v.__class__}")
            if isinstance(v, MsiImage):
                scale_factor = ref_width / v.thumbnail.width
                self.items[k].enlarge(scale_factor)
                self.canvas.itemconfig(k, image=self.items[k].tk_img)

    def save(self):
        """Save the current state of the canvas"""
        # get the file path to save the state
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        data_to_save = {"cm_per_pixel": self.cm_per_pixel, "items": [], 'database_path': self.database_path}

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
                self.database_path = data["database_path"]
            except KeyError:
                pass
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

    def find_wildcard(self, wildcard):
        """find the tag with the wildcard"""
        items = self.canvas.find_all()
        matched_items = []
        for item in items:
            logging.debug(f"item: {item}, tags: {self.canvas.gettags(item)[0]}")
            if wildcard in self.canvas.gettags(item)[0]:
                matched_items.append(self.canvas.gettags(item))
        return matched_items

    def reset_tp(self):
        """
        Reset the teaching points
        """
        # remove all the teaching points from the canvas
        logging.debug("Resetting teaching points")
        for k, v in self.items.items():
            if isinstance(v, TeachableImage):
                if v.teaching_points is not None:
                    v.teaching_points = {}
                    logging.debug(f"Reset teaching points attributes for {k} successfully")
                if hasattr(v, "teaching_points_updated"):
                    v.teaching_points_updated = False
                    logging.debug(f"Reset teaching_points_updated attributes for {k} successfully")
        # hard remove all the teaching points oval from the canvas with tag including 'tp_'
        try:
            # list all tags
            tps = self.find_wildcard('tp_')
            logging.debug(f"tps: {tps}")
            for tp in tps:
                self.canvas.delete(tp)
            logging.debug("Deleting teaching points from the canvas successfully")
        except AttributeError:
            logging.debug("No teaching points found")
            pass

        # clear the tree view
        try:
            self.tree.delete(*self.tree.get_children())
            logging.debug("Reset the tree view successfully")
        except AttributeError:
            logging.debug("No tree view found")
            pass

    def main(self):
        self.mainloop()


if __name__ == "__main__":
    app = MainApplication()
    app.main()
