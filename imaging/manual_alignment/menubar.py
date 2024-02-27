import tkinter as tk
from tkinter import filedialog, simpledialog, ttk
import logging


from objects import XrayImage, LinescanImage, MsiImage


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
        # Add 'Add metadata' to the file menu
        self.file_menu.add_command(label="Attach database", command=self.app.add_metadata)
        # Add 'Exit' to the file menu
        self.file_menu.add_command(label="Quit", command=self.quit)

        # Add 'View' menu
        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=self.view_menu)
        # add 'Hide Teaching Points View' to the view menu
        self.view_menu.add_command(label="Toggle TP View", command=self.tg_tp_view)
        # update the teaching points view
        self.view_menu.add_command(label="Update TP View", command=self.update_tp_view)
        # a simple way to view the BLOB data in the database
        self.view_menu.add_command(label="View BLOB Data", command=self.app.view_blob_data)

        # Add 'Calc' menu
        self.calc_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Calc", menu=self.calc_menu)
        # Add 'Cm/Px' to the calc menu
        self.calc_menu.add_command(label="cm/Px", command=self.calc_cm_per_px)
        # calculate the MSI machine coordinate
        self.calc_menu.add_command(label="MSI Machine Coord", command=self.calc_msi_machine_coordinate)
        # calculate the transformation matrix
        self.calc_menu.add_command(label="Calc Transformation Matrix", command=self.app.calc_transformation_matrix)
        # convert the machine coordinate to real world coordinate
        self.calc_menu.add_command(label="Machine to Real World", command=self.app.machine_to_real_world)

        # Add a 'Dev' menu
        self.dev_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Dev", menu=self.dev_menu)
        # Add 'Reset tp' to the dev menu
        self.dev_menu.add_command(label="Reset TP", command=self.app.reset_tp)
        self.dev_menu.add_command(label="Set TP Size", command=self.app.set_tp_size)
        # lock all the images
        self.dev_menu.add_command(label="Lock All Images", command=self.app.lock_all)
        # move all teaching points to the top of the canvas
        self.dev_menu.add_command(label="Move All TPs to Top", command=self.app.move_all_tps_to_top)

        # Add 'Export' menu
        self.export_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Export", menu=self.export_menu)
        # Add 'Export TPs' to the export menu
        self.export_menu.add_command(label="Export TPs", command=self.export_tps)

    def calc_msi_machine_coordinate(self):
        for k, v in self.app.items.items():
            if isinstance(v, MsiImage):
                logging.debug(f"Calculating the MSI machine coordinate for {k}")
                v.update_tp_coords()

    def add_images(self):
        """Load the images from the file paths"""
        file_paths = filedialog.askopenfilenames()
        for file_path in file_paths:
            for k, v in self.app.items.items():
                try:
                    if v.path == file_path:
                        raise ValueError(f"{file_path} has already been loaded")
                except AttributeError:
                    pass
            if self.app.n_xray * self.app.n_linescan == 0:
                # let the user choose if input image is an xray image(x) or a linescan image(l) or an msi image (m)
                image_type = simpledialog.askstring("Image Type",
                                                    "Is this an xray image(x) or a linescan image(l) or an msi image "
                                                    "(m)?")
                if image_type == "x":
                    self.app.n_xray += 1
                    loaded_image = XrayImage.from_path(file_path)
                elif image_type == "l":
                    self.app.n_linescan += 1
                    loaded_image = LinescanImage.from_path(file_path)
                elif image_type == "m":
                    loaded_image = MsiImage.from_path(file_path)
                else:
                    raise ValueError("You need to choose an image type")
            else:
                loaded_image = MsiImage.from_path(file_path)
            logging.debug(f"Loaded image: {loaded_image}")
            loaded_image.create_im_on_canvas(self.app)
            self.app.items[loaded_image.tag] = loaded_image

    def quit(self):
        self.app.quit()

    def generate_tree_view(self):
        """generate a tree view using the teaching points, with the parent node being the image tag"""
        # clear the tree view
        self.app.tree.delete(*self.app.tree.get_children())
        # get the teaching points
        for k, v in self.app.items.items():
            try:
                if v.teaching_points is not None:
                    # add the image tag to the tree view
                    parent = self.app.tree.insert("", "end", text=k, values=("image", "", "", ""))
                    # add the teaching points to the tree view
                    for i, tp in v.teaching_points.items():
                        self.app.tree.insert(parent, "end", text=i, values=("", tp[0], tp[1], tp[2]))
            except AttributeError:
                pass

    def update_tp_view(self):
        # if the teaching points view does not exist, create one
        if not hasattr(self.app, 'tree'):
            logging.debug("Creating a treeview to display teaching points")
            # create a treeview to display teaching points
            self.app.tree_frame = tk.Frame(self.app)
            self.app.tree_frame.pack(side=tk.RIGHT, fill=tk.Y)
            self.app.tree = ttk.Treeview(self.app.tree_frame,
                                         columns=('label', 'x', 'y', 'd'),
                                         selectmode='browse')
            self.app.tree.heading('#0', text='img')
            self.app.tree.heading('label', text='label')
            self.app.tree.heading('x', text='x')
            self.app.tree.heading('y', text='y')
            self.app.tree.heading('d', text='d')
            self.app.tree.column('#0', width=100)
            self.app.tree.column('label', width=50)
            self.app.tree.column('x', width=50)
            self.app.tree.column('y', width=50)
            self.app.tree.column('d', width=50)
            self.app.tree.pack(side=tk.LEFT, fill=tk.Y)
            self.app.tree_visible = True
        # update the teaching points view
        self.generate_tree_view()

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
        if len(self.app.scale_line) < 2:
            raise ValueError("You need to draw two vertical lines to calculate the scale")
        elif len(self.app.scale_line) > 2:
            raise ValueError("You have drawn more than two vertical lines")
        else:
            # calculate the distance between the two scale lines
            pixel_distance = abs(
                self.app.canvas.coords(self.app.scale_line[1])[0] - self.app.canvas.coords(self.app.scale_line[0])[0])
            # calculate the distance in real world
            real_distance = simpledialog.askfloat("Real Distance", "Real Distance (cm):")
            # calculate the scale
            self.app.cm_per_pixel = real_distance / pixel_distance
            # create a text on the canvas to display the scale
            text = tk.Text(self.app.canvas, height=1, width=20)
            text.insert(tk.END, f"1cm = {pixel_distance / real_distance} pixel")
            text.config(state="disabled")
            self.app.canvas.create_window(100, 100, window=text, tags="cm_per_px_text")

    def export_tps(self):
        """Export the teaching points to a json file"""
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        if file_path:
            self.app.export_tps(file_path)
            print(f"Teaching points have been exported to {file_path}")
        else:
            print("No file path is given")
