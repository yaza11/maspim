import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import logging

from matplotlib.backend_bases import MouseButton

logger = logging.getLogger('msi_workflow.' + __name__)


class InteractiveImage:
    def __init__(self, image, mode):
        self.image = image.copy()

        self.x_pixels = []
        self.y_pixels = []
        self.x_data = []
        self.y_data = []

        self.previous_backend = matplotlib.get_backend()
        matplotlib.use('QtAgg')

        # short side is 5 units long
        short_side = .5
        scale = short_side / min(image.shape[:2])
        self.fig, self.ax = plt.subplots(
            figsize=(image.shape[1] * scale, image.shape[0] * scale),
            num='Add point w/ left mouse, del w/ right mouse, finish with middle mouse'
        )
        self.canvas = self.fig.canvas

        if mode == "line":
            on_click_f = self.on_click_line
            self.update = self.update_line
        elif mode == "rect":
            on_click_f = self.on_click_rect
            self.update = self.update_rect
        elif mode == "punchholes":
            on_click_f = self.on_click_punch
            self.update = self.update_punch
        else:
            raise ValueError()

        self.canvas.mpl_connect('button_press_event', on_click_f)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # Draw the image on the axes
        self.ax.imshow(self.image)
        self.ax.axis('off')

    def _add_coords(self, event):
        self.x_pixels.append(event.x)
        self.y_pixels.append(event.y)

        self.x_data.append(event.xdata)
        self.y_data.append(event.ydata)

    def _remove_coords(self):
        self.x_pixels.pop()
        self.y_pixels.pop()

        self.x_data.pop()
        self.y_data.pop()

    def on_click_line(self, event):
        # add point to line
        if event.button is MouseButton.LEFT:
            self._add_coords(event)
            self.update_line()

        # remove point from line
        elif event.button is MouseButton.RIGHT:
            if self.x_pixels:
                self._remove_coords()
                self.update_line()
        # close line
        elif (event.button is MouseButton.MIDDLE) and (len(self.x_pixels) >= 2):
            self.x_pixels.append(self.x_pixels[0])
            self.y_pixels.append(self.y_pixels[0])

            self.x_data.append(self.x_data[0])
            self.y_data.append(self.y_data[0])
            self.update_line()
            self.close_line()

    def on_click_rect(self, event):
        if event.button is MouseButton.LEFT:
            self._add_coords(event)
            self.update_rect()
        # remove point from line
        elif event.button is MouseButton.RIGHT:
            if self.x_pixels:
                self._remove_coords()
                self.update_rect()
        elif (event.button is MouseButton.MIDDLE) and (len(self.x_pixels) >= 2):
            self.canvas.close()
            # restore backend
            matplotlib.use(self.previous_backend)

    def on_click_punch(self, event):
        if event.button is MouseButton.LEFT:
            self._add_coords(event)
            self.update_punch()
        # remove point from line
        elif event.button is MouseButton.RIGHT:
            if self.x_pixels:
                self._remove_coords()
                self.update_punch()
        elif (event.button is MouseButton.MIDDLE) and (len(self.x_pixels) >= 2):
            self.canvas.close()
            # restore backend
            matplotlib.use(self.previous_backend)

    def update_line(self):
        # clear ax
        self.ax.clear()
        # redraw
        self.ax.imshow(self.image)
        # add line
        self.ax.plot(self.x_data, self.y_data)
        # update
        self.canvas.draw()

    def close_line(self):
        self.ax.fill(self.x_data, self.y_data)
        self.canvas.draw()

        # empty lists
        self.x_pixels.clear()
        self.y_pixels.clear()

        self.x_data.clear()
        self.y_data.clear()

    def update_rect(self):
        # Clear ax
        self.ax.clear()
        # Redraw the image
        self.ax.imshow(self.image)

        # Calculate the width and height of the rectangle
        if len(self.x_data) == 2:
            x1, x2 = min(self.x_data), max(self.x_data)
            y1, y2 = min(self.y_data), max(self.y_data)
            width = x2 - x1
            height = y2 - y1

            # Add the rectangle patch to the axes
            rect = patches.Rectangle(
                (x1, y1),
                width=width,
                height=height,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            self.ax.add_patch(rect)

        # Update the canvas
        self.canvas.draw()

    def update_punch(self):
        def add_square(center_x, center_y, length):
            s = patches.Rectangle(
                (center_x - length / 2, center_y - length / 2),
                width=length,
                height=length,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            self.ax.add_patch(s)

        # Clear ax
        self.ax.clear()
        # Redraw the image
        self.ax.imshow(self.image)

        # add square shaped dots
        scale = 5 / self.image.shape[1]  # assuming image is 5 cm long
        self._punchhole_size = .2 / scale  # assuming hole is .5 cm wide

        [
            add_square(self.x_data[i], self.y_data[i], self._punchhole_size)
            for i in range(len(self.x_data))
        ]

        self.canvas.draw()

    def show(self):
        logger.debug('inside show')
        plt.show(block=True)
        logger.debug('no longer blocked')
        matplotlib.use(self.previous_backend)
        logger.debug(f"using backend {self.previous_backend}")


if __name__ == "__main__":
    # Example usage
    image = ...  # Your image data
    interactive_image = InteractiveImage(image)
    interactive_image.show()
