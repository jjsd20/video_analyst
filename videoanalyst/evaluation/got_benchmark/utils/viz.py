from __future__ import absolute_import

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

fig_dict = {}
patch_dict = {}


def show_frame(image,
               boxes=None,
               fig_n=1,
               pause=0.001,
               linewidth=3,
               cmap=None,
               colors=None,
               legends=None):
    r"""Visualize an image w/o drawing rectangle(s).
    
    Args:
        image (numpy.ndarray or PIL.Image): Image to show.
        boxes (numpy.array or a list of numpy.ndarray, optional): A 4 dimensional array
            specifying rectangle [left, top, width, height] to draw, or a list of arrays
            representing multiple rectangles. Default is ``None``.
        fig_n (integer, optional): Figure ID. Default is 1.
        pause (float, optional): Time delay for the plot. Default is 0.001 second.
        linewidth (int, optional): Thickness for drawing the rectangle. Default is 3 pixels.
        cmap (string): Color map. Default is None.
        color (tuple): Color of drawed rectanlge. Default is None.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[..., ::-1])

    if not fig_n in fig_dict or \
        fig_dict[fig_n].get_size() != image.size[::-1]:
        fig = plt.figure(fig_n)
        plt.axis('off')
        fig.tight_layout()
        fig_dict[fig_n] = plt.imshow(image, cmap=cmap)
    else:
        fig_dict[fig_n].set_data(image)

    if boxes is not None:
        if not isinstance(boxes, (list, tuple)):
            boxes = [boxes]

        if colors is None:
            colors = ['r', 'g', 'b', 'c', 'm', 'y'] + \
                list(mcolors.CSS4_COLORS.keys())
        elif isinstance(colors, str):
            colors = [colors]

        if not fig_n in patch_dict:
            patch_dict[fig_n] = []
            for i, box in enumerate(boxes):
                patch_dict[fig_n].append(
                    patches.Rectangle((box[0], box[1]),
                                      box[2],
                                      box[3],
                                      linewidth=linewidth,
                                      edgecolor=colors[i % len(colors)],
                                      facecolor='none',
                                      alpha=0.7 if len(boxes) > 1 else 1.0))
            for patch in patch_dict[fig_n]:
                fig_dict[fig_n].axes.add_patch(patch)
        else:
            for patch, box in zip(patch_dict[fig_n], boxes):
                patch.set_xy((box[0], box[1]))
                patch.set_width(box[2])
                patch.set_height(box[3])

        if legends is not None:
            fig_dict[fig_n].axes.legend(patch_dict[fig_n],
                                        legends,
                                        loc=1,
                                        prop={'size': 8},
                                        fancybox=True,
                                        framealpha=0.5)

    plt.pause(pause)
    plt.draw()


def show_frame_with_boxes(image,
                          boxes=None,
                          pause=0.001,
                          linewidth=3,
                          colors=None,
                          legends=None):
    r"""Visualize an image with optional rectangle(s) using OpenCV.

    Args:
        image (numpy.ndarray or PIL.Image): Image to show.
        boxes (numpy.array or a list of numpy.ndarray, optional): A 4 dimensional array
            specifying rectangle [left, top, width, height] to draw, or a list of arrays
            representing multiple rectangles. Default is ``None``.
        pause (float, optional): Time delay for the plot. Default is 0.001 second.
        linewidth (int, optional): Thickness for drawing the rectangle. Default is 3 pixels.
        colors (list, optional): List of colors for the rectangles. Default is None.
        legends (list, optional): List of legends for the rectangles. Default is None.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Convert image from BGR to RGB if needed
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if colors is None:
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)] + \
                 [(255, 255, 255)] * (len(boxes) - 6)  # Default colors if not provided
    elif isinstance(colors, str):
        colors = [colors]

    if boxes is not None:
        if not isinstance(boxes, (list, tuple)):
            boxes = [boxes]

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            # Convert color from BGR to RGB if needed
            if isinstance(color, str):
                color = tuple(
                    int(color.lstrip('#')[i:i + 2], 16)
                    for i in (0, 2, 4))[::-1]
            elif isinstance(color, tuple):
                color = color[::-1]  # Convert to BGR for OpenCV

            # Draw the rectangle
            box = [int(b) for b in box]
            cv2.rectangle(image, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), color, linewidth)

    # Display the image
    cv2.imshow('Frame', image)
    cv2.waitKey(int(pause * 1000))  # Convert pause from seconds to milliseconds


'''
 # Optionally add legends (not directly supported in OpenCV, so we draw text)
    if legends is not None:
        if not isinstance(legends, (list, tuple)):
            legends = [legends]

        for i, legend in enumerate(legends):
            box = boxes[i]
            color = colors[i % len(colors)]
            # Convert color from BGR to RGB if needed
            if isinstance(color, str):
                color = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[::-1]
            elif isinstance(color, tuple):
                color = color[::-1]  # Convert to BGR for OpenCV

            # Draw the legend text
            text_position = (box[0], box[1] - 10)
            cv2.putText(image, legend, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the image again to show the text
    cv2.imshow('Frame', image)
    cv2.waitKey(int(pause * 1000))  # Convert pause from seconds to milliseconds


'''

# Example usage
if __name__ == "__main__":
    # Create a sample image
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    # Define some sample boxes
    boxes = [[50, 50, 100, 100], [150, 150, 150, 150]]

    # Define some sample colors
    colors = ['green', 'red']

    # Define some sample legends
    legends = ['Box 1', 'Box 2']

    # Show the image with boxes
    show_frame(image, boxes, colors=colors, legends=legends)
