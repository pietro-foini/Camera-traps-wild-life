import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def compose_polygon(x: int, y: int, width: int, height: int) -> Polygon:
    """
    Create a shapely polygon (rectangle) starting from coordinates.

    :param x: the x coordinate of the upper left corner
    :param y: the y coordinate of the upper left corner
    :param width: the width of the rectangle
    :param height: the height of the rectangle
    :return: the shapely related polygon
    """
    x_min, y_min = x, y
    x_max, y_max = x + width, y + height
    # Create polygon.
    polygon = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])

    return polygon


def decompose_polygon(polygon: Polygon) -> tuple[int, int, int, int]:
    """
    Get the coordinates starting from a shapely polygon (rectangle).

    :param polygon: a shapely polygon representing a rectangle
    :return: the x and y coordinates of the upper left corner, the width and the height of the related rectangle
    """
    x_min, y_min, x_max, y_max = polygon.bounds
    # Compute width and height.
    width = x_max - x_min
    height = y_max - y_min
    # Compute the initial coordinates (x, y) of the box.
    x = x_min
    y = y_min

    return int(x), int(y), int(width), int(height)


def get_bbox_without_intersection(boxes: list[Polygon]) -> list[tuple[int, int, int, int]]:
    """
    Get the minimum bounding boxes that do not intersect each other from a list of input polygons.

    :param boxes: a list of shapely polygons
    :return: the x and y coordinates of the upper left corner, the width and the height of the minimum bounding boxes
    """
    # Merge all the boxes.
    polygons = unary_union(boxes)

    # Get all the polygons after the union.
    if polygons.geom_type == "MultiPolygon":
        polygons = list(polygons.geoms)
    else:
        polygons = [polygons]

    # Get minimum rotated rectangle.
    contours = [decompose_polygon(p.minimum_rotated_rectangle) for p in polygons if not p.is_empty]

    return contours


def crop_random_bbox(image: np.array, min_area: int) -> np.array:
    """
    Crop a random bounding box from the image while ensuring the area is not less than the specified minimum area.

    :param image: the input image
    :param min_area: the minimum area constraint for the bounding box
    :return: the cropped image representing the random bounding box
    """
    # Get image dimensions.
    height, width = image.shape[:2]

    while True:
        # Generate random top-left and bottom-right coordinates.
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(x1, width)
        y2 = np.random.randint(y1, height)

        # Calculate bounding box area.
        bbox_area = (x2 - x1) * (y2 - y1)

        if bbox_area >= min_area:
            # Crop the bounding box from the image.
            cropped_image = image[y1:y2, x1:x2]
            return cropped_image


def expand_bbox(x: int, y: int, width: int, height: int, percentage: float) -> tuple[int, int, int, int]:
    """
    Expand a box by a given percentage.

    :param x: the x-coordinate of the upper left corner of the box
    :param y: the y-coordinate of the upper left corner of the box
    :param width: the width of the box
    :param height: the height of the box
    :param percentage: the expansion percentage
    :return: a tuple containing the new x-coordinate, y-coordinate, width, and height of the expanded box.
    """
    # Calculate the expansion amount for width and height
    width_expansion = width * percentage / 100
    height_expansion = height * percentage / 100

    # Calculate the new dimensions
    new_x = int(max(x - width_expansion / 2, 0))
    new_y = int(max(y - height_expansion / 2, 0))
    new_w = int(width + width_expansion)
    new_h = int(height + height_expansion)

    return new_x, new_y, new_w, new_h
