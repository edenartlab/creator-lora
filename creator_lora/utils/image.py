import requests
from io import BytesIO
import  PIL.Image as PilImage

def load_pil_image(path_or_url: str):
    path_or_url = path_or_url
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        try:
            response = requests.get(path_or_url)
            pil_image = PilImage.open(BytesIO(response.content))
        except:
            raise Exception(f'Could not retrieve image from url:\n{path_or_url}')
    else:
        pil_image = PilImage.open(path_or_url)

    return pil_image

def split_pil_image_into_quadrants(image):
    """
    Split a PIL image into 4 quadrants and return them as a list.

    Parameters:
    - image: PIL Image object

    Returns:
    - List of 4 PIL Image objects representing the quadrants
    """
    # Get the width and height of the image
    width, height = image.size

    # Calculate the coordinates for the four quadrants
    left_upper = (0, 0, width // 2, height // 2)
    right_upper = (width // 2, 0, width, height // 2)
    left_lower = (0, height // 2, width // 2, height)
    right_lower = (width // 2, height // 2, width, height)

    # Crop the image to get the four quadrants
    quadrant1 = image.crop(left_upper)
    quadrant2 = image.crop(right_upper)
    quadrant3 = image.crop(left_lower)
    quadrant4 = image.crop(right_lower)

    # Return the quadrants as a list
    return [quadrant1, quadrant2, quadrant3, quadrant4]