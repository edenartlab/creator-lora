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