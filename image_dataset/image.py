from PIL import Image
import io
import ujson as json
import zstandard as zstd

class ImageData:
    def __init__(self, image : bytes, meta : bytes):
        self._img = Image.open(io.BytesIO(image), formats=["webp"])
        self._meta = json.loads(zstd.decompress(meta).decode("utf-8"))
    
    @property
    def img(self):
        return self._img
    
    @property
    def image(self):
        return self._img

    @property
    def meta(self):
        return self._meta
    
    @property
    def metadata(self):
        return self._meta

