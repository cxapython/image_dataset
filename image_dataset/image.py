import io

import filetype
import ujson as json
import zstandard as zstd
from PIL import Image


class ImageData:
    def __init__(self, file_name, image: bytes, meta: bytes):
        kind = filetype.guess(image)
        self._img_type = kind.extension
        self._file_name = file_name
        self._img = Image.open(io.BytesIO(image))
        self._meta = json.loads(zstd.decompress(meta).decode("utf-8"))

    @property
    def img_type(self):
        return self._img_type

    @property
    def file_name(self):
        return self._file_name

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
