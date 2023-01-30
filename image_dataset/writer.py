import contextlib
import io
import numpy as np
import zstandard as zstd
import os
import struct
from PIL import Image
from typing import Any
import ujson as json

class ImageDatasetBuilder:
    def __init__(self, path : str, chunk_size : int) -> None:
        self._path = path
        self._block_index = 0
        self._inblock_index = 0
        self._buffer = io.BytesIO()
        self._index = np.full((chunk_size,), -1, dtype=np.int64)
        self._chunk_size = chunk_size
    
    def _close_block(self):
        val = self._buffer.getvalue()
        if len(val) == 0:
            return
        buf_index = zstd.compress(self._index.tobytes(), level=10)
        len_index = len(buf_index)

        if not os.path.exists(self._path):
            os.makedirs(self._path)
        with open(
            os.path.join(self._path, "data_{}.bin".format(self._block_index)),
            "wb"
        ) as fp:
            fp.write(struct.pack("<I", len_index))
            fp.write(buf_index)
            fp.write(val)
        self._block_index += 1
        self._inblock_index = 0
        self._buffer = io.BytesIO()
        self._index = np.full((self._index.shape[0],), -1, dtype=np.int64)

    def close(self):
        self._close_block()
    
    def write(self, image : Image.Image, meta : Any):
        self._index[self._inblock_index] = self._buffer.tell()
        self._inblock_index += 1
        img_buf = io.BytesIO()
        image.save(img_buf, format="webp", quality=100)
        data_val = img_buf.getvalue()
        meta_val = zstd.compress(json.dumps(meta, ensure_ascii=False).encode("utf-8"), level=10)

        len_data = len(data_val)
        len_meta = len(meta_val)

        self._buffer.write(struct.pack("<II", len_meta, len_data))
        self._buffer.write(meta_val)
        self._buffer.write(data_val)

        if self._inblock_index == self._chunk_size:
            self._close_block()


@contextlib.contextmanager
def make_dataset(path : str, chunk_size : int = 65536):
    builder = ImageDatasetBuilder(path, chunk_size)
    try:
        yield builder
    finally:
        builder.close()

