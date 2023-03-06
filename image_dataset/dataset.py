
from typing import Iterator
import os
import numpy as np
import zstandard as zstd
from .image import ImageData
import struct
from functools import lru_cache

class DatasetBlockIterator:
    def __init__(self, block : "DatasetBlock") -> None:
        self._block = block
        self._fp = open(block.path, "rb")
        self._fp.seek(block._index_offset, os.SEEK_SET)
        self.end_fp = os.path.getsize(self._block.path) - (4 + self._block.len_index)
    
    def __next__(self) -> ImageData:
        if self._fp.tell() == self.end_fp:
            raise StopIteration()
        tmp = self._fp.read(8)
        if len(tmp) == 0:
            raise StopIteration()
        elif len(tmp) < 8:
            raise IOError("Unexpected end of file")
        
        len_meta, len_data = struct.unpack("<II", tmp)
        meta = self._fp.read(len_meta)
        if len(meta) < len_meta:
            raise IOError("Unexpected end of file")
        data = self._fp.read(len_data)
        if len(data) < len_data:
            raise IOError("Unexpected end of file")
        return ImageData(data, meta)



class DatasetBlock:
    def __init__(self, path : str):
        fp = open(path, "rb")
        fp.seek(-4, 2)  #文件倒数第4个字节
        self.len_index = struct.unpack("<I", fp.read(4))[0]
        fp.seek(-(4+self.len_index), 2)  #索引开始的位置
        self._index = np.frombuffer(zstd.decompress(fp.read(self.len_index)), dtype=np.int64)
        self._index_offset = 0
        self._fp = fp
        self.path = path
        self._len = (self._index >= 0).sum()

    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, index : int) -> ImageData:
        if index < 0 or index >= self._index.shape[0]:
            raise IndexError("Index out of range")
        index_st = self._index[index]
        if index_st < 0:
            raise IndexError("Index out of range")
        self._fp.seek(self._index_offset + index_st, os.SEEK_SET)
        len_meta, len_data = struct.unpack("<II", self._fp.read(8))
        meta = self._fp.read(len_meta)
        data = self._fp.read(len_data)
        return ImageData(data, meta)

    def __iter__(self) -> DatasetBlockIterator:
        return DatasetBlockIterator(self)


class ImageDataset:
    def __init__(self, path : str, chunk_size : int = 65536):
        self._prefix = path
        self._chunk_size = chunk_size
        self._num_files = 0
        self._len = 0
        while os.path.exists(self._get_data_path(self._num_files)):
            self._num_files += 1
        
        if self._num_files > 0:
            self._len = self._chunk_size * (self._num_files - 1)
            self._len += len(self._get_block(self._num_files - 1))
    
    def _get_data_path(self, index : int) -> str:
        return os.path.join(self._prefix, "data_{}.bin".format(index))
    
    @lru_cache(maxsize=32)
    def _get_block(self, index : int) -> DatasetBlock:
        return DatasetBlock(self._get_data_path(index))
    
    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, index : int) -> ImageData:
        if index < 0:
            index = index % self._len
        if index >= self._len:
            raise IndexError("Index out of range")
        block_index = index // self._chunk_size
        block = self._get_block(block_index)
        return block[index % self._chunk_size]
    
    def __iter__(self) -> Iterator[ImageData]:
        for i in range(self._num_files):
            for img in self._get_block(i):
                yield img
