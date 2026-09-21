"""Bounded, process-local reuse of aligned raster readers."""

import os
from collections import OrderedDict
from contextlib import contextmanager

from data.raster_alignment import aligned_raster


class RasterReaderPool:
    """Keep VRT metadata open without loading complete rasters into memory.

    Spawned workers receive an empty pool. Forked workers discard inherited
    readers before use. The limit includes both dynamic and static sources.
    """

    def __init__(self, grid, max_items=0):
        if max_items < 0:
            raise ValueError("max_open_rasters must be nonnegative")
        self.grid = grid
        self.max_items = max_items
        self._pid = os.getpid()
        self._readers = OrderedDict()

    def __getstate__(self):
        return {
            "grid": self.grid,
            "max_items": self.max_items,
            "_pid": None,
            "_readers": OrderedDict(),
        }

    @contextmanager
    def borrow(self, path, resampling):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if not self.max_items:
            with aligned_raster(path, self.grid, resampling=resampling) as reader:
                yield reader
            return
        key = (str(path), str(resampling))
        if key not in self._readers:
            if len(self._readers) >= self.max_items:
                _, context = self._readers.popitem(last=False)
                context.__exit__(None, None, None)
            context = aligned_raster(path, self.grid, resampling=resampling)
            context.__enter__()
            self._readers[key] = context
        self._readers.move_to_end(key)
        yield self._readers[key].dataset

    def close(self):
        for context in self._readers.values():
            context.__exit__(None, None, None)
        self._readers.clear()

    def __del__(self):
        if hasattr(self, "_readers"):
            self.close()
