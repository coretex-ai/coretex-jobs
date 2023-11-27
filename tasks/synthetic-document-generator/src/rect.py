import numpy as np

from .point import Point2D


class Rect:

    def __init__(self, tl: Point2D, tr: Point2D, br: Point2D, bl: Point2D) -> None:
        self.tl = tl
        self.tr = tr
        self.br = br
        self.bl = bl

    @property
    def width(self) -> int:
        return self.tr.x - self.tl.x

    @property
    def height(self) -> int:
        return self.bl.y - self.tl.y

    def numpy(self) -> np.ndarray:
        return np.array([
            [self.tl.x, self.tl.y],
            [self.tr.x, self.tr.y],
            [self.br.x, self.br.y],
            [self.bl.x, self.bl.y]
        ], dtype = np.float32)
