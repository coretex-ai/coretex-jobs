class Point2D:

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(
            self.x + other.x,
            self.y + other.y
        )

    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(
            self.x - other.x,
            self.y - other.y
        )
