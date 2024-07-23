import math

def dist_2D(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    # The following is more convenient, but slower.
    # return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

def dist_2_segment(x, y, x1, y1, x2, y2):
    area_2 = abs(x * (y1 - y2) + x1 * (y2 - y) + x2 * (y - y1))
    return area_2 / dist_2D(x1, y1, x2, y2)

INF = 1 << 20

class DistItem():
    def __init__(self, x1, y1, r, dtype, x2=INF, y2=INF, r2=0) -> None:
        '''
        If `dtype` == 'Euclid':
            - `(x1, y1)`: the center point
            - `r`: the maximum radium / distance
            - `(x2, y2)`: default `(INF, INF)`. If not, it indicates this item
            represents a segment with `(x1, y1)` and `(x2, y2)` as endpoints

        If `dtype` == 'Chebyshev':
            - `(x1, y1)`: the center point
            - `r, r2`: the maximum distance along x-axis and y-axis separately
            - `(x2, y2)`: default `(INF, INF)`. If not, it indicates this item
            represents a horizontal or vertical segment with `(x1, y1)` and
            `(x2, y2)` as endpoints
        '''
        self.x1 = x1
        self.y1 = y1
        self.r = r
        assert dtype == 'Euclid' or dtype == 'Chebyshev'
        self.dtype = dtype

        self.x2 = x2
        self.y2 = y2
        self.r2 = r2

    def check_dist(self, x, y, lamda) -> bool:
        if self.dtype == 'Euclid':
            if self.x2 == INF:
                dist = dist_2D(x, y, self.x1, self.y1)
            else:
                dist = dist_2_segment(x, y, self.x1, self.y1, self.x2, self.y2)
            return dist < self.r * lamda
        else:
            # In the case of self.dtype == 'Chebyshev', distance along x-axis
            # and y-axis will be calculated separately
            if self.x2 == INF:
                dist_x = abs(x - self.x1)
                dist_y = abs(y - self.y1)
                return dist_x < self.r * lamda and dist_y < self.r2 * lamda
            else:
                # For vertical segment, only the distance to it from a point in
                # the corresponding horizontal region will be calculated
                # Similar for horizontal segment
                if self.x1 == self.x2:
                    dist = abs(x - self.x1) if self.y1 <= y <= self.y2 or self.y2 <= y <= self.y1 else INF
                    return dist < self.r * lamda
                else:
                    dist = abs(y - self.y1) if self.x1 <= x <= self.x2 or self.x2 <= x <= self.x1 else INF
                    return dist < self.r2 * lamda
