import numpy as np

class NotSoCleverCreator:
    def __init__(self,
                 canvas_size=(64, 64),
                 center_region=(56, 56),
                 square_size=(9, 9)):
        self.canvas_size = canvas_size
        self.center_region = center_region
        self.square_size = square_size
        self._calc_coords()
        self.indices = None

    def _calc_coords(self):
        xx_start = (self.canvas_size[0] - self.center_region[0]) // 2
        xx_end = self.center_region[0] + xx_start
        xx_indices = np.arange(xx_start, xx_end + 1)

        yy_start = (self.canvas_size[1] - self.center_region[1]) // 2
        yy_end = self.center_region[1] + yy_start
        yy_indices = np.arange(yy_start, yy_end + 1)

        self.coords = np.reshape(
            np.stack(np.meshgrid(xx_indices, yy_indices), -1),
            (-1, 2)).astype('float32')

    @classmethod
    def one_hot_coords(cls, canvas_size, coords):
        coords = coords.astype('int32')
        n_coords = coords.shape[0]
        one_hot = np.zeros((n_coords, *canvas_size))
        one_hot[list(range(n_coords)), coords[:, 0], coords[:, 1]] = 1
        one_hot = np.expand_dims(one_hot, -1)
        return one_hot.astype('float32')

    def quadrant_split(self):
        mid_x = self.canvas_size[0] // 2
        mid_y = self.canvas_size[1] // 2
        test_indices = (self.coords[:, 0] >= mid_x) & (self.coords[:, 1] >= mid_y)
        train_coords = self.coords[~test_indices]
        test_coords = self.coords[test_indices]
        return train_coords, test_coords

    def uniform_split(self):
        split = int(0.75 * self.coords.shape[0])
        if self.indices is None:
            self.indices = np.random.permutation(self.coords.shape[0])
        train_indices = self.indices[:split]
        test_indices = self.indices[split:]

        return self.coords[train_indices], self.coords[test_indices]
