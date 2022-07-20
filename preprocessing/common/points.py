
import numpy as np


class ObjectPoints:
    def __init__(self,
                 points,
                 num_points,
                 segmentation,
                 surrounding_context_factor=1.25):
        """

        Args:
            points (list<np.ndarray>):
            num_points (list<int>):
            segmentation (list<np.ndarray>):
            surrounding_context (float):
        """
        self.points = points
        self.num_points = num_points
        self.segmentation = segmentation
        self.surrounding_context_factor = surrounding_context_factor

        # run dtype checks
        for i in range(len(self.points)):
            assert isinstance(self.points[i], np.ndarray)
            assert isinstance(self.segmentation[i], np.ndarray)
            if self.segmentation[i].shape != (0,):
                assert len(self.segmentation[i].shape) == 2
                assert self.points[i].shape[0] == self.segmentation[i].shape[0]