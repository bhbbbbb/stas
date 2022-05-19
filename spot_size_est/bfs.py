from argparse import Namespace
from typing import Tuple, NamedTuple, List, Union
from collections import deque
from torch import Tensor
from torchvision.transforms import functional as TF
import numpy as np

class C:
    WHITE = 255
    BLACK = 0


class Pos(NamedTuple):
    x: int
    y: int

    def left(self):
        return Pos(self.x - 1, self.y)
    
    def right(self):
        return Pos(self.x + 1, self.y)

    def up(self):
        return Pos(self.x, self.y + 1)
    
    def down(self):
        return Pos(self.x, self.y - 1)
    
    def adj_list(self):
        return [self.down(), self.right(), self.up(), self.left()]


class Spot(Namespace):
    color: int
    size: int
    start_pos: Tuple[int, int]
    def __init__(
        self,
        color: int,
        size: int,
        start_pos: Tuple[int, int],
    ):
        super().__init__()
        self.color = color
        self.size = size
        self.start_pos = start_pos
        return

NOT_FOUND_WHITE_SPOT = Spot(1, 0, Pos(-1, -1))

class BFS:
    spots: List[Spot]
    mask: np.ndarray
    MIN_WTHIE_FILL_THRESHOLD: int = 200

    def __init__(self, mask: Union[np.ndarray, Tensor]):
        if isinstance(mask, Tensor):
            mask = mask.numpy()
        self.mask = mask.squeeze(axis=0)
        self.found = np.zeros_like(self.mask, dtype=bool)
        return
    
    @classmethod
    def get_smallest_white_spot(cls, mask: Tensor, estimatation_level: int = 0):
        down_scale_factor = 2 ** estimatation_level
        if estimatation_level > 0:
            h, w = mask.squeeze(0).shape
            h, w = h // down_scale_factor, w // down_scale_factor
            mask = TF.resize(mask, [h, w], interpolation=TF.InterpolationMode.NEAREST)
        obj = cls(mask)
        spots = obj.start_bfs()
        smallest = 9999999999999
        smallest_spot = None
        for spot in spots:
            if spot.color > 0 and spot.size < smallest:
                smallest = spot.size
                smallest_spot = spot
        if smallest_spot is None:
            return NOT_FOUND_WHITE_SPOT
        smallest_spot.size *= down_scale_factor ** 2
        return smallest_spot
    

    @property
    def black_fill_threshold(self):
        h, w = self.mask.shape[-2:]
        return h * w // 10

    def fill_noise(self, white_threshold: int, print_spots: bool = False):
        self._fill_noise(C.BLACK, self.black_fill_threshold, print_spots)
        self._fill_noise(C.WHITE, max(self.MIN_WTHIE_FILL_THRESHOLD, white_threshold), print_spots)
        return self.mask

    def _fill_noise(self, noise_color: int, threshold: int, print_spots: bool = False):
        h, w = self.mask.shape[-2:]
        spots: List[Spot] = []
        spots_masks: List[np.ndarray] = []
        for i in range(h):
            for j in range(w):
                pos = Pos(i, j)
                if not self.found[pos]:
                    found_saved = np.copy(self.found)
                    spot = self.bfs(pos)
                    if spot.color == noise_color:
                        if print_spots:
                            print(spot)
                        spots.append(spot)
                        spots_masks.append(found_saved ^ self.found)
        
        color_to_fill = C.WHITE if noise_color == C.BLACK else C.BLACK
        if noise_color == C.BLACK:
            self.found = np.zeros_like(self.mask, dtype=bool) ## reinit found
        for spot_mask, spot in zip(spots_masks, spots):
            if spot.size < threshold:
                self.mask[spot_mask] = color_to_fill
            elif noise_color == C.BLACK:
                self.found[spot_mask] = True
        return self.mask
    
    def start_bfs(self):
        h, w = self.mask.shape[-2:]
        spots: List[Spot] = []
        for i in range(h):
            for j in range(w):
                pos = Pos(i, j)
                if not self.found[pos]:
                    spot = self.bfs(pos)
                    spots.append(spot)
        # for spot in spots:
        #     print(spot)
        return spots


    def is_pos_valid(self, pos: Pos):
        return (
            pos.x >= 0 and pos.y >= 0
            and pos.x < self.mask.shape[0] and pos.y < self.mask.shape[1]
        )
    
    def get_adj_list(self, pos: Pos):
        return [p for p in pos.adj_list() if self.is_pos_valid(p)]
    

    def bfs(self, pos: Pos):

        size = 1
        color = self.mask[pos]

        queue = deque()
        self.found[pos] = True
        queue.append(pos)


        while queue:
            p = queue.popleft()

            for adj in self.get_adj_list(p):
                if self.found[adj] or color != self.mask[adj]:
                    continue
                self.found[adj] = True
                size += 1
                queue.append(adj)
        
        return Spot(color, size, pos)
        