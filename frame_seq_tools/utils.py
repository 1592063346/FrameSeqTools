import numpy as np
import math
import random

import cv2

from frame_seq_tools.tools import DistItem, dist_2D, dist_2_segment

class FrameSeqTools():
    '''
    Frame sequence merge functions:
        - `forced_merge`
        - `gradual_transition_merge`
        - `push_transition_merge`
        - `wipe_transition_merge`
        - `frame_crop_split`

    Data enhancement functions:
        - `frame_darken`
        - `frame_glare`
        - `frame_extract`
        - `frame_crop`

    The shape of frames is `(frame_num, height, width, 3)`.
    '''
    def __init__(self, frames1: np.array, frames2: np.array = np.array([])) -> None:
        self.frames1 = frames1
        self.frames2 = frames1 if frames2.shape == (0, ) else frames2
        assert self.frames1.ndim == 4 and self.frames1.shape[3] == 3
        assert self.frames2.ndim == 4 and self.frames2.shape[3] == 3

    def __include(self, x, y, lamda, item_list):
        for item in item_list:
            if item.check_dist(x, y, lamda):
                return True
        return False

    def __gradual(self, frames1, frames2, type):
        # -1 for random
        if type == -1:
            type = random.randint(0, 2)

        frame_num = frames1.shape[0]

        if type == 0:
            res = np.zeros(frames1.shape)
            for i in range(frame_num):
                lamda = (i + 1) / (frame_num + 1)
                res[i] = frames1[i] * (1 - lamda) + frames2[i] * lamda

            return np.rint(res).astype(np.uint8), frame_num
        else:
            # around 0.5 ~ 1.2s when fps == 25
            # full transition will be:
            #   1. frame1 -> pure plot, `frame_num` frames
            #   2. pure plot,           `extra_frame_num` frames
            #   3. pure plot -> frame2, `frame_num` frames
            extra_frame_num = random.randint(12, 30)
            pure_plot = np.full(frames1.shape[1:], 0x00 if type == 1 else 0xFF).astype(np.uint8)

            slice1 = np.zeros(frames1.shape)
            slice2 = np.zeros(frames1.shape)
            for i in range(frame_num):
                lamda = (i + 1) / (frame_num + 1)
                slice1[i] = frames1[i] * (1 - lamda) + pure_plot * lamda
                slice2[i] = frames2[i] * lamda + pure_plot * (1 - lamda)

            return np.concatenate((
                np.rint(slice1).astype(np.uint8),
                np.array([pure_plot] * extra_frame_num),
                np.rint(slice2).astype(np.uint8)
            )), frame_num * 2 + extra_frame_num

    def __push(self, frames1, frames2, type):
        coef = [
            [1, 0], [-1, 0], [0, -1], [0, 1],
            [1, -1], [1, 1], [-1, 1], [-1, -1]
        ]
        # -1 for random
        if type == -1:
            type = random.randint(0, 7)

        frame_num = frames1.shape[0]
        height = frames1.shape[1]
        width = frames1.shape[2]

        res = np.zeros(frames1.shape).astype(np.uint8)
        for i in range(frame_num):
            lamda = (frame_num - i) / (frame_num + 1)
            for j in range(height):
                for k in range(width):
                    new_j = round(j - coef[type][0] * lamda * height)
                    new_k = round(k - coef[type][1] * lamda * width)
                    if 0 <= new_j < height and 0 <= new_k < width:
                        res[i][j][k] = frames2[i][new_j][new_k]
                    else:
                        res[i][j][k] = frames1[i][j][k]

        return res

    def __wipe(self, frames1, frames2, type):
        # -1 for random
        if type == -1:
            type = random.randint(0, 9)

        frame_num = frames1.shape[0]
        height = frames1.shape[1]
        width = frames1.shape[2]

        match (type >> 1):
            case 0:
                cx = (height - 1) / 2
                cy = (width - 1) / 2
                item_list = [DistItem(cx, cy, dist_2D(cx, cy, 0, 0), 'Euclid')]
            case 1:
                grid_n = random.randint(6, 10)
                rx = height / grid_n / 2
                ry = width / grid_n / 2
                cxs = np.linspace(0, height, grid_n + 1)[1:] - rx
                cys = np.linspace(0, width, grid_n + 1)[1:] - ry
                item_list = [DistItem(cxs[i], cys[j], rx, 'Chebyshev', r2=ry)
                             for i in range(grid_n) for j in range(grid_n)]
            case 2:
                stripe_n = random.randint(6, 10)
                ry = width / stripe_n / 2
                cys = np.linspace(0, width, stripe_n + 1)[1:] - ry
                item_list = [DistItem(0, cys[i], ry, 'Euclid', x2=(height - 1), y2=cys[i])
                             for i in range(stripe_n)]
            case 3:
                stripe_n = random.randint(6, 10)
                pys = np.linspace(0, width, stripe_n + 1)
                r = dist_2_segment((height - 1) / 2, 0, 0, 0, height - 1, pys[1])
                item_list = [DistItem(0, pys[i], r, 'Euclid', x2=(height - 1), y2=pys[i + 1])
                             for i in range(stripe_n)]
                item_list += [DistItem(height - 1, pys[i], r, 'Euclid', x2=0, y2=pys[i + 1])
                              for i in range(stripe_n)]
            case 4:
                line_n = random.randint(6, 10)
                ex_width = (random.random() * 0.6 + 0.3) * width * line_n
                pxs = np.linspace(0, height - 1, line_n + 1)
                pys = np.linspace(0, -ex_width, line_n)
                item_list = [DistItem(pxs[i], pys[i], 0, 'Chebyshev', pxs[i + 1], pys[i], width + ex_width)
                             for i in range(line_n)]

        res = np.zeros(frames1.shape).astype(np.uint8)
        # Optimize run time in some way, due to a position keeps after it has been wiped
        wiped = np.full((height, width), False)
        for i in range(frame_num):
            lamda = (i + 1) / (frame_num + 1)
            for j in range(height):
                for k in range(width):
                    if not wiped[j][k]:
                        if (type & 1) == 0:
                            wiped[j][k] = self.__include(j, k, lamda, item_list)
                        else:
                            wiped[j][k] = not self.__include(j, k, 1 - lamda, item_list)
                    res[i][j][k] = frames2[i][j][k] if wiped[j][k] else frames1[i][j][k]

        return res

    def __transition_kernel(
        self,
        transition_type: str = '',
        transition_frame_num: int = 0,
        transition_arg: int = 0
    ):
        # -1 for random
        if transition_frame_num == -1:
            # around 0.2 ~ 1s when fps == 25
            transition_frame_num = random.randint(5, 25)

        if transition_type in ['gradual', 'push', 'wipe']:
            frame_slice1 = self.frames1[-(transition_frame_num):]
            frame_slice2 = self.frames2[0:transition_frame_num]
            len1 = self.frames1.shape[0] - transition_frame_num
            len2 = self.frames2.shape[0] - transition_frame_num
            match transition_type:
                case 'gradual':
                    # update transition_frame_num because an extra slice of black or white plots will be added
                    transition_frames, transition_frame_num = self.__gradual(frame_slice1, frame_slice2, transition_arg)
                case 'push':
                    transition_frames = self.__push(frame_slice1, frame_slice2, transition_arg)
                case 'wipe':
                    transition_frames = self.__wipe(frame_slice1, frame_slice2, transition_arg)
        else:
            len1 = self.frames1.shape[0]
            len2 = self.frames2.shape[0]
            transition_frames = np.array([]).reshape((0, ) + self.frames1.shape[1:]).astype(np.uint8)
            transition_frame_num = 0

        one_hot = np.zeros(len1 + len2 + transition_frame_num, ).astype(np.uint8)
        one_hot[(2 * len1 + transition_frame_num - 1) // 2] = 1
        multi_hot = np.concatenate((
            np.zeros(len1 - 1, ).astype(np.uint8),
            np.ones(transition_frame_num + 1, ).astype(np.uint8),
            np.zeros(len2, ).astype(np.uint8)
        ))

        return np.concatenate((self.frames1[0:len1], transition_frames, self.frames2[-len2:])), one_hot, multi_hot

    def forced_merge(self):
        return self.__transition_kernel()

    def gradual_transition_merge(self, gradual_frame_num=-1, gradual_type=-1):
        '''
        `gradual_type`:
            - 0: normal
            - 1: transition via black plots
            - 2: transition via white plots
            - -1: random
        '''
        return self.__transition_kernel(
                transition_type='gradual',
                transition_frame_num=gradual_frame_num,
                transition_arg=gradual_type
            )

    def push_transition_merge(self, push_frame_num=-1, push_type=-1):
        '''
        `push_type`:
            - 0~3: in order [↑, ↓, →, ←]
            - 4~7: in order [↗, ↖, ↙, ↘]
            - -1: random
        '''
        return self.__transition_kernel(
                transition_type='push',
                transition_frame_num=push_frame_num,
                transition_arg=push_type
            )

    def wipe_transition_merge(self, wipe_frame_num=-1, wipe_type=-1):
        '''
        `wipe_type`:
        If wipe_type is even, the new shot is INSIDE the shape described below;
        if wipe_type is odd, it's OUTSIDE (animation of shape is reversed accordingly).
            - 0 | 1: a circle expanding from the center
            - 2 | 3: an expanding grid
            - 4 | 5: some expanding vertical stripes
            - 6 | 7: some expanding cross stripes
            - 8 | 9: some horizontal stripes wiped sequentially at equal time intervals
            - -1: random
        '''
        return self.__transition_kernel(
                transition_type='wipe',
                transition_frame_num=wipe_frame_num,
                transition_arg=wipe_type
            )

    def get_window(self, type, ratio):
        r = ratio / 2
        bias = (1 - ratio) / 2
        c_coord = [
            [0.5, 0.5],
            [0.5 - bias, 0.5], [0.5, 0.5 - bias], [0.5, 0.5 + bias], [0.5 + bias, 0.5],
            [0.5 - bias, 0.5 - bias], [0.5 - bias, 0.5 + bias], [0.5 + bias, 0.5 - bias], [0.5 + bias, 0.5 + bias]
        ]
        # -1 for random
        if type == -1:
            type = random.randint(0, 8)

        height = self.frames1.shape[1]
        width = self.frames1.shape[2]

        return int((c_coord[type][0] - r) * height), int((c_coord[type][1] - r) * width),\
               math.ceil(ratio * height), math.ceil(ratio * width)

    def frame_crop_split(self, location_type=-1, window_ratio=0.8):
        '''
        `location_type`:
            - 0: center
            - 1~4: in order [top, left, right, bottom]
            - 5~8: in order [top left, top right, bottom left, bottom right]
            - -1: random
        '''
        sx, sy, lx, ly = self.get_window(location_type, window_ratio)

        frame_num = self.frames1.shape[0]
        height = self.frames1.shape[1]
        width = self.frames1.shape[2]

        cut = int((random.random() * 0.5 + 0.25) * frame_num)

        res = np.zeros(self.frames1.shape).astype(np.uint8)
        for i in range(frame_num):
            res[i] = cv2.resize(
                self.frames1[i][sx:(sx + lx), sy:(sy + ly)],
                dsize=(width, height),
                interpolation=cv2.INTER_LINEAR
            ) if i >= cut else self.frames1[i]

        one_hot = np.zeros(frame_num, ).astype(np.uint8)
        one_hot[cut - 1] = 1
        multi_hot = one_hot

        return res, one_hot, multi_hot

    def merge(self, prob=[0.36, 0.36, 0.1, 0.06, 0.12]):
        prob = np.array(prob)
        assert prob.shape == (5, )
        prob /= prob.sum()

        f = np.random.choice([
            self.forced_merge,
            self.gradual_transition_merge,
            self.push_transition_merge,
            self.wipe_transition_merge,
            self.frame_crop_split
        ], p=prob)
        return f()

    ###############################################

    def frame_darken(self, dark_coef=0.85):
        pure_black = np.zeros(self.frames1.shape[1:])
        
        frame_num = self.frames1.shape[0]

        res = np.zeros(self.frames1.shape)
        for i in range(frame_num):
            res[i] = self.frames1[i] * (1 - dark_coef) + pure_black * dark_coef

        return res.astype(np.uint8)

    def frame_glare(self, color_type=-1):
        '''
        `color_type`:
            - 0~7: in order [red, green, blue, yellow, cyan, orange, pink, purple]
            - -1: random in fragments of a certain length
        '''
        # The following color generation is based on RGB
        # cv2 uses BGR. Pay attention!
        pure = [None for _ in range(8)]
        pure[0] = np.zeros(self.frames1.shape[1:])
        pure[0][:, :, 0] = 0xFF
        pure[1] = np.zeros(self.frames1.shape[1:])
        pure[1][:, :, 1] = 0xFF
        pure[2] = np.zeros(self.frames1.shape[1:])
        pure[2][:, :, 2] = 0xFF
        pure[3] = (pure[0] + pure[1])
        pure[4] = (pure[1] + pure[2])
        pure[5] = (pure[0] + pure[1] * (97 / 255))
        pure[6] = (pure[0] + pure[1] * (192 / 255) + pure[2] * (203 / 255))
        pure[7] = (pure[0] * (160 / 255) + pure[1] * (32 / 255) + pure[2] * (240 / 255))

        frame_num = self.frames1.shape[0]
        frame_per_piece = random.randint(10, 25)

        res = np.zeros(self.frames1.shape)
        count = 0
        color = color_type if color_type != -1 else random.randint(0, 7)
        coef = random.random() * 0.15 + 0.55
        for i in range(frame_num):
            res[i] = self.frames1[i] * coef + pure[color] * (1 - coef)
            count += 1
            if color_type == -1 and count == frame_per_piece:
                count = 0
                color = random.randint(0, 7)
                coef = random.random() * 0.15 + 0.55

        return res.astype(np.uint8)

    def frame_extract(self, gap=2):
        frame_num = self.frames1.shape[0]

        res = np.zeros(self.frames1.shape).astype(np.uint8)
        for i in range(frame_num):
            res[i] = self.frames1[i // gap * gap]

        return res

    def frame_crop(self, location_type=-1, window_ratio=0.8):
        '''
        `location_type`:
            - 0: center
            - 1~4: in order [top, left, right, bottom]
            - 5~8: in order [top left, top right, bottom left, bottom right]
            - -1: random
        '''
        sx, sy, lx, ly = self.get_window(location_type, window_ratio)

        frame_num = self.frames1.shape[0]
        height = self.frames1.shape[1]
        width = self.frames1.shape[2]

        res = np.zeros(self.frames1.shape).astype(np.uint8)
        for i in range(frame_num):
            res[i] = cv2.resize(
                self.frames1[i][sx:(sx + lx), sy:(sy + ly)],
                dsize=(width, height),
                interpolation=cv2.INTER_LINEAR
            )

        return res
