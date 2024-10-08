import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
add_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(add_dir)
import os.path as osp
from glob import glob
import numpy as np
import torch
import torch.utils.data as data
import math
from PIL import Image, ImageDraw
import random
import pickle
#######################################MRI###################################################

def custom_sort(item):
    return int(item.split('_')[-1].split('.npy')[0])

def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask

def random_bbox(img_shape=(240,240), max_bbox_shape=(30, 30), max_bbox_delta=10, min_margin=20):
    """Generate a random bbox for the mask on a given image.

    In our implementation, the max value cannot be obtained since we use
    `np.random.randint`. And this may be different with other standard scripts
    in the community.

    Args:
        img_shape (tuple[int]): The size of a image, in the form of (h, w).
        max_bbox_shape (int | tuple[int]): Maximum shape of the mask box,
            in the form of (h, w). If it is an integer, the mask box will be
            square.
        max_bbox_delta (int | tuple[int]): Maximum delta of the mask box,
            in the form of (delta_h, delta_w). If it is an integer, delta_h
            and delta_w will be the same. Mask shape will be randomly sampled
            from the range of `max_bbox_shape - max_bbox_delta` and
            `max_bbox_shape`. Default: (40, 40).
        min_margin (int | tuple[int]): The minimum margin size from the
            edges of mask box to the image boarder, in the form of
            (margin_h, margin_w). If it is an integer, margin_h and margin_w
            will be the same. Default: (20, 20).

    Returns:
        tuple[int]: The generated box, (top, left, h, w).
    """
    if not isinstance(max_bbox_shape, tuple):
        max_bbox_shape = (max_bbox_shape, max_bbox_shape)
    if not isinstance(max_bbox_delta, tuple):
        max_bbox_delta = (max_bbox_delta, max_bbox_delta)
    if not isinstance(min_margin, tuple):
        min_margin = (min_margin, min_margin)
        
    img_h, img_w = img_shape[:2]
    max_mask_h, max_mask_w = max_bbox_shape
    max_delta_h, max_delta_w = max_bbox_delta
    margin_h, margin_w = min_margin

    if max_mask_h > img_h or max_mask_w > img_w:
        raise ValueError(f'mask shape {max_bbox_shape} should be smaller than '
                         f'image shape {img_shape}')
    if (max_delta_h // 2 * 2 >= max_mask_h
            or max_delta_w // 2 * 2 >= max_mask_w):
        raise ValueError(f'mask delta {max_bbox_delta} should be smaller than'
                         f'mask shape {max_bbox_shape}')
    if img_h - max_mask_h < 2 * margin_h or img_w - max_mask_w < 2 * margin_w:
        raise ValueError(f'Margin {min_margin} cannot be satisfied for img'
                         f'shape {img_shape} and mask shape {max_bbox_shape}')

    # get the max value of (top, left)
    max_top = img_h - margin_h - max_mask_h
    max_left = img_w - margin_w - max_mask_w
    # randomly select a (top, left)
    top = np.random.randint(margin_h, max_top)
    left = np.random.randint(margin_w, max_left)
    # randomly shrink the shape of mask box according to `max_bbox_delta`
    # the center of box is fixed
    delta_top = np.random.randint(0, max_delta_h // 2 + 1)
    delta_left = np.random.randint(0, max_delta_w // 2 + 1)
    top = top + delta_top
    left = left + delta_left
    h = max_mask_h - delta_top
    w = max_mask_w - delta_left
    return (top, left, h, w)

def brush_stroke_mask(img_shape,
                      num_vertices=(4, 12),
                      mean_angle=2 * math.pi / 5,
                      angle_range=2 * math.pi / 15,
                      brush_width=(4, 10),
                      max_loops=4,
                      dtype='uint8'):
    """Generate free-form mask.

    The method of generating free-form mask is in the following paper:
    Free-Form Image Inpainting with Gated Convolution.

    When you set the config of this type of mask. You may note the usage of
    `np.random.randint` and the range of `np.random.randint` is [left, right).

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 12).
        mean_angle (float): Mean value of the angle in each vertex. The angle
            is measured in radians. Default: 2 * math.pi / 5.
        angle_range (float): Range of the random angle.
            Default: 2 * math.pi / 15.
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (12, 40).
        max_loops (int): The max number of for loops of drawing strokes.
        dtype (str): Indicate the data type of returned masks.
            Default: 'uint8'.

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    img_h, img_w = img_shape[:2]
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, tuple):
        min_width, max_width = brush_width
    elif isinstance(brush_width, int):
        min_width, max_width = brush_width, brush_width + 1
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    average_radius = math.sqrt(img_h * img_h + img_w * img_w) / 8
    mask = Image.new('L', (img_w, img_h), 0)

    loop_num = np.random.randint(1, max_loops)
    num_vertex_list = np.random.randint(
        min_num_vertices, max_num_vertices, size=loop_num)
    angle_min_list = np.random.uniform(0, angle_range, size=loop_num)
    angle_max_list = np.random.uniform(0, angle_range, size=loop_num)

    for loop_n in range(loop_num):
        num_vertex = num_vertex_list[loop_n]
        angle_min = mean_angle - angle_min_list[loop_n]
        angle_max = mean_angle + angle_max_list[loop_n]
        angles = []
        vertex = []

        # set random angle on each vertex
        angles = np.random.uniform(angle_min, angle_max, size=num_vertex)
        reverse_mask = (np.arange(num_vertex, dtype=np.float32) % 2) == 0
        angles[reverse_mask] = 2 * math.pi - angles[reverse_mask]

        h, w = mask.size

        # set random vertices
        vertex.append((np.random.randint(0, w), np.random.randint(0, h)))
        r_list = np.random.normal(
            loc=average_radius, scale=average_radius // 2, size=num_vertex)
        for i in range(num_vertex):
            r = np.clip(r_list[i], 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))
        # draw brush strokes according to the vertex and angle list
        draw = ImageDraw.Draw(mask)
        width = np.random.randint(min_width, max_width)
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2,
                          v[0] + width // 2, v[1] + width // 2),
                         fill=1)
    # randomly flip the mask
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.array(mask).astype(dtype=getattr(np, dtype))
    mask = mask[:, :, None]
    return mask


class LQDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and LR image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, mode, data_root, mod_input='dose_4', mod_target='dose_full', context=True, mask=False,
                 augment=False, normalize_type=1):
        super().__init__()
        # siemens
        manufacturer = ['siemens', 'uexplorer']
        base_input = []
        for manu in manufacturer:
            if mode == 'train':
                image_path = os.path.join(data_root, manu, mod_input, 'train_npy')
            elif mode == 'test':
                image_path = os.path.join(data_root, manu, mod_input, 'test_npy')
            elif mode == 'val':
                image_path = os.path.join(data_root, manu, mod_input, 'val_npy')
            # siemens
            patient_ids = os.listdir(os.path.join(image_path))
            patient_ids.sort()
            if mode in ['test', 'val']:
                patient_ids = patient_ids[:10]
            patient_lists = []  # todo
            for ind, name in enumerate(patient_ids):
                patient_list = [os.path.join(image_path, name, n) for n in os.listdir(os.path.join(image_path, name))]
                patient_list = sorted(patient_list, key=custom_sort)
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    if mode in ['test', 'val']:
                        patient_lists = patient_lists + cat_patient_list[::3]
                    else:
                        patient_lists = patient_lists + cat_patient_list
                else:
                    if mode in ['test', 'val']:
                        patient_lists = patient_lists + patient_list[1:len(patient_list) - 1][::3]
                    else:
                        patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_input = base_input + patient_lists

        base_target = [i.replace(mod_input, mod_target) for i in base_input]

        attr_dict = dict()
        for manu in manufacturer:
            with open(os.path.join(data_root, manu, 'attr.pkl'), 'rb') as f:
                attr_d = pickle.load(f)
                attr_dict.update(attr_d)

        self.inputs = base_input
        self.targets = base_target
        self.context = context
        self.mask = mask
        self.mode = mode
        self.augment = augment
        self.base_size = 32
        self.attr_dict = attr_dict
        self.normalize_type = normalize_type
        self.mod_input = mod_input
        self.mod_target = mod_target

    def __getitem__(self, index):
        input_p, target_p = self.inputs[index], self.targets[index]
        if not self.context:  # 2D
            name = input_p.split('/')[-1].split('__')[0]
            attr = self.attr_dict[name]
            input = np.load(input_p)
            target = np.load(target_p)

            if self.mode == 'train':
                if random.random() < 0.15:
                    input = np.flip(input, axis=0)
                    target = np.flip(target, axis=0)
                elif random.random() > 0.85:
                    input = np.flip(input, axis=1)
                    target = np.flip(target, axis=1)
                if random.random() < 0.15:
                    input = np.transpose(input, (1, 0))
                    target = np.transpose(target, (1, 0))

            input = input.astype(np.float32)[np.newaxis, ...]
            target = target.astype(np.float32)[np.newaxis, ...]
            raise ValueError
        else:  # 2.5D
            name = input_p.split('~')[1].split('/')[-1].split('__')[0]
            attr = self.attr_dict[name]
            inputs, targets = [], []
            for i in range(3):
                input = np.load(input_p.split('~')[i + 1])
                target = np.load(target_p.split('~')[i + 1])
                inputs.append(input)
                targets.append(target)
            input = np.stack(inputs)
            target = np.stack(targets)

            if self.normalize_type == 1:
                input = np.clip(input, 0, attr[self.mod_input]['perc995']) / attr[self.mod_input]['perc995']
                target = np.clip(target, 0, attr[self.mod_input]['perc995']) / attr[self.mod_input]['perc995']
                input = input * 2 - 1
                target = target * 2 - 1
            elif self.normalize_type == 2:
                input = (input - attr[self.mod_input]['mean']) / attr[self.mod_input]['std']
                target = (target - attr[self.mod_input]['mean']) / attr[self.mod_input]['std']
            elif self.normalize_type == 3:
                input = input / attr[self.mod_input]['mean']
                target = target / attr[self.mod_input]['mean']
            elif self.normalize_type == 4:
                input = input / (attr[self.mod_input]['mean'] + 3 * attr[self.mod_input]['std'])
                target = target / (attr[self.mod_input]['mean'] + 3 * attr[self.mod_input]['std'])
            elif self.normalize_type == 5:
                input = (input - 10000) / 5000
                target = (target - 10000) / 5000

            if self.mode == 'train':
                if random.random() < 0.15:
                    input = np.flip(input, axis=1)
                    target = np.flip(target, axis=1)
                elif random.random() > 0.85:
                    input = np.flip(input, axis=2)
                    target = np.flip(target, axis=2)
                if random.random() < 0.15:
                    input = np.transpose(input, (0, 2, 1))
                    target = np.transpose(target, (0, 2, 1))

            input = input.astype(np.float32)
            target = target.astype(np.float32)

        return {"input": input, "target": target}

    def __len__(self):
        return len(self.inputs)


class LQDataset_Multidose(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and LR image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, mode, data_root, mod_target='dose_full', context=True, mask=False,
                 augment=False, normalize_type=1):
        super().__init__()
        # siemens
        manufacturer = ['siemens', 'uexplorer']
        base_input, base_target = [], []
        for mod_input in ['dose_2', 'dose_4', 'dose_10', 'dose_20', 'dose_50', 'dose_100']:
        # for mod_input in ['dose_4', 'dose_10']:
            for manu in manufacturer:
                if mode == 'train':
                    image_path = os.path.join(data_root, manu, mod_input, 'train_npy')
                elif mode == 'test':
                    image_path = os.path.join(data_root, manu, mod_input, 'test_npy')
                elif mode == 'val':
                    image_path = os.path.join(data_root, manu, mod_input, 'val_npy')
                patient_ids = os.listdir(os.path.join(image_path))
                patient_ids.sort()
                if mode in ['test', 'val']:
                    patient_ids = patient_ids[:10]
                patient_lists = []  # todo
                for ind, name in enumerate(patient_ids):
                    patient_list = [os.path.join(image_path, name, n) for n in os.listdir(os.path.join(image_path, name))]
                    if len(patient_list) == 0:
                        continue
                    patient_list = sorted(patient_list, key=custom_sort)
                    if context:
                        cat_patient_list = []
                        for i in range(1, len(patient_list) - 1):
                            patient_path = ''
                            for j in range(-1, 2):
                                patient_path = patient_path + '~' + patient_list[i + j]
                            cat_patient_list.append(patient_path)
                        if mode in ['test', 'val']:
                            patient_lists = patient_lists + cat_patient_list[::3]
                        else:
                            patient_lists = patient_lists + cat_patient_list
                    else:
                        if mode in ['test', 'val']:
                            patient_lists = patient_lists + patient_list[1:len(patient_list) - 1][::3]
                        else:
                            patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
                base_input = base_input + patient_lists
                base_target = base_target + [i.replace(mod_input, mod_target) for i in patient_lists]

        attr_dict = dict()
        for manu in manufacturer:
            with open(os.path.join(data_root, manu, 'attr.pkl'), 'rb') as f:
                attr_d = pickle.load(f)
                attr_dict.update(attr_d)

        self.inputs = base_input
        self.targets = base_target
        self.context = context
        self.mask = mask
        self.mode = mode
        self.augment = augment
        self.base_size = 32
        self.attr_dict = attr_dict
        self.normalize_type = normalize_type
        self.mod_target = mod_target

    def __getitem__(self, index):
        input_p, target_p = self.inputs[index], self.targets[index]
        if not self.context:  # 2D
            name = input_p.split('/')[-1].split('__')[0]
            attr = self.attr_dict[name]
            input = np.load(input_p)
            target = np.load(target_p)

            if self.mode == 'train':
                if random.random() < 0.15:
                    input = np.flip(input, axis=0)
                    target = np.flip(target, axis=0)
                elif random.random() > 0.85:
                    input = np.flip(input, axis=1)
                    target = np.flip(target, axis=1)
                if random.random() < 0.15:
                    input = np.transpose(input, (1, 0))
                    target = np.transpose(target, (1, 0))

            input = input.astype(np.float32)[np.newaxis, ...]
            target = target.astype(np.float32)[np.newaxis, ...]
            raise ValueError
        else:  # 2.5D
            name = input_p.split('~')[1].split('/')[-1].split('__')[0]
            attr = self.attr_dict[name]
            inputs, targets = [], []
            for i in range(3):
                input = np.load(input_p.split('~')[i + 1])
                target = np.load(target_p.split('~')[i + 1])
                inputs.append(input)
                targets.append(target)
            input = np.stack(inputs)
            target = np.stack(targets)

            mod_input = input_p.split('~')[1].split('/')[-4]
            if self.normalize_type == 1:
                input = np.clip(input, 0, attr[mod_input]['perc995']) / attr[mod_input]['perc995']
                target = np.clip(target, 0, attr[mod_input]['perc995']) / attr[mod_input]['perc995']
                input = input * 2 - 1
                target = target * 2 - 1
            elif self.normalize_type == 2:
                input = (input - attr[mod_input]['mean']) / attr[mod_input]['std']
                target = (target - attr[mod_input]['mean']) / attr[mod_input]['std']
            elif self.normalize_type == 3:
                input = input / attr[mod_input]['mean']
                target = target / attr[mod_input]['mean']
            elif self.normalize_type == 4:
                input = input / (attr[mod_input]['mean'] + 3 * attr[mod_input]['std'])
                target = target / (attr[mod_input]['mean'] + 3 * attr[mod_input]['std'])
            elif self.normalize_type == 5:
                input = (input - 10000) / 5000
                target = (target - 10000) / 5000

            if self.mode == 'train':
                if random.random() < 0.15:
                    input = np.flip(input, axis=1)
                    target = np.flip(target, axis=1)
                elif random.random() > 0.85:
                    input = np.flip(input, axis=2)
                    target = np.flip(target, axis=2)
                if random.random() < 0.15:
                    input = np.transpose(input, (0, 2, 1))
                    target = np.transpose(target, (0, 2, 1))

            input = input.astype(np.float32)
            target = target.astype(np.float32)

        return {"input": input, "target": target}

    def __len__(self):
        return len(self.inputs)


def dynamic_resize_collate(batch):
    # (237, 423), (151, 283)
    h_list, w_list = [], []
    for image in batch:
        input = image['input']
        h_list.append(input.shape[1])
        w_list.append(input.shape[2])
    h_max, h_min = max(h_list), min(h_list)
    w_max, w_min = max(w_list), min(w_list)
    h_max = int(np.ceil(h_max / 32) * 32)
    w_max = int(np.ceil(w_max / 32) * 32)
    h_min = int(np.floor(h_min / 32) * 32)
    w_min = int(np.floor(w_min / 32) * 32)
    h_list = np.arange(h_min, h_max+1, 32)
    w_list = np.arange(w_min, w_max+1, 32)
    new_h = random.choice(h_list)
    new_w = random.choice(w_list)

    inputs, targets = [], []
    for image in batch:
        input, target = image['input'], image['target']
        h, w = input.shape[1:]
        # 随机裁剪
        if h > new_h:
            start_h = np.random.randint(0, h - new_h + 1)
            input = input[:, start_h:start_h + new_h, :]
            target = target[:, start_h:start_h + new_h, :]
        if w > new_w:
            start_w = np.random.randint(0, w - new_w + 1)
            input = input[:, :, start_w:start_w + new_w]
            target = target[:, :, start_w:start_w + new_w]
        # 填充
        if h < new_h:
            pad_h = (new_h - h) // 2
            input = np.pad(input, ((0, 0), (pad_h, new_h - h - pad_h), (0, 0)), mode='reflect')
            target = np.pad(target, ((0, 0), (pad_h, new_h - h - pad_h), (0, 0)), mode='reflect')
        if w < new_w:
            pad_w = (new_w - w) // 2
            input = np.pad(input, ((0, 0), (0, 0), (pad_w, new_w - w - pad_w)), mode='reflect')
            target = np.pad(target, ((0, 0), (0, 0), (pad_w, new_w - w - pad_w)), mode='reflect')
        inputs.append(torch.from_numpy(input).float())
        targets.append(torch.from_numpy(target).float())

    # 将图像堆叠成一个 tensor
    return {"input": torch.stack(inputs), "target": torch.stack(targets)}


if __name__ == '__main__':
    from sampler import RandomSampler
    train_ds = LQDataset_Multidose(mode='test', data_root='/home/mnt/data/AIGC/PET/preprocessed',
                                   mod_target='dose_full', context=True, mask=False, augment=False, normalize_type=3)
    print(len(train_ds))
    # train_ds = LQDataset(mode='test', data_root='/home/mnt/data/AIGC/PET/preprocessed',
    #                      mod_input='dose_2', mod_target='dose_full', context=True, mask=False, augment=False, normalize_type=3)
    # print(len(train_ds))
    # train_ds = LQDataset(mode='test', data_root='/home/mnt/data/AIGC/PET/preprocessed',
    #                      mod_input='dose_4', mod_target='dose_full', context=True, mask=False, augment=False, normalize_type=3)
    # print(len(train_ds))
    # train_ds = LQDataset(mode='test', data_root='/home/mnt/data/AIGC/PET/preprocessed',
    #                      mod_input='dose_10', mod_target='dose_full', context=True, mask=False, augment=False, normalize_type=3)
    # print(len(train_ds))

    # train_sampler = RandomSampler(dataset=train_ds, batch_size=2,
    #                               num_iter=100000,
    #                               restore_iter=0)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_ds,
    #     batch_size=2,
    #     sampler=train_sampler,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=2,
    #     pin_memory=True,
    #     collate_fn=dynamic_resize_collate
    # )
    # print(len(train_loader))
    # loader = iter(train_loader)
    # inputs = next(loader)
    # input = inputs['input']
    # print(input.shape)
    # target = inputs['target']
    # img = Image.fromarray((input[0, 1].numpy() * 255).astype(np.uint8))
    # img.save('/home/SENSETIME/fengshixiang1/Desktop/img1.png')
    # img = Image.fromarray((target[0, 1].numpy() * 255).astype(np.uint8))
    # img.save('/home/SENSETIME/fengshixiang1/Desktop/img2.png')
    # img = Image.fromarray((input[1, 1].numpy() * 255).astype(np.uint8))
    # img.save('/home/SENSETIME/fengshixiang1/Desktop/img3.png')
    # img = Image.fromarray((target[1, 1].numpy() * 255).astype(np.uint8))
    # img.save('/home/SENSETIME/fengshixiang1/Desktop/img4.png')
