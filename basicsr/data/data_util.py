import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F

from basicsr.data.transforms import mod_crop
from basicsr.utils import img2tensor, scandir


def read_img_seq(path, require_mod_crop=False, scale=1, return_imgname=False):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Default False.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
        list[str]: Returned image name list.
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]

    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs


def generate_frame_indices(crt_idx, max_frame_num, num_frames, padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    ::

        lq.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(f'{input_key} folder and {gt_key} folder should both in lmdb '
                         f'formats. But received {input_key}: {input_folder}; '
                         f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(dict([(f'{input_key}_path', lmdb_key), (f'{gt_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.strip().split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths


import os
import os.path as osp

def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    # 获取输入和 GT 文件夹中的所有文件（忽略扩展名）
    input_paths = {osp.splitext(osp.basename(f))[0]: f for f in os.listdir(input_folder)}
    gt_paths = {osp.splitext(osp.basename(f))[0]: f for f in os.listdir(gt_folder)}

    # 检查文件数量是否一致
    assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(gt_paths)}.')

    paths = []
    for basename, gt_file in gt_paths.items():
        # 根据文件名模板生成输入文件名（忽略扩展名）
        input_name = filename_tmpl.format(basename)
        
        # 查找输入文件夹中匹配的文件（忽略扩展名）
        if input_name not in input_paths:
            raise FileNotFoundError(f'{input_name} is not in {input_key}_paths.')

        # 获取完整的文件路径
        input_path = osp.join(input_folder, input_paths[input_name])
        gt_path = osp.join(gt_folder, gt_file)

        # 添加到路径列表
        paths.append({f'{input_key}_path': input_path, f'{gt_key}_path': gt_path})

    return paths


def triple_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paths for input, gt, and mask from folders.

    Args:
        folders (list[str]): A list of folder paths. The order of list should be [input_folder, gt_folder, mask_folder].
        keys (list[str]): A list of keys identifying folders, e.g., ['lq', 'gt', 'mask'].
        filename_tmpl (str): Template for each filename.

    Returns:
        list[str]: Returned paths for input, gt, and mask.
    """
    assert len(folders) == 3, ('The len of folders should be 3 with [input_folder, gt_folder, mask_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 3, f'The len of keys should be 3 with [input_key, gt_key, mask_key]. But got {len(keys)}'
    
    input_folder, gt_folder, mask_folder = folders
    input_key, gt_key, mask_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    mask_paths = list(scandir(mask_folder))

    paths = []
    for gt_path in gt_paths:
        basename, _ = osp.splitext(osp.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}.raw'
        mask_name = f'{filename_tmpl.format(basename)}.png'

        input_path = osp.join(input_folder, input_name)
        mask_path = osp.join(mask_folder, mask_name)

        assert input_name in input_paths, f'{input_name} is not in {input_key}_paths.'
        assert mask_name in mask_paths, f'{mask_name} is not in {mask_key}_paths.'

        gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path), (f'{mask_key}_path', mask_path)]))
    
    return paths

def quadruple_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paths for input, gt, and two masks (masko and maskp) from folders.
    
    Args:
        folders (list[str]): A list of folder paths. The order should be 
            [input_folder, gt_folder, masko_folder, maskp_folder].
        keys (list[str]): A list of keys identifying folders, e.g., ['lq', 'gt', 'masko', 'maskp'].
        filename_tmpl (str): Template for filename prefixes (without extension).
    
    Returns:
        list[dict]: A list of dictionaries, each containing paths for input, gt, masko, and maskp.
    """
    assert len(folders) == 4, ('The len of folders should be 4 with [input_folder, gt_folder, masko_folder, maskp_folder]. '
                              f'But got {len(folders)}')
    assert len(keys) == 4, f'The len of keys should be 4 with [input_key, gt_key, masko_key, maskp_key]. But got {len(keys)}'
    
    input_folder, gt_folder, masko_folder, maskp_folder = folders
    input_key, gt_key, masko_key, maskp_key = keys

    # Get all files in each folder (with extensions)
    input_files = {osp.splitext(f)[0]: f for f in scandir(input_folder)}
    gt_files = {osp.splitext(f)[0]: f for f in scandir(gt_folder)}
    masko_files = {osp.splitext(f)[0]: f for f in scandir(masko_folder)}
    maskp_files = {osp.splitext(f)[0]: f for f in scandir(maskp_folder)}

    paths = []
    # Use gt files as the reference for pairing
    for gt_basename, gt_filename in gt_files.items():
        # Apply filename template to get the base name pattern
        basename_pattern = filename_tmpl.format(gt_basename)
        
        # Find matching files in other folders by prefix
        input_match = next((k for k in input_files.keys() if k.startswith(basename_pattern)), None)
        masko_match = next((k for k in masko_files.keys() if k.startswith(basename_pattern)), None)
        maskp_match = next((k for k in maskp_files.keys() if k.startswith(basename_pattern)), None)
        
        # Verify all required files exist
        assert input_match is not None, f'No matching input file found for pattern "{basename_pattern}"'
        assert masko_match is not None, f'No matching masko file found for pattern "{basename_pattern}"'
        assert maskp_match is not None, f'No matching maskp file found for pattern "{basename_pattern}"'
        
        # Construct full paths
        paths.append({
            f'{input_key}_path': osp.join(input_folder, input_files[input_match]),
            f'{gt_key}_path': osp.join(gt_folder, gt_filename),
            f'{masko_key}_path': osp.join(masko_folder, masko_files[masko_match]),
            f'{maskp_key}_path': osp.join(maskp_folder, maskp_files[maskp_match])
        })
    
    return paths

    
# def quadruple_paths_from_folder(folders, keys, filename_tmpl):
#     """Generate paths for input, gt, mask, and ref from folders.

#     Args:
#         folders (list[str]): A list of folder paths. The order of list should be 
#             [input_folder, gt_folder, mask_folder, ref_folder].
#         keys (list[str]): A list of keys identifying folders, e.g., ['lq', 'gt', 'mask', 'ref'].
#         filename_tmpl (str): Template for each filename.

#     Returns:
#         list[str]: Returned paths for input, gt, mask, and ref.
#     """
#     assert len(folders) == 4, ('The len of folders should be 4 with [input_folder, gt_folder, mask_folder, ref_folder]. '
#                                f'But got {len(folders)}')
#     assert len(keys) == 4, f'The len of keys should be 4 with [input_key, gt_key, mask_key, ref_key]. But got {len(keys)}'
    
#     input_folder, gt_folder, mask_folder, ref_folder = folders
#     input_key, gt_key, mask_key, ref_key = keys

#     input_paths = list(scandir(input_folder))
#     gt_paths = list(scandir(gt_folder))
#     mask_paths = list(scandir(mask_folder))
#     ref_paths = list(scandir(ref_folder))

#     paths = []
#     for gt_path in gt_paths:
#         basename, _ = osp.splitext(osp.basename(gt_path))
#         input_name = f'{filename_tmpl.format(basename)}.raw'
#         mask_name = f'{filename_tmpl.format(basename)}.png'
#         ref_name = f'{filename_tmpl.format(basename)}.JPG'  # 假设ref图片格式是.jpg，可以根据需求修改

#         input_path = osp.join(input_folder, input_name)
#         mask_path = osp.join(mask_folder, mask_name)
#         ref_path = osp.join(ref_folder, ref_name)

#         assert input_name in input_paths, f'{input_name} is not in {input_key}_paths.'
#         assert mask_name in mask_paths, f'{mask_name} is not in {mask_key}_paths.'
#         assert ref_name in ref_paths, f'{ref_name} is not in {ref_key}_paths.'

#         gt_path = osp.join(gt_folder, gt_path)
#         paths.append(dict([
#             (f'{input_key}_path', input_path), 
#             (f'{gt_key}_path', gt_path), 
#             (f'{mask_key}_path', mask_path), 
#             (f'{ref_key}_path', ref_path)
#         ]))
    
#     return paths


def paired_paths_from_folder_avif(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    # for gt_path in gt_paths:
    #     basename, ext = osp.splitext(osp.basename(gt_path))
    #     input_name = f'{filename_tmpl.format(basename)}{ext}'
    #     input_path = osp.join(input_folder, input_name)
    #     assert input_name in input_paths, f'{input_name} is not in {input_key}_paths.'
    #     gt_path = osp.join(gt_folder, gt_path)
    #     paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    # return paths
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        ext = '.avif'
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, f'{input_name} is not in {input_key}_paths.'
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths


def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3, 4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x
