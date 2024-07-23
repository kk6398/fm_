from dataclasses import dataclass, replace

import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from ..dataset.types import Batch
#

@dataclass
class CroppingCfg:
    image_shape: tuple[int, int] | int
    flow_scale_multiplier: int
    patch_size: int


def resize_batch(batch: Batch, shape: tuple[int, int]) -> Batch:
    b, f, _, _, _ = batch.videos.shape       # [1, 20, 3, 3024, 4032]    # 20 是帧数 20张图像
    # print("batch.videos.shape", batch.videos.shape)                           # 循环了2次？   for model 1次  for flow 1次
    h, w = shape                             #  model : (180, 240)                     # 第二次   flow : (720, 960)
    # print("shape: ", shape)
    videos = rearrange(batch.videos, "b f c h w -> (b f) c h w")       # [1, 20, 3, 3024, 4032] -> [20, 3, 3024, 4032]   #
    videos = F.interpolate(videos, (h, w), mode="bilinear", align_corners=False)  # [20, 3, 3024, 4032] -> [20, 3, 180, 240]
    videos = rearrange(videos, "(b f) c h w -> b f c h w", b=b, f=f)        # [20, 3, 180, 240] -> [1, 20, 3, 180, 240]

    return replace(batch, videos=videos)


def compute_patch_cropped_shape(
    shape: tuple[int, int],
    patch_size: int,
) -> tuple[int, int]:
    h, w = shape      # 180 240
    # # 将变量 h 的值调整为最接近且小于等于 h 的能被 patch_size 整除的数。 17//3 = 5  5*3=15
    h_new = (h // patch_size) * patch_size   
    w_new = (w // patch_size) * patch_size

    return h_new, w_new


def center_crop_images(
    images: Float[Tensor, "*batch channel height width"],
    new_shape: tuple[int, int],
) -> Float[Tensor, "*batch channel cropped_height cropped_width"]:
    *_, h, w = images.shape
    h_new, w_new = new_shape
    row = (h - h_new) // 2
    col = (w - w_new) // 2
    return images[..., row : row + h_new, col : col + w_new]


def center_crop_intrinsics(
    intrinsics: Float[Tensor, "*#batch 3 3"] | None,
    old_shape: tuple[int, int],
    new_shape: tuple[int, int],
) -> Float[Tensor, "*batch 3 3"] | None:
    """Modify the given intrinsics to account for center cropping."""

    if intrinsics is None:
        return None

    h_old, w_old = old_shape    # 160  224
    h_new, w_new = new_shape    # (180, 240)
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_old / w_new  # fx      # * 180/160
    intrinsics[..., 1, 1] *= h_old / h_new  # fy      # * 240/224
    return intrinsics


def patch_crop_batch(batch: Batch, patch_size: int) -> Batch:          # patch_size==32  
    _, _, _, h, w = batch.videos.shape      #    torch.Size([1, 20, 3, 180, 240])
    old_shape = (h, w)               # 180, 240
    new_shape = compute_patch_cropped_shape((h, w), patch_size)     # 160, 224
    return replace(
        batch,
        intrinsics=center_crop_intrinsics(batch.intrinsics, old_shape, new_shape),   # 根据新的shape对intrinsics进行裁剪
        videos=center_crop_images(batch.videos, new_shape),
    )


def get_image_shape(
    original_shape: tuple[int, int],
    cfg: CroppingCfg,
) -> tuple[int, int]:
    # If the image shape is exact, return it.
    if isinstance(cfg.image_shape, tuple):
        return cfg.image_shape

    # Otherwise, the image shape is assumed to be an approximate number of pixels.
    h, w = original_shape        # 3024, 4032
    # print("h: ", h)
    # print("w: ", w)
    # print("cfg.image_shape: ", cfg.image_shape)
    scale = (cfg.image_shape / (h * w)) ** 0.5         # 43200 / (3024*4032) ** 0.5 = 0.0595
    # print("scale: ", scale)
    return (round(h * scale), round(w * scale))        # 0.0595* 3024=180       (180, 240)


def crop_and_resize_batch_for_model(
    batch: Batch,           # batch.videos.shape torch.Size([1, 20, 3, 3024, 4032])
    cfg: CroppingCfg,       # cfg.cropping: CroppingCfg(image_shape=43200, flow_scale_multiplier=4, patch_size=32)
) -> tuple[Batch, tuple[int, int]]:
    # Resize the batch to the desired model input size.           # [3, 3024, 4032]
    image_shape = get_image_shape(tuple(batch.videos.shape[-2:]), cfg)       # (180, 240)
    batch = resize_batch(batch, image_shape)               # [1, 20, 3, 180, 240]

    # print("batch: ", batch)
    # Record the pre-cropping shape.
    _, _, _, h, w = batch.videos.shape                 # torch.Size([1, 42, 3, 180, 240])

    # print("batch.videos.shape0000: ", batch.videos.shape)           # torch.Size([1, 20, 3, 180, 240])
    # Center-crop the batch so it's cleanly divisible by the patch size.
    return patch_crop_batch(batch, cfg.patch_size), (h, w)              # patch_size==32  


def crop_and_resize_batch_for_flow(batch: Batch, cfg: CroppingCfg) -> Batch:
    # Figure out the image size that's used for flow.
    image_shape = get_image_shape(tuple(batch.videos.shape[-2:]), cfg)
    flow_shape = tuple(dim * cfg.flow_scale_multiplier for dim in image_shape)

    # Resize the batch to match the desired flow shape.
    batch = resize_batch(batch, flow_shape)

    # Center-crop the batch so it's cleanly divisible by the patch size times the flow
    # multiplier. This ensures that the aspect ratio matches the model input's aspect
    # ratio.
    return patch_crop_batch(batch, cfg.patch_size * cfg.flow_scale_multiplier)          # patch_size==32  flow_scale_multiplier==4


def resize_to_cover(
    image: Image.Image,
    shape: tuple[int, int],
) -> tuple[
    Image.Image,  # the image itself
    tuple[int, int],  # image shape after scaling, before cropping
]:
    w_old, h_old = image.size
    h_new, w_new = shape

    # Figure out the scale factor needed to cover the desired shape with a uniformly
    # scaled version of the input image. Then, resize the input image.
    scale_factor = max(h_new / h_old, w_new / w_old)
    h_scaled = round(h_old * scale_factor)
    w_scaled = round(w_old * scale_factor)
    image_scaled = image.resize((w_scaled, h_scaled), Image.LANCZOS)

    # Center-crop the image.
    x = (w_scaled - w_new) // 2
    y = (h_scaled - h_new) // 2
    image_cropped = image_scaled.crop((x, y, x + w_new, y + h_new))
    return image_cropped, (h_scaled, w_scaled)


def resize_to_cover_with_intrinsics(
    images: list[Image.Image],
    shape: tuple[int, int],
    intrinsics: Float[Tensor, "*batch 3 3"] | None,
) -> tuple[
    list[Image.Image],  # cropped images
    Float[Tensor, "*batch 3 3"] | None,  # intrinsics, adjusted for cropping
]:
    scaled_images = []
    for image in images:
        image, old_shape = resize_to_cover(image, shape)
        scaled_images.append(image)

    if intrinsics is not None:
        intrinsics = center_crop_intrinsics(intrinsics, old_shape, shape)

    return scaled_images, intrinsics
