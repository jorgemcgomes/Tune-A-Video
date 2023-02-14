import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
import random
import torch


IMAGE_EXTENSION = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def short_size_scale(images, size):
    h, w = images.shape[-2:]
    short, long = (h, w) if h < w else (w, h)

    scale = size / short
    long_target = int(scale * long)

    target_size = (size, long_target) if h < w else (long_target, size)

    return torch.nn.functional.interpolate(
        input=images, size=target_size, mode="bilinear", antialias=True
    )


def random_short_side_scale(images, size_min, size_max):
    size = random.randint(size_min, size_max)
    return short_size_scale(images, size)


def random_crop(images, height, width):
    image_h, image_w = images.shape[-2:]
    h_start = random.randint(0, image_h - height)
    w_start = random.randint(0, image_w - width)
    return images[:, :, h_start : h_start + height, w_start : w_start + width]


def center_crop(images, height, width):
    image_h, image_w = images.shape[-2:]
    h_start = (image_h - height) // 2
    w_start = (image_w - width) // 2
    return images[:, :, h_start : h_start + height, w_start : w_start + width]


class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example


class ImageSequenceDataset(Dataset):
    def __init__(
        self,
        path: str,
        prompt_ids: torch.Tensor,
        prompt: str,
        n_sample_frame: int = 8,
        sampling_rate: int = 1,
        stride: int = 1,
        image_mode: str = "RGB",
        image_size: int = 512,
        crop: str = "center",
    ):
        self.path = path
        self.images = self.get_image_list(path)
        self.n_images = len(self.images)

        self.n_sample_frame = n_sample_frame
        self.sampling_rate = sampling_rate

        self.sequence_length = (n_sample_frame - 1) * sampling_rate + 1
        if self.n_images < self.sequence_length:
            raise ValueError
        self.stride = stride

        self.image_mode = image_mode
        self.image_size = image_size
        crop_methods = {
            "center": center_crop,
            "random": random_crop,
        }
        if crop not in crop_methods:
            raise ValueError
        self.crop = crop_methods[crop]

        self.prompt = prompt
        self.prompt_ids = prompt_ids

    def __len__(self):
        return (self.n_images - self.sequence_length) // self.stride + 1

    def __getitem__(self, index):
        frame_indices = self.get_frame_indices(index)
        frames = [self.load_frame(i) for i in frame_indices]
        frames = self.transform(frames)

        return {
            "images": frames,
            "prompt_ids": self.prompt_ids,
        }

    def transform(self, frames):
        frames = self.tensorize_frames(frames)
        frames = short_size_scale(frames, size=self.image_size)
        frames = self.crop(frames, height=self.image_size, width=self.image_size)
        return frames

    @staticmethod
    def tensorize_frames(frames):
        frames = rearrange(np.stack(frames), "f h w c -> c f h w")
        return torch.from_numpy(frames).div(255) * 2 - 1

    def load_frame(self, index):
        image_path = os.path.join(self.path, self.images[index])
        return Image.open(image_path).convert(self.image_mode)

    def get_frame_indices(self, index):
        frame_start = self.stride * index
        return (frame_start + i * self.sampling_rate for i in range(self.n_sample_frame))

    @staticmethod
    def get_image_list(path):
        images = []
        for file in sorted(os.listdir(path)):
            if file.endswith(IMAGE_EXTENSION):
                images.append(file)
        return images
