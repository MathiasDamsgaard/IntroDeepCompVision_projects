from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


class FrameImageDataset(torch.utils.data.Dataset):
    """Dataset that returns individual frames from videos.

    Used for training and validation where we treat each frame independently.
    """

    def __init__(
        self,
        root_dir: str = "/dtu/datasets1/02516/ufc10",
        split: str = "train",
        transform: None | transforms.Compose = None,
    ) -> None:
        base_path = Path(root_dir) / "frames" / split
        self.frame_paths = sorted(base_path.glob("*/*/*.jpg"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame_paths)

    def _get_meta(self, attr: str, value: str) -> pd.DataFrame:
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | None, int]:
        frame_path = self.frame_paths[idx]
        video_name = frame_path.parent.name
        video_meta = self._get_meta("video_name", video_name)

        # Robust label extraction: handle zero or multiple metadata rows
        if video_meta.empty:
            msg = f"No metadata found for video_name='{video_name}' (frame={frame_path})"
            raise KeyError(msg)

        labels = video_meta["label"].unique()
        if len(labels) > 1:
            msg = f"Multiple different labels found for video_name='{video_name}': {labels}."
            raise ValueError(msg)

        label = int(labels[0])

        frame = Image.open(frame_path).convert("RGB")

        frame = self.transform(frame) if self.transform else transforms.ToTensor()(frame)

        if not isinstance(frame, torch.Tensor):
            msg = "Transform must return a torch.Tensor"
            raise TypeError(msg)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    """Dataset that returns all frames of a video together.

    Used for testing where we want to aggregate predictions across all frames of a video.
    If stack_frames=True: returns tensor of shape [C, n_frames, H, W]
    If stack_frames=False: returns list of n_frames tensors, each of shape [C, H, W]
    """

    def __init__(
        self,
        root_dir: str = "/dtu/datasets1/02516/ufc10",
        split: str = "train",
        transform: transforms.Compose | None = None,
        stack_frames: bool = True,
    ) -> None:
        base_path = Path(root_dir) / "videos" / split
        self.video_paths = sorted(base_path.glob("*/*.avi"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames

        self.n_sampled_frames = 10  # Number of frames sampled per video

    def __len__(self) -> int:
        return len(self.video_paths)

    def _get_meta(self, attr: str, value: str) -> pd.DataFrame:
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | list[torch.Tensor], int]:
        video_path = self.video_paths[idx]
        video_name = video_path.stem
        video_meta = self._get_meta("video_name", video_name)

        # Robust label extraction: handle zero or multiple metadata rows
        if video_meta.empty:
            msg = f"No metadata found for video_name='{video_name}' (video={video_path})"
            raise KeyError(msg)

        labels = video_meta["label"].unique()
        if len(labels) > 1:
            msg = f"Multiple different labels found for video_name='{video_name}': {labels}."
            raise ValueError(msg)

        label = int(labels[0])

        # Convert to string, do string manipulation, then convert back to Path for compatibility
        video_path_str = str(video_path)
        video_frames_dir = Path(video_path_str.split(".avi")[0].replace("videos", "frames"))
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [transforms.ToTensor()(frame) for frame in video_frames]

        # Cast to tensor and permute to [C, H, W]
        frames = [frame if isinstance(frame, torch.Tensor) else transforms.ToTensor()(frame) for frame in frames]

        if self.stack_frames:
            # Stack frames into single tensor: [n_frames, C, H, W] -> [C, n_frames, H, W]
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        if not (isinstance(frames, torch.Tensor | list)):
            msg = "Transform must return a torch.Tensor or list of torch.Tensors"
            raise TypeError(msg)

        if isinstance(frames, list) and any(not isinstance(f, torch.Tensor) for f in frames):
            msg = "All items in frames list must be torch.Tensor"
            raise TypeError(msg)

        return frames, label

    def load_frames(self, frames_dir: Path) -> list[Image.Image]:
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = frames_dir / f"frame_{i}.jpg"
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


class FlowVideoDataset(FrameVideoDataset):
    """Dataset that returns optical flow frames of a video together.

    Inherits from FrameVideoDataset but looks for optical flow frames instead.
    """

    def __init__(
        self,
        root_dir: str = "/dtu/datasets1/02516/ucf101_noleakage/flows",
        split: str = "train",
        transform: transforms.Compose | None = None,
        stack_frames: bool = True,
    ) -> None:
        base_path = Path(root_dir) / "flows" / split
        self.video_paths = sorted(base_path.glob("*/*.avi"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames

        self.n_sampled_frames = 10  # Number of frames sampled per video

    def load_frames(self, frames_dir: Path) -> list[Image.Image]:
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file_x = frames_dir / f"flow_x_{i}.jpg"
            frame_file_y = frames_dir / f"flow_y_{i}.jpg"
            frame_x = Image.open(frame_file_x).convert("L")  # Grayscale
            frame_y = Image.open(frame_file_y).convert("L")  # Grayscale
            # Combine x and y flow into a two-channel image
            frame = Image.merge("LA", (frame_x, frame_y))
            frames.append(frame)

        return frames

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    root_dir = "/dtu/datasets1/02516/ufc10"

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split="val", transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split="val", transform=transform, stack_frames=True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split="val", transform=transform, stack_frames=False)

    frameimage_loader = DataLoader(frameimage_dataset, batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset, batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset, batch_size=8, shuffle=False)

    for _frames, _labels in frameimage_loader:
        pass  # [batch, channels, height, width]

    for video_frames, _labels in framevideolist_loader:
        for _frame in video_frames:  # loop through number of frames
            pass  # [batch, channels, height, width]

    for _video_frames, _labels in framevideostack_loader:
        pass  # [batch, channels, number of frames, height, width]
