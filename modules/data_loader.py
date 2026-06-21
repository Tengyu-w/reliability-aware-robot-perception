# data_loader.py
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
import torch

class SimpleAVIDataset(Dataset):
    """Simplified AVI video dataset"""

    def __init__(self, video_dir, label_df=None, num_frames=16,
                 img_size=224, is_train=True, class_to_idx=None):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = is_train
        self.class_to_idx = class_to_idx

        # Get all AVI files
        self.video_files = []
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.lower().endswith('.avi'):
                    self.video_files.append(os.path.join(root, file))

        print(f"Found {len(self.video_files)} AVI files in {video_dir}")

        if len(self.video_files) == 0:
            print(f"Warning: No AVI files found in {video_dir}")
            # Try other video formats
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in ['.mp4', '.mov', '.mkv']):
                        self.video_files.append(os.path.join(root, file))
            print(f"Found other format videos: {len(self.video_files)}")

        # Match labels
        self.labels = []
        if label_df is not None and len(self.video_files) > 0 and class_to_idx is not None:
            # Create label mapping. Prefer exact matching to avoid cases like
            # "video1" accidentally matching "video10".
            label_map = {}
            for _, row in label_df.iterrows():
                video_name = str(row['video_name']).strip()
                label_name = str(row['label_name']).strip()
                if label_name in class_to_idx:
                    base_name = os.path.splitext(os.path.basename(video_name))[0]
                    label_map[video_name] = class_to_idx[label_name]
                    label_map[base_name] = class_to_idx[label_name]

            # Assign labels for each video
            for video_path in self.video_files:
                video_file = os.path.basename(video_path)
                video_name = os.path.splitext(video_file)[0]

                label = label_map.get(video_file)
                if label is None:
                    label = label_map.get(video_name)

                if label is not None:
                    self.labels.append(label)
                else:
                    # Mark with -1 if no label found
                    self.labels.append(-1)

            # Filter out samples without labels (only for training)
            if is_train:
                valid_indices = [i for i, label in enumerate(self.labels) if label != -1]
                self.video_files = [self.video_files[i] for i in valid_indices]
                self.labels = [self.labels[i] for i in valid_indices]
                print(f"Valid training samples: {len(self.video_files)}")
        else:
            # Test set: use default label 0 for all samples
            self.labels = [0] * len(self.video_files)

        # Data transformations
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]

        # Extract frames
        frames = self.extract_frames_simple(video_path)

        if frames is None or len(frames) == 0:
            # Create blank frames
            frames = [np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                      for _ in range(self.num_frames)]

        # Ensure correct number of frames
        if len(frames) > self.num_frames:
            # Uniform sampling
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.num_frames:
            # Repeat last frame
            last_frame = frames[-1] if len(frames) > 0 else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            while len(frames) < self.num_frames:
                frames.append(last_frame.copy())

        # Apply transformations
        frames_tensor = []
        for frame in frames:
            frame_pil = Image.fromarray(frame)
            frame_tensor = self.transform(frame_pil)
            frames_tensor.append(frame_tensor)

        frames_tensor = torch.stack(frames_tensor)

        return frames_tensor, torch.tensor(label, dtype=torch.long)

    def extract_frames_simple(self, video_path):
        """Simplified frame extraction function"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                return None

            frames = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.img_size, self.img_size))

                frames.append(frame)
                frame_count += 1

                # Only read enough frames
                if frame_count >= self.num_frames * 2:  # Read more for sampling
                    break

            cap.release()

            if len(frames) == 0:
                print(f"No frames in video: {video_path}")
                return None

            return frames

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None
