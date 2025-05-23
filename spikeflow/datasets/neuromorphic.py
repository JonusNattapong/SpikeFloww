import torch
import torch.utils.data as data
import numpy as np
import os
import urllib.request
import tarfile
from typing import Tuple, Optional, Callable, List
from pathlib import Path

class NMNIST(data.Dataset):
    """Neuromorphic MNIST dataset with temporal spike encoding"""
    
    url = "https://www.garrickorchard.com/datasets/n-mnist"
    
    def __init__(self, 
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 time_window: int = 100,
                 dt: float = 1.0):
        
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.time_window = time_window
        self.dt = dt
        
        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found. Use download=True to download it.')
        
        # Load data
        self.data, self.targets = self._load_data()
    
    def _check_exists(self) -> bool:
        """Check if dataset exists"""
        return (self.root / 'processed').exists()
    
    def download(self):
        """Download and process N-MNIST dataset"""
        if self._check_exists():
            return
        
        print("Processing N-MNIST dataset...")
        os.makedirs(self.root / 'processed', exist_ok=True)
        
        # Generate synthetic N-MNIST for demo (replace with real download)
        self._generate_synthetic_nmnist()
    
    def _generate_synthetic_nmnist(self):
        """Generate synthetic N-MNIST data for demonstration"""
        n_samples = 1000 if self.train else 200
        
        data_list = []
        targets_list = []
        
        for i in range(n_samples):
            # Create sparse spike tensor (time, height, width, polarity)
            spikes = torch.zeros(self.time_window, 34, 34, 2)
            
            # Add random spike patterns
            n_spikes = np.random.randint(50, 200)
            for _ in range(n_spikes):
                t = np.random.randint(0, self.time_window)
                x = np.random.randint(0, 34)
                y = np.random.randint(0, 34)
                p = np.random.randint(0, 2)
                spikes[t, x, y, p] = 1.0
            
            data_list.append(spikes)
            targets_list.append(np.random.randint(0, 10))
        
        # Save processed data
        torch.save({
            'data': data_list,
            'targets': targets_list
        }, self.root / 'processed' / f'{"train" if self.train else "test"}.pt')
    
    def _load_data(self) -> Tuple[List[torch.Tensor], List[int]]:
        """Load processed data"""
        file_path = self.root / 'processed' / f'{"train" if self.train else "test"}.pt'
        data_dict = torch.load(file_path)
        return data_dict['data'], data_dict['targets']
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        spike_data = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            spike_data = self.transform(spike_data)
        
        return spike_data, target


class DVSGesture(data.Dataset):
    """DVS Gesture recognition dataset"""
    
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 time_window: int = 200):
        
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.time_window = time_window
        
        if download:
            self.download()
        
        self.data, self.targets = self._load_data()
    
    def download(self):
        """Download DVS gesture dataset"""
        print("Generating synthetic DVS gesture data...")
        os.makedirs(self.root / 'processed', exist_ok=True)
        
        # Generate synthetic gesture data
        n_samples = 800 if self.train else 200
        n_classes = 11  # DVS Gesture has 11 classes
        
        data_list = []
        targets_list = []
        
        for i in range(n_samples):
            # Create temporal gesture pattern
            spikes = torch.zeros(self.time_window, 128, 128, 2)
            
            # Simulate gesture trajectory
            class_id = np.random.randint(0, n_classes)
            self._generate_gesture_pattern(spikes, class_id)
            
            data_list.append(spikes)
            targets_list.append(class_id)
        
        torch.save({
            'data': data_list,
            'targets': targets_list
        }, self.root / 'processed' / f'{"train" if self.train else "test"}_gesture.pt')
    
    def _generate_gesture_pattern(self, spikes: torch.Tensor, class_id: int):
        """Generate synthetic gesture pattern"""
        # Simple trajectory simulation
        center_x, center_y = 64, 64
        radius = 20 + class_id * 5
        
        for t in range(self.time_window):
            angle = 2 * np.pi * t / self.time_window + class_id * np.pi / 6
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            if 0 <= x < 128 and 0 <= y < 128:
                # Add spikes around trajectory
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if 0 <= x+dx < 128 and 0 <= y+dy < 128:
                            if np.random.random() < 0.3:
                                pol = np.random.randint(0, 2)
                                spikes[t, x+dx, y+dy, pol] = 1.0
    
    def _load_data(self):
        """Load processed gesture data"""
        file_path = self.root / 'processed' / f'{"train" if self.train else "test"}_gesture.pt'
        if not file_path.exists():
            self.download()
        
        data_dict = torch.load(file_path)
        return data_dict['data'], data_dict['targets']
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], self.targets[idx]


class NCALTECH101(data.Dataset):
    """Neuromorphic Caltech101 dataset"""
    
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 time_window: int = 300):
        
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.time_window = time_window
        
        # Generate demo data
        self._generate_demo_data()
        self.data, self.targets = self._load_data()
    
    def _generate_demo_data(self):
        """Generate demo N-Caltech101 data"""
        os.makedirs(self.root / 'processed', exist_ok=True)
        
        n_samples = 600 if self.train else 150
        n_classes = 101
        
        data_list = []
        targets_list = []
        
        for i in range(n_samples):
            # Create object recognition spike pattern
            spikes = torch.zeros(self.time_window, 180, 240, 2)
            class_id = np.random.randint(0, n_classes)
            
            # Add object-like spike patterns
            self._generate_object_pattern(spikes, class_id)
            
            data_list.append(spikes)
            targets_list.append(class_id)
        
        torch.save({
            'data': data_list,
            'targets': targets_list
        }, self.root / 'processed' / f'{"train" if self.train else "test"}_caltech.pt')
    
    def _generate_object_pattern(self, spikes: torch.Tensor, class_id: int):
        """Generate object-like spike pattern"""
        # Create different patterns for different classes
        pattern_type = class_id % 5
        
        if pattern_type == 0:  # Circular objects
            center_x, center_y = 90, 120
            radius = 30 + (class_id % 20)
            for t in range(0, self.time_window, 5):
                for angle in np.linspace(0, 2*np.pi, 20):
                    x = int(center_x + radius * np.cos(angle))
                    y = int(center_y + radius * np.sin(angle))
                    if 0 <= x < 180 and 0 <= y < 240:
                        spikes[t, x, y, np.random.randint(0, 2)] = 1.0
        
        elif pattern_type == 1:  # Linear objects
            for t in range(0, self.time_window, 3):
                start_x = 30 + (class_id % 30)
                for i in range(100):
                    x = start_x + i
                    y = 120 + int(20 * np.sin(i * 0.1))
                    if 0 <= x < 180 and 0 <= y < 240:
                        spikes[t, x, y, np.random.randint(0, 2)] = 1.0
        
        # Add more pattern types...
    
    def _load_data(self):
        """Load processed data"""
        file_path = self.root / 'processed' / f'{"train" if self.train else "test"}_caltech.pt'
        data_dict = torch.load(file_path)
        return data_dict['data'], data_dict['targets']
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], self.targets[idx]
