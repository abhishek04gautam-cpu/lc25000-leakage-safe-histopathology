"""Lightweight dataset container used by the LC25000 framework."""
import numpy as np

class Dataset:
    def __init__(self, name, X, y, data_type, class_names=None, groups=None):
        self.name = name
        self.X = np.asarray(X)
        self.y = np.asarray(y).astype(int).flatten()
        self.data_type = data_type
        self.class_names = class_names
        self.groups = None if groups is None else np.asarray(groups)
        self.num_classes = len(np.unique(self.y))

    def summary(self):
        print("=" * 80)
        print(f"Dataset: {self.name}")
        print("=" * 80)
        print("Samples:", self.X.shape[0])
        if self.data_type == "image" and len(self.X.shape) >= 4:
            print("Image shape:", self.X.shape[1:])
        elif len(self.X.shape) > 1:
            print("Features:", self.X.shape[1])
        else:
            print("Features: 1")
        unique, counts = np.unique(self.y, return_counts=True)
        print("Classes:", self.num_classes)
        print("Class distribution:")
        for class_index, count in zip(unique, counts):
            class_label = self.class_names[int(class_index)] if self.class_names and int(class_index) < len(self.class_names) else str(class_index)
            print(f"  Class {int(class_index)} ({class_label}): {int(count)}")
        if self.groups is not None:
            print("Unique groups:", len(np.unique(self.groups)))
