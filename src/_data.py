import torch
import scipy.io
import numpy as np


from torchvision import transforms
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split, Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BDGP(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + "BDGP.mat")
        self.x1 = torch.from_numpy(data["X1"].astype(np.float32))
        self.x2 = torch.from_numpy(data["X2"].astype(np.float32))
        self.labels = torch.from_numpy(data["Y"].transpose().astype(np.float32))
        self.views = [self.x1, self.x2]

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]



class AffNoisyMNIST(Dataset):
    def __init__(self, path, n_samples=500000):
        data = np.load(path)
        scaler = MinMaxScaler()

        # Load the views from the dataset and reshape
        view1 = data["view_0"].reshape(70000, -1)
        view2 = data["view_1"].reshape(70000, -1)

        # Convert to torch tensors
        view1 = torch.from_numpy(view1)
        view2 = torch.from_numpy(view2)
        self.transform = transforms.RandomAffine(degrees=0.5, translate=(0.0, 0.0), scale=(1, 1), shear=0.0)

        self.n_samples = n_samples

        # Store the views and labels
        self.original_views = [view1, view2]
        self.labels = torch.from_numpy(data["labels"])

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get the corresponding index in the original dataset (since original size is 70k)
        original_idx = idx % 70000
        
        if idx >= 70000:
            view1 = self.original_views[0][original_idx].reshape(28, 28)  # Reshape back to image format
            view2 = self.original_views[1][original_idx].reshape(28, 28)

            # Apply affine transformation to both views
            view1 = self.transform(view1.unsqueeze(0)).squeeze(0).view(-1)  # Apply transform and flatten
            view2 = self.transform(view2.unsqueeze(0)).squeeze(0).view(-1)  # Apply transform and flatten

        else:
            view1 = self.original_views[0][idx]
            view2 = self.original_views[1][idx]

        label = self.labels[original_idx]
        return [view1, view2], label


class Caltech20(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path, simplify_cells=True)
        scaler = MinMaxScaler()

        data = list(data["X"].transpose())
        views = [scaler.fit_transform(v.astype(np.float32)) for v in data]
        self.views = [torch.from_numpy(v) for v in views]
        self.labels = scipy.io.loadmat(path)["Y"] - 1
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return 2386
    
    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]


class Handwritten(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path, simplify_cells=True)
        scaler = MinMaxScaler()
        data = list(data["X"].transpose())

        views = [scaler.fit_transform(v.astype(np.float32)) for v in data]
        self.views = [torch.from_numpy(views[0]), torch.from_numpy(views[2])]
        self.labels = scipy.io.loadmat(path)["Y"]
        self.labels = torch.from_numpy(self.labels)
    
    def __len__(self):
        return 2000
    
    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]



class Reuters(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path, simplify_cells=True)
        scaler = MinMaxScaler()
        train = data['x_train']
        test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']

        views = []

        for v_train, v_test in zip(train, test):
            v = np.vstack((v_train, v_test))
            views.append(scaler.fit_transform(v.astype(np.float32)))

        self.views = [torch.from_numpy(v) for v in views]
        self.labels = np.hstack((y_train, y_test))
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return 18758
    
    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]



class Synthetic(Dataset):
    def __init__(self, n_samples=7000, noise=0.075, random_state=42, contamination=0.3):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.n_views = 2
        self.contamination = contamination  # proportion of contamination in the dataset
        
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=2.0, random_state=random_state)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        self.views = []
        for i in range(self.n_views):
            X_view = X.copy()
            
            if i == 1:
                rotation_matrix = np.array([[np.cos(1), -np.sin(1)], 
                                            [np.sin(1), np.cos(1)]])  # Small rotation
                X_view = np.dot(X_view, rotation_matrix)
            self.views.append(X_view)
                
        self.labels = torch.from_numpy(y)
        self.views = [torch.from_numpy(view).float() for view in self.views]

    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]



def train_test_split(dataset, train_ratio=0.8, seed=42):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    X_train = [dataset.views[i][train_dataset.indices] for i in range(len(dataset.views))]
    X_test = [dataset.views[i][test_dataset.indices] for i in range(len(dataset.views))]

    y_train = dataset.labels[train_dataset.indices].squeeze()
    y_test = dataset.labels[test_dataset.indices].squeeze()


    return X_train, X_test, y_train, y_test


def affnoisy_mnist_split(dataset, train_ratio=0.8, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # We only split the original 70,000 samples
    original_indices = np.arange(dataset.n_samples)
    np.random.shuffle(original_indices)

    split_point = int(dataset.n_samples * train_ratio)
    train_indices = original_indices[:split_point]
    test_indices = original_indices[split_point:]

    #For training, we'll use both original and transformed samples
    train_transformed_indices = np.random.randint(dataset.n_samples, dataset.n_samples + 1, size=dataset.n_samples - len(train_indices))
    train_indices = np.concatenate([train_indices, train_transformed_indices])

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Function to extract views and labels
    def extract_data(subset):
        loader = torch.utils.data.DataLoader(subset, batch_size=len(subset), num_workers=4, shuffle=False)
        data = next(iter(loader))
        views, labels = data[0], data[1]
        return views, labels.squeeze()

    # Extract train and test data
    X_train, y_train = extract_data(train_dataset)
    X_test, y_test = extract_data(test_dataset)

    # Separate views
    X_train = [X_train[0], X_train[1]]
    X_test = [X_test[0], X_test[1]]


    return X_train, X_test, y_train, y_test


def load_data(dataset: str) -> tuple:
    if dataset == "bdgp":
        data = BDGP("../data/")
        return train_test_split(data)

    elif dataset == "caltech20":
        data = Caltech20("../data/Caltech101-20.mat")
        return train_test_split(data)

    elif dataset == "handwritten":
        data = Handwritten("../data/handwritten.mat")
        return train_test_split(data)
    
    elif dataset == "reuters":
        data = Reuters("../data/Reuters_dim10.mat")
        return train_test_split(data)

    elif dataset == "noisy":
        data = AffNoisyMNIST("../data/noisymnist_train.npz")
        return affnoisy_mnist_split(data)

    elif dataset == "2d":
        data = Synthetic()
        return train_test_split(data)



