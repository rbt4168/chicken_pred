# Chicken Disease Classification

This project is a deep learning-based classifier for distinguishing between different types of chicken. It utilizes PyTorch for model building and training, and includes a custom dataset class and a classifier model.

![](/confusion_matrixx.png)

## Training

The training script trains a classifier model on the provided dataset. You can adjust the hyperparameters such as learning rate, batch size, number of epochs, etc., directly in the `train.py` script.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
split_ratio = 0.8
n_epochs = 100
n_batch = 32
lr = 3e-4
weight_decay = 1e-5
```

## Model

The `Classifier` model is defined in `src/c_model.py`. It is a simple convolutional neural network (CNN) designed for image classification tasks.

```python
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # Define the network layers here

    def forward(self, x):
        # Define the forward pass
        return x
```

## Dataset

The `ChickenDataset` class is defined in `src/c_dataset.py`. It reads image paths and labels from a CSV file and applies necessary transformations.

The dataset can be obtained from [Kaggle: Chicken Disease](https://www.kaggle.com/datasets/allandclive/chicken-disease-1).

```python
class ChickenDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image and label, apply transforms, and return
        return image, label

    def get_label_map(self):
        # Return a mapping of labels
        return label_map
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
