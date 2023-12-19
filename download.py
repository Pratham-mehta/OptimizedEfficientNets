import torchvision

# Specify the root directory where the dataset will be downloaded.
root_dir = './data'

# Create the Food101 dataset object.
# This will automatically download the dataset if it is not already present.
food101_dataset = torchvision.datasets.Food101(root=root_dir, download=True)

print("Download complete!")