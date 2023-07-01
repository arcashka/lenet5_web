import lenet5
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
from PIL import Image

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 20

N_CLASSES = 10


# define transforms
# transforms.ToTensor() automatically scales the images to [0,1] range
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data',
                               train=True,
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data',
                               train=False,
                               transform=transforms)

# define the data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)


parser = argparse.ArgumentParser(
    prog='LeNet5'
)
parser.add_argument('-l', '--load')
parser.add_argument('-s', '--save')
parser.add_argument('--mnist', type=int)
parser.add_argument('--image')

args = parser.parse_args()

model = lenet5.LeNet5(N_CLASSES).to(DEVICE)

if args.load is not None:
    model.load_state_dict(torch.load(args.load))
else:
    torch.manual_seed(RANDOM_SEED)

    print(f"Using {DEVICE} device")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, _ = lenet5.training_loop(
        model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

if args.save is not None:
    torch.save(model.state_dict(), args.save)

if args.mnist is not None:
    model.eval()
    subset = Subset(valid_dataset, [args.mnist])
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    x, y = next(iter(loader))
    print(f'x size: {x.shape}')

    mnist_data = datasets.MNIST(root='mnist_data', train=False)
    image_tensor, _ = mnist_data[args.mnist]
    image_array = np.array(image_tensor)
    image_array = image_array.reshape(28, 28)
    with torch.no_grad():
        x = x.to(DEVICE)
        prediction = model(x)
        labels = torch.argmax(prediction[0], 1)
        print(f"predicted: {labels[0]}, actual: {y[0]}")

if args.image is not None:
    image = Image.open(args.image)
    image = image.resize((32, 32))
    image = image.convert("L")
    image = np.array(image)
    # image = image / 255.0

    image_to_show = Image.fromarray(image, mode='L')
    scaled_image = image_to_show.resize(
       (int(image_to_show.width * 50), int(image_to_show.height * 50)))
    # scaled_image.show()

    image = image.reshape((1, 1, 32, 32))
    print(f"image size: {image.shape}")
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.to(DEVICE)
    prediction = model(image_tensor)
    labels = torch.argmax(prediction[0], 1)
    print(f"predictions: {prediction[0]}")
    print(f"predicted: {labels[0]}")

