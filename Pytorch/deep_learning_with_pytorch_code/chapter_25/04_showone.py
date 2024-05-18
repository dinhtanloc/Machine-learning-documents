import matplotlib.pyplot as plt
import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

plt.imshow(trainset.data[7])
plt.show()
