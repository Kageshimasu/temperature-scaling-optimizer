import torch
import torchvision
import torchvision.models as models


def main():
    data_path = './data'
    batch_size = 34
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR100(data_path, transform=transforms, train=True, download=True)
    model = models.resnet18(pretrained=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for inputs, labels in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        optimizer.zero_grad()
        a = model(inputs)
        loss = criterion(a, labels)
        loss.backward()
        optimizer.step()
        print(float(loss))
    torch.save(model.state_dict(), './model.pth')


if __name__ == '__main__':
    main()
