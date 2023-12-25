import torch
import torchvision
import swanlab
from torch.utils.data import DataLoader
from load_datasets import DatasetLoader
import os


if __name__ == "__main__":
    num_epochs = 10
    lr = 1e-4
    batch_size = 8
    image_size = 256
    num_classes = 2

    # Initialize swanlab
    swanlab.init(
        experiment_name="ResNet50",
        description="Train ResNet50 for cat and dog classification.",
        config={
            "datasets": "cats_and_dogs",
            "optim": "Adam",
            "lr": lr,
            "criterion": "torch.nn.CrossEntropyLoss",
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "image_size": image_size,
            "num_class": num_classes,
        }
    )

    TrainDataset = DatasetLoader("datasets/train.csv", image_size=(image_size, image_size))
    ValDataset = DatasetLoader("datasets/val.csv", image_size=(image_size, image_size))
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(ValDataset, batch_size=1, shuffle=False)

    # Load pre-trained model.
    model = torchvision.models.resnet50(pretrained=True)
    # Replace the last fully connected layer.
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    # Train
    device = torch.device('mps')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for iter, (inputs, labels) in enumerate(TrainDataLoader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, iter + 1,
                                                                          len(TrainDataLoader), loss.item()))
            swanlab.log({"train_loss": loss.item()})

    # Test
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in ValDataLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: {:.2f}%'.format(correct / total * 100))
    swanlab.log({"validation_acc": correct / total * 100})

    # Save Checkpoint
    if not os.path.exists("checkpoint"):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs("checkpoint")

    torch.save(model.state_dict(), 'checkpoint/lastest_checkpoint.pth')
    print("The checkpoint has been saved in './checkpoint'")