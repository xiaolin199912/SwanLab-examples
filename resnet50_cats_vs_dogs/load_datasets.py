import csv
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    def __init__(self, csv_file, image_size=(128, 128)):
        self.csv_file = csv_file
        self.image_size = image_size
        self.data = self.load_data()
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.current_dir = os.path.dirname(os.path.abspath(__file__))


    def load_data(self):
        with open(self.csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            data = list(csv_reader)
        return data

    def preprocess_image(self, image_path):
        image_path = os.path.join(self.current_dir+'/datasets', image_path)
        # 读取图像
        image = Image.open(image_path)
        # 使用Torchvision的transform进行图像预处理
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = self.preprocess_image(image_path)
        return image, int(label)

