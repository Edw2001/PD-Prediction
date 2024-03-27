import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights
import os


preprocess = transforms.Compose([
    transforms.Resize(256),                          
    transforms.CenterCrop(224),                      
    transforms.ToTensor(),                           
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()

resnet.eval()

if torch.cuda.is_available():
    resnet.cuda()


def load_data(directory):
    dataset = datasets.ImageFolder(directory, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    return dataloader


def extract_features(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label_batch in dataloader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            feature_batch = resnet(inputs)
            features.extend(feature_batch.cpu().numpy())
            labels.extend(label_batch.numpy())
    return features, labels


dataloader = load_data('D:\\PaHaW\\PaHaW\\PaHaW\\PaHaW_visualization')

features,labels = extract_features(dataloader)
pd_features = [feature for feature, label in zip(features, labels) if label == 0]
health_features = [feature for feature, label in zip(features, labels) if label == 1]
print(dataloader.dataset.classes)
print(len(health_features))


