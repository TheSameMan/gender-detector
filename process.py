from torch.nn import Module, Conv2d, MaxPool2d, Linear, Dropout
from torch.nn.functional import relu, log_softmax
from torch import load, cuda
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from argparse import ArgumentParser
from PIL import Image
from glob import glob
import os
import sys
import json


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        # 128x128x3
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        # 126x126x32
        self.conv2 = Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        # 124x124x32
        self.max_pool1 = MaxPool2d(2, 2)
        
        # 62x62x32
        self.conv3 = Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        # 60x60x32
        self.conv4 = Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        # 58x58x32
        self.max_pool2 = MaxPool2d(2, 2)
         
        # 29x29x32
        self.conv5 = Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        # 27x27x64
        self.conv6 = Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        # 25x25x64
        self.max_pool3 = MaxPool2d(2, 2)

        # 12x12x64
        self.conv7 = Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        # 10x10x64
        self.conv8 = Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        # 8x8x64
        self.max_pool4 = MaxPool2d(2, 2)

        # 4x4x64
        self.fc1 = Linear(4*4*64, 512)        
        self.drop = Dropout(0.2)
        self.fc2 = Linear(512, 2)
        
    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = self.max_pool1(x)
            
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = self.max_pool2(x)

        x = relu(self.conv5(x))
        x = relu(self.conv6(x))
        x = self.max_pool3(x)

        x = relu(self.conv7(x))
        x = relu(self.conv8(x))
        x = self.max_pool4(x)
             
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.drop(x)        
        x = self.fc2(x)
        x = log_softmax(x, -1)
            
        return x


def predict(img_path, model, transform, use_cuda):
    image = Image.open(img_path).convert('RGB')# convert for gray images
    image = transform(image).float().unsqueeze(0)

    if use_cuda:
        image = image.cuda()

    model.eval()

    index = model(image).data.cpu().numpy().argmax()# select max class value

    return index

if __name__ == '__main__':
    model = Net()

    model_path = 'model.pt'
    use_cuda = cuda.is_available()
    try:
        if use_cuda:
            location = lambda storage: storage.cuda()
            model.cuda()
        else:
            location = 'cpu'
        model.load_state_dict(load(model_path, map_location=location))
    except FileNotFoundError:
        print('Model not found!')

    transform = Compose([Resize([128, 128]),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])

    parser = ArgumentParser()
    parser.add_argument('folder', type=str)
    args = parser.parse_args(sys.argv[1:])

    img_folder = args.folder
    names = [os.path.basename(x) for x in glob(img_folder + '/*')]

    classes = ['female', 'male']
    dic = {}
    for idx, file in enumerate(names):
        index = predict(img_folder + file, model, transform, use_cuda)
        dic[file] = classes[index]

    with open('process_results.json', 'w') as file:
        json.dump(dic, file, indent=4)
