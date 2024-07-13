import torchvision.transforms as transforms

from src.c_model import Classifier

if __name__ == "__main__":
    main_tfm = transforms.Compose([
        transforms.ToTensor(),
    ])

    model = Classifier()
    print(model)