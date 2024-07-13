import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn.functional as F

from src.c_model import Classifier

if __name__ == "__main__":
    main_tfm = transforms.Compose([
        transforms.ToTensor(),
    ])

    model = Classifier()
    model.load_state_dict(torch.load("./models/model_best.pth"))
    model.eval()

    img = Image.open("./Train/cocci.4.jpg").convert("RGB")
    #Image.open("test.jpg").convert("RGB")
    img = main_tfm(img).unsqueeze(0)

    pred = model(img)
    mappingx = {
        'Coccidiosis': 0,
        'Healthy': 1,
        'New Castle Disease': 2,
        'Salmonella': 3
    }
    softmax = F.softmax(pred, dim=1).tolist()[0]
    softmax = {k: softmax[v] for k, v in mappingx.items()}
    for k, v in softmax.items():
        print(k, ": " + str(round(v*100, 2)) + "%")