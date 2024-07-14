import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.c_model import Classifier
from src.c_dataset import ChickenDataset

if __name__ == "__main__":
    main_tfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Classifier().to(device)
    model.load_state_dict(torch.load("./models/model_best.pth"))
    model.eval()

    data = ChickenDataset("./train_data.csv", tfm=main_tfm)
    mappingx = data.get_label_map()
    dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

    confusion_matrix = np.zeros((len(mappingx), len(mappingx)))
    sum_entropy = 0
    for img, label in tqdm(dataloader):
        pred = model(img.to(device))
        prob = F.softmax(pred, dim=1)
        _, pred = torch.max(pred, 1)
        entropy = -torch.sum(prob * torch.log(prob), dim=1)
        sum_entropy += torch.sum(entropy).item()
        for i in range(len(label)):
            confusion_matrix[label[i]][pred[i]] += 1
            
    print("average entropy:", sum_entropy / len(data))

    print("accuracy:", np.trace(confusion_matrix) / np.sum(confusion_matrix))

    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)


    plt.figure(figsize=(10, 7))
    plt.imshow(confusion_matrix, cmap="Blues")
    # print number in each cell
    for i in range(len(mappingx)):
        for j in range(len(mappingx)):
            plt.text(j, i, str(round(confusion_matrix[i, j]*100, 2))+"%", ha='center', va='center', color='black' if confusion_matrix[i, j] < 0.5 else 'white')
    
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(mappingx)), mappingx.keys())
    plt.yticks(range(len(mappingx)), mappingx.keys())
    plt.savefig("confusion_matrixx.png")
    plt.show()

    # img = Image.open("./Train/cocci.5.jpg").convert("RGB")
    #Image.open("test.jpg").convert("RGB")
    # img = main_tfm(img).unsqueeze(0)

    # pred = model(img)
    # softmax = F.softmax(pred, dim=1).tolist()[0]
    # softmax = {k: softmax[v] for k, v in mappingx.items()}
    # for k, v in softmax.items():
    #     print(k, ": " + str(round(v*100, 2)) + "%")