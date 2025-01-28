import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image

# Initialiser l'application Flask
app = Flask(__name__)

def findConv2dOutShape(hin, win, conv, pool=2):
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = np.floor((hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    wout = np.floor((win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    if pool:
        hout /= pool
        wout /= pool
    return int(hout), int(wout)

# Définir l'architecture du modèle CNN_TUMOR
class CNN_TUMOR(nn.Module):
    def __init__(self, params):
        super(CNN_TUMOR, self).__init__()

        Cin, Hin, Win = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        # Convolution Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h, w = findConv2dOutShape(Hin, Win, self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv3)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv4)

        # Compute the flatten size
        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

# Paramètres du modèle
params_model = {
    "shape_in": (3, 256, 256),
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2
}

# Instancier le modèle
cnn_model = CNN_TUMOR(params_model)

# Définir l'approche matérielle (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnn_model.to(device)

# Charger les poids du modèle
model_path = "app/saved_model/weights.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Passer en mode évaluation

# Définir les transformations pour l'image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dictionnaire des classes
CLA_label = {0: 'Brain Tumor', 1: 'Healthy'}

# Route d'accueil pour afficher le formulaire d'upload d'image
@app.route('/')
def index():
    return render_template('web/index.html')
@app.route('/predire')
def predire():
    return render_template('web/sign-up.html')
# Route pour uploader une image et faire une prédiction
import base64
from io import BytesIO

@app.route('/commencer', methods=['POST'])
def commencer():
    if 'file' not in request.files:
        return "Aucun fichier uploadé", 400

    file = request.files['file']
    if file.filename == '':
        return "Aucun fichier sélectionné", 400

    try:
        # Ouvrir l'image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Convertir l'image en base64 pour l'afficher dans le template
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Appliquer les transformations
        img_tensor = transform(img).unsqueeze(0)  # Ajouter une dimension batch

        # Faire une prédiction
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_class = torch.max(output, 1)
            predicted_label = CLA_label[predicted_class.item()]

        # Passer l'image et la prédiction au template
        return render_template('web/article-details.html', 
                               prediction=predicted_label, 
                               img_data=img_str)

    except Exception as e:
        return f"Erreur lors de la prédiction : {e}", 500


# Route pour afficher les résultats
@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('web/article-details.html', prediction=prediction)

# Démarrer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
