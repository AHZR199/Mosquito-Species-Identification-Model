from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use gpu if avaialble (only tested on gpu so far)

model = models.resnet50()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 67)


state_dict = torch.load('best_model.pth', map_location=device)

state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
])

label_map = {
    0: 'aedes_aegypti',
    1: 'aedes_albopictus',
    2: 'aedes_atlanticus',
    3: 'aedes_canadensis',
    4: 'aedes_cantator',
    5: 'aedes_condolescens',
    6: 'aedes_dorsalis',
    7: 'aedes_fairfax-1',
    8: 'aedes_flavescens',
    9: 'aedes_hendersoni',
    10: 'aedes_infirmatus',
    11: 'aedes_japonicus',
    12: 'aedes_mediovittatus',
    13: 'aedes_melanimon',
    14: 'aedes_nigromaculis',
    15: 'aedes_sierrensis',
    16: 'aedes_sollicitans',
    17: 'aedes_spilotus',
    18: 'aedes_sticticus',
    19: 'aedes_taeniorhynchus',
    20: 'aedes_tortilis',
    21: 'aedes_triseriatus_sl',
    22: 'aedes_trivittatus',
    23: 'aedes_vexans',
    24: 'anopheles_cf-coustani',
    25: 'anopheles_coustani',
    26: 'anopheles_crucians_sl',
    27: 'anopheles_freeborni',
    28: 'anopheles_funestus_sl',
    29: 'anopheles_gambiae_sl',
    30: 'anopheles_maculipalpis',
    31: 'anopheles_pharoensis',
    32: 'anopheles_pretoriensis',
    33: 'anopheles_pseudopunctipennis',
    34: 'anopheles_punctipennis',
    35: 'anopheles_quadrimaculatus',
    36: 'anopheles_rufipes',
    37: 'anopheles_squamosus',
    38: 'anopheles_tenebrosus',
    39: 'anopheles_ziemanni',
    40: 'coquillettidia_perturbans',
    41: 'culex_antillummagnorum',
    42: 'culex_bahamensis',
    43: 'culex_coronator',
    44: 'culex_erraticus',
    45: 'culex_nigripalpus',
    46: 'culex_pipiens_sl',
    47: 'culex_restuans',
    48: 'culex_salinarius',
    49: 'culex_tarsalis',
    50: 'culex_territans',
    51: 'culiseta_incidens',
    52: 'culiseta_inornata',
    53: 'culiseta_melanura',
    54: 'deinocerites_cancer',
    55: 'deinocerites_cuba-1',
    56: 'mansonia_titillans',
    57: 'orthopodomyia_signifera',
    58: 'psorophora_ciliata',
    59: 'psorophora_columbiae',
    60: 'psorophora_cyanescens',
    61: 'psorophora_discolor',
    62: 'psorophora_ferox',
    63: 'psorophora_howardii',
    64: 'psorophora_pygmaea',
    65: 'psorophora_signipennis',
    66: 'uranotaenia_sapphirina'
}
reverse_label_map = {v: k for k, v in label_map.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify():
    files = request.files.getlist('files')
    if not files:
        return 'No files part', 400

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)


            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0)
            image = image.to(device)

            # Inference
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = torch.max(probabilities) * 100
                predicted_index = predicted.item()
                predicted_species = reverse_label_map.get(predicted_index, "Unknown Species")
                confidence_value = confidence.item() if predicted_species != "Unknown Species" else 0

            results.append((filename, predicted_species, f"{confidence_value:.2f}"))
    return render_template('result.html', results=results)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888) #DO NOT CHANGE!!!!!!!
