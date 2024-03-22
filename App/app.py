# created by abdullah zubair for honours undergraduate thesis (university of calgary 2024)
# part of the soghigian lab (UCVM)  
# linkedin: https://www.linkedin.com/in/a-zubair-calgary/


#Flask web application that uses a trained machine learning model to identify mosquito species from uploaded images
#the application allows users to upload one or more images and displays the top 5 predicted species for each image
#it uses a pre-trained ResNet50 model that has been fine-tuned on a dataset of mosquito images
#the model is loaded from a saved weights file and used for inference on the uploaded images


from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'  #directory to store uploaded images
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  #maximum file size allowed (16MB) CAN BE CHANGED

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU

#load the pre-trained ResNet50 model and modify the last layer for mosquito species classification
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 70)  #70 is the number of mosquito species used for the project PLEASE CHANGE AS NEEDED

model_weight_path = 'best_model.pth'  #path to the saved model weights file
state_dict = torch.load(model_weight_path, map_location=device) 
model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})  #loadING the weights into the model

model = model.to(device) 
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

#dict maps the model's output indices to the corresponding mosquito species names
label_map = {
    0: 'aedes_aegypti',
    1: 'aedes_albopictus',
    2: 'aedes_atlanticus',
    3: 'aedes_canadensis',
    4: 'aedes_cantator',
    5: 'aedes_cinereus',
    6: 'aedes_condolescens',
    7: 'aedes_dorsalis',
    8: 'aedes_excrucians',
    9: 'aedes_fairfax-1',
    10: 'aedes_flavescens',
    11: 'aedes_hendersoni',
    12: 'aedes_infirmatus',
    13: 'aedes_japonicus',
    14: 'aedes_mediovittatus',
    15: 'aedes_melanimon',
    16: 'aedes_nigromaculis',
    17: 'aedes_pullatus',
    18: 'aedes_sierrensis',
    19: 'aedes_sollicitans',
    20: 'aedes_spilotus',
    21: 'aedes_sticticus',
    22: 'aedes_taeniorhynchus',
    23: 'aedes_tortilis',
    24: 'aedes_triseriatus_sl',
    25: 'aedes_trivittatus',
    26: 'aedes_vexans',
    27: 'anopheles_cf-coustani',
    28: 'anopheles_coustani',
    29: 'anopheles_crucians_sl',
    30: 'anopheles_freeborni',
    31: 'anopheles_funestus_sl',
    32: 'anopheles_gambiae_sl',
    33: 'anopheles_maculipalpis',
    34: 'anopheles_pharoensis',
    35: 'anopheles_pretoriensis',
    36: 'anopheles_pseudopunctipennis',
    37: 'anopheles_punctipennis',
    38: 'anopheles_quadrimaculatus',
    39: 'anopheles_rufipes',
    40: 'anopheles_squamosus',
    41: 'anopheles_tenebrosus',
    42: 'anopheles_ziemanni',
    43: 'coquillettidia_perturbans',
    44: 'culex_antillummagnorum',
    45: 'culex_bahamensis',
    46: 'culex_coronator',
    47: 'culex_erraticus',
    48: 'culex_nigripalpus',
    49: 'culex_pipiens_sl',
    50: 'culex_restuans',
    51: 'culex_salinarius',
    52: 'culex_tarsalis',
    53: 'culex_territans',
    54: 'culiseta_incidens',
    55: 'culiseta_inornata',
    56: 'culiseta_melanura',
    57: 'deinocerites_cancer',
    58: 'deinocerites_cuba-1',
    59: 'mansonia_titillans',
    60: 'orthopodomyia_signifera',
    61: 'psorophora_ciliata',
    62: 'psorophora_columbiae',
    63: 'psorophora_cyanescens',
    64: 'psorophora_discolor',
    65: 'psorophora_ferox',
    66: 'psorophora_howardii',
    67: 'psorophora_pygmaea',
    68: 'psorophora_signipennis',
    69: 'uranotaenia_sapphirina'
}

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
            filename = secure_filename(file.filename)  # sanitize the filename for security!
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  
            file.save(img_path) 

            image = Image.open(img_path).convert('RGB')  #
            image = transform(image).unsqueeze(0)  
            image = image.to(device)  

            with torch.no_grad():  
                outputs = model(image)  
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] #softmaxxxxxx
                top_probs, top_preds = torch.topk(probabilities, 5)  #getsthe top 5 probabilities and predicted classes
                top_species = [(label_map[idx], prob.item() * 100) for idx, prob in zip(top_preds.tolist(), top_probs)]  #maps the predicted classes to species names and probabilities

            results.append((filename, top_species))  #

    return render_template('result.html', results=results) 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}  #only certain files allowed

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename) 

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8888)  #Run the Flask app on host 0.0.0.0 and port 8888 CAN BE CHANGED AS NEEDED
