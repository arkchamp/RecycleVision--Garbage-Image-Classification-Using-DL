import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="RecycleVision",
    page_icon="♻️",
    layout="wide"
)

# ------------------------------
# Class Names (match dataset order)
# ------------------------------
class_names = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

# ------------------------------
# Model Definition (ResNet50 Custom Head)
# ------------------------------
class ResNet50_Custom(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base_model = models.resnet50(weights=None)  # no weights since we load state_dict
        num_features = base_model.fc.in_features
        base_model.fc = nn.Identity()
        self.features = base_model
        self.fc1 = nn.Linear(num_features, 512)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    model = ResNet50_Custom(num_classes=len(class_names))
    model_path = r"C:\Users\arkha\jupyter-workspace\recycle_vision_project\models\resnet50_best.pth"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ------------------------------
# Image Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image):
    img_tensor = transform(image).unsqueeze(0)  # shape [1,3,224,224]
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top3_prob, top3_idx = torch.topk(probs, 3)
    return [(class_names[idx], prob.item()) for idx, prob in zip(top3_idx, top3_prob)]


# Streamlit pages and navigation
st.sidebar.title("RecycleVision Navigation")
page = st.sidebar.radio("Go to", ["💡 Intro", "🔎 Prediction", "👤 About Me"])

# Intro Page
if page == "💡 Intro":
    st.title("♻️ RecycleVision: Garbage Image Classification")
    st.markdown("""
    Welcome to **RecycleVision**!  
    
    This project classifies waste into **10 categories** using a deep learning model (ResNet50).  
    ↳ Built a **deep learning model** that classifies images of waste into categories like plastic, metal, glass, paper and organic etc.  
    
    This system will assist in automating recycling by sorting garbage based on image input, using a deep learning model deployed via a simple user interface.


    - Dataset: Garbage Classification (10 classes)  
    - Models Trained: ResNet50, MobileNetV2, EfficientNetB0, GoogleNet, VGG16  
    - Final Best Model: **ResNet50**  

    Lets dive into the **Prediction page** to try it out!
    """)

elif page == "🔎 Prediction":
    st.title("🔍 Garbage Image Prediction")

    uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])
    pasted_url = st.text_input("📋 Or paste an image URL here:")

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    elif pasted_url:
        try:
            response = requests.get(pasted_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("⚠️ Could not load image from pasted URL")

    if image is not None:
        st.image(image, caption="Input Image", width=400)

        st.write("Classifying...")
        preds = predict(image)

        st.success(f"**Predicted Class:** {preds[0][0]} ({preds[0][1]*100:.2f}%)")

        st.write("### Top-3 Predictions")
        for cls, prob in preds:
            st.write(f"- {cls}: {prob*100:.2f}%")


# About Us Page
elif page == "👤 About Me":
    st.title("👨‍💻 About Me")
    st.markdown("""
    **Abdullah Khatri**   
      
    • **Data Science learner** — working on RecycleVision - Garbage Classification  
    ↳ DL models, Transfer Learning and Streamlit apps
    • This project was built as part of the mini-project series to apply deep learning concepts.  
    
    • This app demonstrates a simple multipages as requested by my mentor, Wokring fine as expected with the accuracy.
    """)
