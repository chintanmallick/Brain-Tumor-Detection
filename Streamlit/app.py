import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# ============================================================
# 🧠 CNN Architecture (EXACT SAME as your Colab training)
# ============================================================
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(100352, 512)   # matches your Colab model exactly
        self.fc2 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 100352)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ============================================================
# 🔹 Image preprocessing (same as training)
# ============================================================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


# ============================================================
# 🔥 Load trained model weights
# ============================================================
@st.cache_resource
def load_model():
    model = BrainTumorCNN()
    state_dict = torch.load(
        "model/brain_tumor_cnn_model_best.pth",
        map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Load model + preprocessing
model = load_model()
transform = get_transform()
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


# ============================================================
# 🔥 Grad-CAM implementation
# ============================================================
def generate_gradcam(model, img_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.conv3
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    outputs = model(img_tensor)
    _, pred_class = torch.max(outputs, 1)

    model.zero_grad()
    outputs[0, pred_class].backward()

    grad = gradients[0][0]
    act = activations[0][0]

    weights = torch.mean(grad, dim=(1, 2))
    cam = torch.zeros(act.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam.detach().numpy(), 0)
    cam -= cam.min()
    cam /= cam.max()

    return cam, pred_class.item()


# ============================================================
# 🔥 Overlay Grad-CAM heatmap
# ============================================================
def overlay_gradcam(original_img, cam):
    cam = cv2.resize(cam, (original_img.size[0], original_img.size[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 - cam * 255), cv2.COLORMAP_JET)
    img_np = np.array(original_img)
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return superimposed


# ============================================================
# 🎨 Streamlit App UI
# ============================================================
st.title("🧠 Brain Tumor MRI Classification (CNN + Grad-CAM)")
st.write("Upload an MRI image to classify the tumor type and view Grad-CAM visualization.")

uploaded = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", width=350)

    if st.button("🔍 Predict Tumor Type"):
        st.write("⏳ Processing image...")

        start = time.time()

        # Preprocess the image
        img_tensor = transform(img).unsqueeze(0)

        # Generate Grad-CAM
        cam, pred_idx = generate_gradcam(model, img_tensor)
        pred_class = classes[pred_idx]

        # Confidence scores
        with torch.no_grad():
            outputs = model(img_tensor)
            conf = torch.softmax(outputs, dim=1)[0] * 100

        # Overlay Grad-CAM
        heatmap_img = overlay_gradcam(img, cam)

        st.success(f"### 🧠 Predicted Tumor Type: **{pred_class.upper()}**")
        st.write(f"⏱️ Inference Time: {time.time() - start:.2f} seconds")

        st.subheader("📊 Model Confidence")
        st.bar_chart({classes[i]: conf[i].item() for i in range(4)})

        st.subheader("🔥 Grad-CAM Visualization")
        st.image(heatmap_img, width=450)
