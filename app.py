import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import timm  # Import timm for NASNet
import torch.nn.functional as F

# Define transforms for your dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class
class GrapeLeafDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load dataset
data_dir = "Grapevine_Leaves_Image_Dataset"  # Update with the correct path to your dataset
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory '{data_dir}' not found!")

classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
num_classes = len(classes)
print(f"Total Classes: {num_classes}, Sample Classes: {classes[:5]}")

# Extract image paths and labels
image_paths, labels = [], []
for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure valid image
            image_paths.append(img_path)
            labels.append(class_idx)

# Train-test split
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# Create dataset objects
train_dataset = GrapeLeafDataset(train_paths, train_labels, transform=transform)
test_dataset = GrapeLeafDataset(test_paths, test_labels, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = classes  # Corrected variable name


# Load Pretrained Architectures
def load_model(architecture, num_classes):
    architecture = architecture.lower()
    if architecture == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "vgg":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif architecture == "mobilenet":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif architecture == "densenet":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif architecture == "nasnet":
        model = timm.create_model("nasnetalarge", pretrained=True)
        if hasattr(model, "classifier"):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif hasattr(model, "last_linear"):
            model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
        else:
            raise AttributeError("Could not find the classification layer in NASNetALarge.")
    else:
        raise ValueError("Unknown architecture")
    return model

# Optimizer and Loss Function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Trained Models
@st.cache_resource
def load_trained_model(model_name):
    model = load_model(model_name, num_classes=len(class_names))  # Pass correct number of classes
    model.load_state_dict(torch.load(f"{model_name}.pth"))
    model.to(device)
    model.eval()  # Ensure the model is in evaluation mode
    return model

# Streamlit Sidebar
st.sidebar.title("Grape Leaf Classification")
page = st.sidebar.radio("Select a Page", ["Model Architecture", "Benchmarks", "Inference"])

# Function to display model summary dynamically
def display_model_summary(selected_model):
    if selected_model == "ResNet":
        st.subheader("ResNet - The Deep Learning Giant")
        st.write("""
        **ResNet (Residual Networks)** is a revolutionary architecture that made it possible to train **ultra-deep networks** without suffering from the vanishing gradient problem. 
        By introducing **skip connections** (residual blocks), ResNet allows the network to learn **residual mappings** instead of direct mappings. This results in **faster convergence** and **better performance** as the network becomes deeper. ResNet-50, one of the most popular variants, has **50 layers** and is known for its outstanding performance in **image classification** and **object recognition** tasks. 
        
        **Key Benefits:**
        - **Deeper networks** made feasible 🌐
        - **Faster convergence** with residual learning ⚡
        - **State-of-the-art performance** in image classification 🏆
        """)

    elif selected_model == "EfficientNet":
        st.subheader("EfficientNet - The Efficient Powerhouse")
        st.write("""
        **EfficientNet** takes efficiency to the next level! It introduces a **compound scaling method** to simultaneously scale the **depth**, **width**, and **resolution** of the network in a balanced way. This makes it possible to achieve **superior accuracy** with **fewer parameters**. 
        Whether it's **EfficientNet-B0** (the smallest variant) or **EfficientNet-B7** (the largest), this model pushes the limits of performance and efficiency. 🏋️‍♂️📉
        
        **Key Benefits:**
        - **Smarter scaling** for better accuracy 🌟
        - **Resource-efficient** design 🌱
        - **High accuracy** with **fewer parameters** 🔑
        """)

    elif selected_model == "VGG":
        st.subheader("VGG - The Elegant Simplicity")
        st.write("""
        **VGG16** is an iconic and **elegantly simple** architecture that uses **3x3 convolution filters** stacked in blocks. Despite its simplicity, VGG16 has achieved **impressive results** on numerous image classification tasks. The model's deep structure (16 layers) allows it to learn complex features. However, it is **computationally expensive** due to the **large number of parameters**. 
        
        **Key Benefits:**
        - **Simplicity at its best** 🧩
        - **Deep network** with **excellent feature learning** 🧠
        - **Great baseline** for comparison against other models ⚖️
        """)

    elif selected_model == "MobileNet":
        st.subheader("MobileNet - The Mobile-Friendly Efficiency")
        st.write("""
        **MobileNet** is designed for **mobile and embedded devices**, offering **lightweight** and **efficient** performance. It uses **depthwise separable convolutions**, which significantly reduce the **computational cost**. MobileNet enables **real-time processing** on **resource-constrained** devices, making it perfect for **mobile applications**, **IoT devices**, and **edge computing**. It achieves a **good balance** between **speed** and **accuracy**. 
        
        **Key Benefits:**
        - **Efficient** for **mobile and embedded systems** 💡
        - **Faster and lightweight** without compromising accuracy ⚡
        - **Great for real-time applications** ⏱️
        """)
    
    elif selected_model == "DenseNet":
        st.subheader("DenseNet - The Densely Connected Network")
        st.write("""
        **DenseNet** (Densely Connected Convolutional Network) introduces **dense connectivity** between layers, where each layer receives input from **all previous layers**. This ensures **better feature reuse**, **reduced parameters**, and **improved gradient flow**. It is particularly known for its **efficient parameter utilization** and **high performance on classification tasks**.

        **Key Benefits:**
        - **Dense connections** lead to better **gradient flow** 🎯
        - **Fewer parameters** while maintaining accuracy 📉
        - **Enhanced feature reuse** for better learning 🔄
        """)

    elif selected_model == "NASNet":
        st.subheader("NASNet - The AI-Designed Architecture")
        st.write("""
        **NASNet** (Neural Architecture Search Network) is an **AI-designed** deep learning architecture developed by **Google AI** using **Neural Architecture Search (NAS)**. This model was **automatically optimized** by AI instead of being manually engineered, resulting in **state-of-the-art performance** with **high efficiency**. 

        NASNet comes in two versions:
        - **NASNet-A Large**: High accuracy with more parameters  
        - **NASNet-A Mobile**: Optimized for mobile and embedded applications  

        **Key Benefits:**
        - **AI-optimized architecture** for superior performance 🤖  
        - **Efficient design** balancing accuracy and computational cost ⚡  
        - **Scalable** for both high-end and mobile applications 📱  
        """)

# Page 1: Model Architecture
if page == "Model Architecture":
    st.title("Model Architectures")
    selected_model = st.selectbox("Choose a Model", ["ResNet", "EfficientNet", "VGG", "MobileNet", "DenseNet", "NASNet"])
    model = load_trained_model(selected_model.lower())  # Load trained model
    # Load and display corresponding image from the "images" folder
    image_path = f"images/{selected_model.lower()}.png"
    if os.path.exists(image_path):
        st.image(image_path, caption=f"{selected_model} Architecture", use_column_width=True)
    else:
        st.warning("Architecture image not found!")
    # Display model summary dynamically
    display_model_summary(selected_model)
    st.text(str(model))

# Page 2: Benchmark Results
elif page == "Benchmarks":
    st.title("Benchmark Results")

    # Load and display the saved accuracy plot
    accuracy_plot_path = "grapevine_accuracy_plot.png"

    if os.path.exists(accuracy_plot_path):
        st.image(accuracy_plot_path, caption="Model Performance Comparison", use_column_width=True)
    else:
        st.write("Accuracy plot not found. Please ensure the file exists.")

    # Load and display the saved ROC curve plot
    roc_curve_path = "roc_curve.png"
    conf_matrix_image_path = "resnet_confusion_matrix.png"
    if os.path.exists(roc_curve_path):
        st.image(roc_curve_path, caption="ROC Curve", use_column_width=True)
    else:
        st.write("ROC curve plot not found. Please ensure the file exists.")

      # Load and display the confusion matrix image
    if os.path.exists(conf_matrix_image_path):
        st.image(conf_matrix_image_path, caption="Confusion Matrix", use_column_width=True)
    else:
        st.write("Confusion matrix plot not found. Please ensure the file exists.")

# Page 3: Inference
elif page == "Inference":
    st.title("Grape Leaf Classification Inference")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")  # Ensure the image is RGB
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_tensor = transform(img).unsqueeze(0).to(device)

        selected_model = st.selectbox("Choose a Model", ["ResNet", "EfficientNet", "VGG", "MobileNet", "DenseNet", "NASNet"])
        model = load_trained_model(selected_model.lower())  # Load trained model

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
            confidence, predicted = torch.max(probabilities, 1)  # Get highest probability

            # Get the predicted class index and map to the corresponding leaf name
            predicted_class = predicted.item()
            leaf_name = class_names[predicted_class]
            accuracy = confidence.item() * 100  # Convert to percentage

            # Display predicted grape leaf class and accuracy
            st.write(f"Predicted Grape Leaf Class: {leaf_name}")
            st.write(f"Confidence Level: {accuracy:.2f}%")  # Display accuracy percentage