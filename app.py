import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = z * self.label_embed(labels)
        out = self.net(x)
        return out.view(-1, 1, 28, 28)

model = Generator()
model.load_state_dict(torch.load("generator_mnist.pth", map_location='cpu'))
model.eval()

st.title("🖊️ MNIST Handwritten Digit Generator")
digit = st.selectbox("Choose a digit (0–9)", list(range(10)))

if st.button("Generate"):
    z = torch.randn(5, 100)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        images = model(z, labels).squeeze().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i], cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
