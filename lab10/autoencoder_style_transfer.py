import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set the paths for content and style images
content_image_path = '../data/content_image.jpg'
style_image_path = '../data/style_image.jpg'
output_path = 'output_image.jpg'

# Set other parameters
image_size = (256, 256)
latent_dim = 128
learning_rate = 1e-3
epochs = 80

# Load and preprocess the content and style images
preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

def load_image(image_path, preprocess):
    image = Image.open(image_path).convert('RGB')
    #image pre-process + adding batch dimension
    image = preprocess(image).unsqueeze(0) 
    return image

content_image = load_image(content_image_path, preprocess)
style_image = load_image(style_image_path, preprocess)

# Create the autoencoder model
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 64 * 128, latent_dim)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 64 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 64, 64) 
        x = nn.ReLU()(self.deconv1(x))
        x = nn.ReLU()(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)

# Define the Gram matrix calculation
def gram_matrix(features):
    batch_size, num_channels, height, width = features.size()
    features = features.view(batch_size * num_channels, height * width)
    #Inner product
    gram = torch.matmul(features, features.t())
    #Outer product
    #gram = torch.matmul(features.transpose(1, 2), features)
    gram = gram / (num_channels * height * width) 
    return gram

# Define a helper function to get features from intermediate layers
def get_features(image, layers):
    model = models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    #Compare model._modules.items() with this site
    #https://convnetplayground.fastforwardlabs.com/#/models
    features = {}
    x = image
    for name, module in model._modules.items():
        x = module(x)
        if name in layers:
            features[name] = x
    return features

# Define the style transfer loss function
def style_transfer_loss(generated_image, images):
    style_image = images[0]
    content_image = images[1]
    
    # Get features from the intermediate layers of the model
    layers = ['28']
    generated_image_features = get_features(generated_image, layers)
    style_image_features = get_features(style_image, layers)

    #Style loss 1: Gram matrix from vgg
    gram_style = {}
    gram_generated = {}
    for layer_name, generated_features in generated_image_features.items():
        gram_generated[layer_name] = gram_matrix(generated_features)
        style_features = style_image_features[layer_name]
        gram_style[layer_name] = gram_matrix(style_features)
    # Compute the Gram matrix style loss
    gm_style_loss = 0.0
    for layer_name in layers:
        gm_style_loss += torch.mean(torch.square(gram_generated[layer_name] - gram_style[layer_name]))
    print('  Gram matrix style loss:',gm_style_loss.item())

    #Content loss 2: Output pixel values
    #pix_style_loss = torch.mean(torch.square(style_image - generated_image))
    #print('  Pixel output style loss:',pix_style_loss.item())

    #Content loss 1: Pre-trained VGG embeddings
    #content_features = get_features(content_image, layers[1])
    #vgg_content_loss = torch.mean(torch.square(content_features[layers[1]] - generated_features[layers[1]]))
    #print('  VGG embeddings content loss:',vgg_content_loss.item())
    
    #Content loss 2: Output pixel values
    pix_content_loss = torch.mean(torch.square(content_image - generated_image))
    print('  Pixel output content loss:',pix_content_loss.item())
    
    total_loss = (gm_style_loss*100)+(pix_content_loss*0.01)#+(pix_style_loss)
    return total_loss

# Create the autoencoder model
class StyleTransferModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(StyleTransferModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent_vector = self.encoder(x)
        decoded_output = self.decoder(latent_vector)
        return decoded_output

# Create an instance of the StyleTransferModel
model = StyleTransferModel(encoder, decoder)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Early stopping vars
patience = 10
best_loss = float('inf')
epochs_without_improvement = 0

# Training loop
loss_history = []
for epoch in range(epochs):
    # Forward pass
    output_image = model(content_image)
    
    # Compute the style transfer loss
    loss = style_transfer_loss(output_image, [style_image, content_image])
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Track the loss history
    loss_history.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        
    # Save the output image
    output_image = output_image.squeeze(0).permute(1, 2, 0).detach().numpy()
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_image)
    output_image.save(f"output_image_epoch_{epoch+1}.jpg")

    #Early Stopping
    #if loss.item() < best_loss:
    if best_loss / loss.item() > 1.006  :
        best_loss = loss.item()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print(f"Early stopping")
        break

output_image = model(content_image).detach()

# Save the output image
output_image = output_image.squeeze(0).permute(1, 2, 0).numpy()
output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
output_image = Image.fromarray(output_image)
output_image.save(output_path)

# Plot the training curve
plt.plot(loss_history)
plt.title('Style Transfer Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()