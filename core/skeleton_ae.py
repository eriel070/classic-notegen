import torch
import torch.nn as nn

# ENCODER:
class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Encoder layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=11, stride=2, padding=5)  # Dimension: 1x8000 -> 32x4000
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)  # Dimension: 32x4000 -> 64x2000
        self.conv3 = nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1)  # Dimension: 64x2000 -> 64x1000
        self.conv4 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)  # Dimension: 64x1000 -> 128x500
        self.conv5 = nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)  # Dimension: 128x500 -> 128x250
        self.conv6 = nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)  # Dimension: 128x250 -> 128x125
        self.conv7 = nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=3)  # Dimension: 128x125 -> 256x64
        self.conv8 = nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1)  # Dimension: 256x64 -> 256x32
        self.conv9 = nn.Conv1d(256, 256, kernel_size=2, stride=2, padding=0)  # Dimension: 256x32 -> 256x16
        self.conv10 = nn.Conv1d(256, 256, kernel_size=2, stride=2, padding=0)  # Dimension: 256x16 -> 256x8
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.bn10(self.conv10(x))          # Notice effects of batch norm on encoding, but do not relu 
        return x
    

# DECODER:
class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
         # Decoder layers
        self.deconv1 = nn.ConvTranspose1d(256, 256, kernel_size=2, stride=2, padding=0)  # Dimension: 256x8 -> 256x16
        self.deconv2 = nn.ConvTranspose1d(256, 256, kernel_size=2, stride=2, padding=0)  # Dimension: 256x16 -> 256x32
        self.deconv3 = nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1)  # Dimension: 256x32 -> 256x64
        self.deconv4 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=3, output_padding=1) # Dimension: 256x64 -> 256x125
        self.deconv5 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)  # Dimension: 256x125 -> 128x250  
        self.deconv6 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)  # Dimension: 128x250 -> 128x500
        self.deconv7 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)  # Dimension: 128x500 -> 64x1000
        self.deconv8 = nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1)  # Dimension: 64x1000 -> 64x2000
        self.deconv9 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1) # Dimension: 64x2000 -> 32x4000
        self.deconv10 = nn.ConvTranspose1d(32, 1, kernel_size=11, stride=2, padding=5, output_padding=1) # Dimension: 32x4000 -> 1x8000
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(64)
        self.bn9 = nn.BatchNorm1d(32)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        x = self.relu(self.bn6(self.deconv6(x)))
        x = self.relu(self.bn7(self.deconv7(x)))
        x = self.relu(self.bn8(self.deconv8(x)))
        x = self.relu(self.bn9(self.deconv9(x)))
        x = self.tanh(self.deconv10(x))
        return x


# AUTOENCODER:
#     BOTTLENECK: 256x8
#     PARAMETERS: 1,700,000
class AutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def encode(self, x: torch.Tensor):
        return self.encoder(x)
    
    def decode(self, x: torch.Tensor):
        return self.decoder(x)
    
    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))