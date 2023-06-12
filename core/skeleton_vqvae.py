import torch
import torch.nn as nn

# CODEBOOK
class Codebook(nn.Module):
    
    def __init__(self, codebook_size, codebook_dim, reset_clock, decay):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.reset_clock = reset_clock
        
        self.register_buffer('codebook', torch.randn(codebook_size, codebook_dim))
        self.register_buffer('ema_N', torch.zeros(codebook_size))
        self.register_buffer('ema_m', torch.zeros(codebook_size, codebook_dim))
        self.register_buffer('usage_times', torch.full((codebook_size,), reset_clock))
        
        self.latest_indices = None
        
        
    def forward(self, z):
        """
        Map raw encodings to nearest codebook vectors
        """
 
        with torch.no_grad():
        
            # (B, 256, 8) -> (B*256, 8)
            z_squash = z.view(-1, self.codebook_dim)

            # ((B*256, K, 8) - (B*256, K, 8))**2 => (B*256, K)
            distances = torch.linalg.norm(z_squash[:, None, :] - self.codebook[None, :, :], dim=-1)

            # (B*256, K) -> (B*256,)
            nearest_indices = torch.argmin(distances, dim=-1)

            # (B*256, 8) -> (B, 256, 8)
            nearest_vectors = self.codebook[nearest_indices, :].view(*z.shape)
            
            self.latest_indices = nearest_indices

        return nearest_vectors
    
        
    def update(self, z):
        """
        Update codebook vectors via EMA
        """
    
        assert self.training and self.latest_indices is not None
        
        with torch.no_grad():
        
            z_squash = z.view(-1, self.codebook_dim)

            # 0 ~ (B*256, K)
            one_hot = torch.zeros((z_squash.size(0), self.codebook_size), device=z.device)

            # (B*256, K) <~ (B*256, 1)
            one_hot.scatter_(1, self.latest_indices.unsqueeze(1), 1)

            # (B*256, K) -> (K,)
            n = one_hot.sum(dim=0)

            # (K, B*256) @ (B*256, 8)
            m = torch.matmul(one_hot.t(), z_squash)

            # N := N * γ + n * (1 - γ)
            self.ema_N = self.ema_N * self.decay + n * (1 - self.decay)

            # m := m * γ + m' * (1 - γ)
            self.ema_m = self.ema_m * self.decay + m * (1 - self.decay)

            # e := m / N
            self.codebook = self.ema_m / (self.ema_N.unsqueeze(-1) + 1e-15)
            
            """
            Codebook Resets
                Reset codes unused for self.reset_clock times to random encoding vectors
            """
            
            # (K,) -> (a, 1) -> (a,)
            used = (n > 0).nonzero().squeeze(-1)
            
            # (K,) -> (b, 1) -> (b,)
            unused = (n == 0).nonzero().squeeze(-1)
            
            self.usage_times[used] = self.reset_clock
            
            self.usage_times[unused] -= 1
            
            under_utilized = (self.usage_times == 0).nonzero().squeeze(-1)
            
            if len(under_utilized) > 0:
                
                reset_indices = torch.randint(z_squash.size(0), size=(len(under_utilized),), device=z.device)
                self.codebook[under_utilized] = z_squash[reset_indices]
                self.usage_times[under_utilized] = self.reset_clock
                

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
        x = self.conv10(x)
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


# VQVAE
#     BOTTLENECK: 256x8
#     PARAMETERS: 1,700,000
class VQVAE(nn.Module):
    
    def __init__(self, codebook_size, codebook_dim, num_mappings, reset_clock, decay=0.99):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.codebook = Codebook(codebook_size, codebook_dim, reset_clock, decay)
        self.num_mappings = num_mappings
    
    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        return z
    
    def quantize(self, z: torch.Tensor):
        z_Q = self.codebook(z)
        return z_Q + (z - z.detach())         # Straight-Through Estimator
    
    def decode(self, z_Q: torch.Tensor):
        return self.decoder(z_Q)
    
    def forward(self, x: torch.Tensor):
        return self.decode(self.quantize(self.encode(x)))
    
    def generate(self, batch_size: int):
        indices = torch.randint(high=self.codebook.codebook_size, size=(batch_size, self.num_mappings), \
                                device=self.codebook.codebook.device)
        z = self.codebook.codebook[indices]
        return self.decode(z)