"""SegmentationNN"""
import torch
import torch.nn as nn

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SegmentationNN(nn.Module):
    def __init__(self, num_classes=23, hp=None):
        super().__init__()

        import torchvision.models as models
        self.hp = hp or {}
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        # Use MobileNetV2 as pretrained encoder (efficient and < 5M params)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Freeze early layers to prevent overfitting on small dataset
        self.encoder = mobilenet.features  # Output: (N, 1280, H/32, W/32)
        
        # Optionally freeze some encoder layers
        for param in list(self.encoder.parameters())[:50]:
            param.requires_grad = False
        
        # Decoder: Upsample back to original resolution
        self.decoder = nn.Sequential(
            # (N, 1280, 7, 7) -> (N, 256, 15, 15)
            nn.ConvTranspose2d(1280, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # (N, 256, 15, 15) -> (N, 128, 30, 30)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # (N, 128, 30, 30) -> (N, 64, 60, 60)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # (N, 64, 60, 60) -> (N, 32, 120, 120)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # (N, 32, 120, 120) -> (N, 32, 240, 240)
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final classification layer
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network.
        
        Args:
            x: input tensor of shape (N, C, H, W)
        Returns:
            output tensor of shape (N, num_classes, H, W)
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        # Store original size for potential resizing
        input_size = x.shape[2:]  # (H, W)
        
        # Encode
        features = self.encoder(x)
        
        # Decode
        out = self.decoder(features)
        
        # Ensure output matches input spatial dimensions
        if out.shape[2:] != input_size:
            out = nn.functional.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################


class DummySegmentationModel(nn.Module):
    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1
        self.prediction = _to_one_hot(target_image, num_classes=23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")