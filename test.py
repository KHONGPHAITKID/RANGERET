import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba

# ---------------------------------------------------------
# 1. Circular Padding Convolution (Tránh mất thông tin ở mép ảnh 360 độ)
# ---------------------------------------------------------
class CircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.padding = padding

    def forward(self, x):
        # Pad chiều ngang (Width - góc quay 360 độ) bằng circular
        # Pad chiều dọc (Height) bằng constant/replicate
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, self.padding, self.padding), mode='constant', value=0)
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = CircularConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# ---------------------------------------------------------
# 2. Mamba 2D Bottleneck Block (Lõi của bài báo)
# ---------------------------------------------------------
class Mamba2DBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Khởi tạo Mamba block. d_model là số kênh đặc trưng (C)
        self.mamba = Mamba(
            d_model=dim, 
            d_state=16,  # Kích thước state space
            d_conv=4,    # Local convolution width
            expand=2     # Hệ số mở rộng block
        )

    def forward(self, x):
        """
        x: Tensor 2D từ CNN Encoder có shape (Batch, Channels, Height, Width)
        """
        B, C, H, W = x.shape
        
        # Bước 1: Flatten 2D Feature Map thành chuỗi 1D
        # Đổi shape từ (B, C, H, W) -> (B, H*W, C) để đưa vào Mamba
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_flat = self.norm(x_flat)
        
        # Bước 2: Đi qua Mamba (Học Global Context với O(N))
        # NẾU BẠN MUỐN THÊM NOVELTY CỦA BÀI BÁO: Đây là nơi bạn thiết kế 
        # Cyclic Scan (quét vòng tròn) thay vì chỉ cho Mamba chạy mặc định.
        out_flat = self.mamba(x_flat)
        
        # Bước 3: Reshape ngược lại thành 2D map
        out = rearrange(out_flat, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Residual connection
        return out + x

# ---------------------------------------------------------
# 3. MambaRV Architecture (Toàn bộ mạng)
# ---------------------------------------------------------
class MambaRV(nn.Module):
    def __init__(self, in_channels=5, num_classes=20):
        # in_channels=5 (x, y, z, r, e)
        # num_classes=20 (cho SemanticKITTI)
        super().__init__()
        
        # ENCODER (CNN) - Rút trích đặc trưng cực nhanh
        self.enc1 = ConvBlock(in_channels, 32, stride=1)
        self.enc2 = ConvBlock(32, 64, stride=2)   # Downsample 1/2
        self.enc3 = ConvBlock(64, 128, stride=2)  # Downsample 1/4
        self.enc4 = ConvBlock(128, 256, stride=2) # Downsample 1/8
        
        # BOTTLENECK (MAMBA) - Nắm bắt toàn cục ở độ phân giải thấp
        # Giảm thiểu chiều dài chuỗi L = (H/8) * (W/8) để chạy real-time
        self.mamba_bottleneck = nn.Sequential(
            Mamba2DBlock(dim=256),
            Mamba2DBlock(dim=256),
            Mamba2DBlock(dim=256)
        )
        
        # DECODER (CNN) - Upsample và kết hợp Skip Connection
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128) # +128 từ skip connection
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32 + 32, 32)
        
        # Phân loại cuối cùng (Classification Head)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # --- ENCODER ---
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        
        # --- BOTTLENECK (Mamba) ---
        mamba_out = self.mamba_bottleneck(x4)
        
        # --- DECODER ---
        d3 = self.up3(mamba_out)
        d3 = torch.cat([d3, x3], dim=1) # Skip connection
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)
        
        logits = self.final_conv(d1)
        return logits

# --- Test thử Model ---
if __name__ == "__main__":
    # 1. Khai báo thiết bị (Bắt buộc phải là cuda cho Mamba)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")
    
    if device.type == 'cpu':
        raise RuntimeError("Mamba_ssm yêu cầu GPU (CUDA) để chạy. Vui lòng kiểm tra lại môi trường của bạn!")

    # 2. Mô phỏng đầu vào (Batch=2, Channels=5, H=64, W=2048)
    dummy_input = torch.randn(2, 128, 64, 2048) 
    
    # 3. Khởi tạo model
    model = MambaRV(in_channels=128, num_classes=20)
    
    # 4. ĐƯA CẢ MODEL VÀ INPUT LÊN GPU (Điểm mấu chốt để sửa lỗi)
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Để model ở chế độ evaluation khi test
    model.eval()
    
    with torch.no_grad(): # Tắt tính toán gradient để test tốc độ và tiết kiệm VRAM
        output = model(dummy_input)
        
    print(f"Shape đầu vào: {dummy_input.shape}")
    print(f"Shape đầu ra (Logits): {output.shape}") 
    # Kết quả kỳ vọng: torch.Size([2, 20, 64, 2048]) trên thiết bị cuda:0