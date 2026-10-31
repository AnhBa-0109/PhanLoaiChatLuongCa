import torch
import torch.nn as nn

QUANTILES = (0.10, 0.50, 0.90)

def masked_color_features(x, n_bins=16):
    B = x.shape[0]
    lab = x[:, :3].flatten(2)            # (B, 3, HW)
    mask = (x[:, 3:4].flatten(2) > 0.5).float()  # (B, 1, HW)
    n_pix = mask.sum(dim=2).clamp_min(1.0)       # (B, 1)

    # --- Histogram có trọng số mask ---
    idx = (lab.clamp(0, 1) * n_bins).long().clamp(0, n_bins - 1)   #
    hist = torch.zeros(B, 3, n_bins, device=x.device, dtype=x.dtype)
    hist.scatter_add_(2, idx, mask.expand_as(lab).to(x.dtype))
    hist = hist / n_pix.unsqueeze(-1)    
    hist = hist.flatten(1)              

    # --- mean, std ---
    mean = (lab * mask).sum(2) / n_pix                              
    var = (((lab - mean.unsqueeze(-1)) ** 2) * mask).sum(2) / n_pix
    std = var.clamp_min(0).sqrt()                                   

    # --- phân vị: tính riêng từng ảnh (mỗi ảnh có số pixel thịt khác nhau) ---
    q = torch.tensor(QUANTILES, device=x.device, dtype=x.dtype)
    qs = []
    for i in range(B):
        sel = lab[i][:, mask[i, 0] > 0]                             
        if sel.shape[1] == 0:
            qs.append(torch.zeros(3, len(QUANTILES), device=x.device, dtype=x.dtype))
        else:
            qs.append(torch.quantile(sel, q, dim=1).T)               
    qs = torch.stack(qs).flatten(1)                                  

    return torch.cat([hist, mean, std, qs], dim=1)                  

class ColorBranch(nn.Module):
    def __init__(self, n_bins=16, out_dim=64):
        super().__init__()
        self.n_bins = n_bins
        in_dim = 3 * n_bins + 3 * (2 + len(QUANTILES))
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(in_dim),   
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim
    def forward(self, x):
        with torch.no_grad():        
            feats = masked_color_features(x, self.n_bins)
        return self.mlp(feats)