from guided_diffusion.guided_diffusion_1d import Unet1D, GaussianDiffusion1D
import torch
from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, seq_length, num_classes, t_dim=1) -> None:
        super().__init__()
        self.linear_t = nn.Linear(t_dim, num_classes)
        self.linear_img = nn.Linear(seq_length, num_classes)
    def forward(self, x, t):
        """
        Args:
            x (_type_): [B, N]
            t (_type_): [B,]

        Returns:
                logits [B, num_classes]
        """
        B = x.shape[0]
        t = t.view(B, 1)
        logits = self.linear_t(t.float()) + self.linear_img(x)
        return logits
    
def classifier_cond_fn(x, t, classifier, y, classifier_scale=1):
    """
    return the graident of the classifier outputing y wrt x.
    formally expressed as d_log(classifier(x, t)) / dx
    """
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y]
        grad = torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale
        return grad

model = Unet1D(dim=128, channels=1, self_condition=False)

# input dummy tensors
batch_size = 1
channels = 1    # same as model channels
seq_length = 128  # use diffusion model latent space seq length
dummy_input = torch.randn(batch_size, channels, seq_length)

dummy_time = torch.randint(0, 1000, (batch_size,)).float()  # timesteps in [0, num_timesteps)

# check pass works
output = model(dummy_input, dummy_time)
print("UNet output shape:", output.shape)

diffusion_model = GaussianDiffusion1D(
    model=model,
    seq_length=seq_length,
    timesteps=1000
)

loss = diffusion_model(dummy_input) 
print("Loss:", loss.item())

dummy_classifier = Classifier(seq_length=seq_length, num_classes=2)
dummy_labels = torch.randint(0, 1, (batch_size,))
grad = classifier_cond_fn(dummy_input, dummy_time, dummy_classifier, dummy_labels, classifier_scale=1)
print(grad.shape)
# sampled = diffusion_model.sample(batch_size=4)
# print("Sampled output shape:", sampled.shape)
