import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from pix_transform.pix_transform_net_diff_models import PixTransformNetBase, PixTransformNetDeeper, PixTransformNetAttention, PixTransformNetMultiScale, PixTransformNetResidual

DEFAULT_PARAMS = {
    'model_type': PixTransformNetBase,  # Specify the model class to use
    'greyscale': False,
    'channels': -1,
    'bicubic_input': False,
    'spatial_features_input': True,
    'weights_regularizer': [0.0001, 0.001, 0.001],  # spatial, color, head
    'loss': 'mse',
    'optim': 'adam',
    'lr': 0.001,
    'batch_size': 32,
    'iteration': 1024 * 32,
    'logstep': 512,
    'patch_size': 256,  # Patch size
    'stride': 256  # Stride
}

def normalize_image(img, mean=None, std=None):
    if mean is None:
        mean = np.mean(img, axis=(1, 2), keepdims=True)
    if std is None:
        std = np.std(img, axis=(1, 2), keepdims=True)
    return (img - mean) / std

def prepare_patches(guide_img, source_img, M, D, device):
    guide_patches = torch.zeros((M * M, guide_img.shape[0], D, D)).to(device)
    source_pixels = torch.zeros((M * M, 1)).to(device)
    for i in range(M):
        for j in range(M):
            guide_patches[j + i * M, :, :, :] = guide_img[:, i * D:(i + 1) * D, j * D:(j + 1) * D]
            source_pixels[j + i * M] = source_img[i:(i + 1), j:(j + 1)]
    return guide_patches, source_pixels

def initialize_network(model_class, channels_in, weights_regularizer, lr, device):
    mynet = model_class(channels_in=channels_in, weights_regularizer=weights_regularizer).train().to(device)
    optimizer = optim.Adam(mynet.params_with_regularizer, lr=lr)
    return mynet, optimizer

def PixTransform_diff_models(source_img, guide_img, params=DEFAULT_PARAMS, target_img=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(guide_img.shape) < 3:
        guide_img = np.expand_dims(guide_img, 0)

    if params["channels"] > 0:
        guide_img = guide_img[:params["channels"], :, :]

    if params['greyscale']:
        guide_img = np.mean(guide_img, axis=0, keepdims=True)

    n_channels, hr_height, hr_width = guide_img.shape
    source_img = source_img.squeeze()
    lr_height, lr_width = source_img.shape

    assert hr_height % lr_height == 0

    D = hr_height // lr_height
    M = lr_height

    guide_img = normalize_image(guide_img)
    source_img_mean, source_img_std = np.mean(source_img), np.std(source_img)
    source_img = normalize_image(source_img, source_img_mean, source_img_std)
    if target_img is not None:
        target_img = normalize_image(target_img, source_img_mean, source_img_std)

    if params['spatial_features_input']:
        x = np.linspace(-0.5, 0.5, hr_width)
        x_grid, y_grid = np.meshgrid(x, x, indexing='ij')
        x_grid = np.expand_dims(x_grid, axis=0)
        y_grid = np.expand_dims(y_grid, axis=0)
        guide_img = np.concatenate([guide_img, x_grid, y_grid], axis=0)

    guide_img = torch.from_numpy(guide_img).float().to(device)
    source_img = torch.from_numpy(source_img).float().to(device)
    if target_img is not None:
        target_img = torch.from_numpy(target_img).float().to(device)

    patch_size = params['patch_size']
    stride = params['stride']
    patches = []
    predicted_patches = []
    attention_maps = []  # Store attention maps if available

    for i in range(0, hr_height - patch_size + 1, stride):
        for j in range(0, hr_width - patch_size + 1, stride):
            patch_guide = guide_img[:, i:i + patch_size, j:j + patch_size]
            patch_source = source_img[i // D:(i + patch_size) // D, j // D:(j + patch_size) // D]
            patches.append((patch_guide, patch_source))

    # Initialize the chosen network architecture
    mynet, optimizer = initialize_network(params['model_type'], guide_img.shape[0], params['weights_regularizer'], params['lr'], device)
    loss_fn = torch.nn.MSELoss() if params['loss'] == 'mse' else torch.nn.L1Loss()

    for idx, (patch_guide, patch_source) in enumerate(patches):
        M = patch_source.shape[0]
        guide_patches, source_pixels = prepare_patches(patch_guide, patch_source, M, D, device)
        train_data = torch.utils.data.TensorDataset(guide_patches, source_pixels)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)

        epochs = params["batch_size"] * params["iteration"] // (M * M)
        with tqdm(range(epochs), leave=True) as tnr:
            tnr.set_description(f"Patch {idx+1}/{len(patches)}")
            if target_img is not None:
                tnr.set_postfix(MSE=-1., consistency=-1.)
            else:
                tnr.set_postfix(consistency=-1.)

            for epoch in tnr:
                for x, y in train_loader:
                    optimizer.zero_grad()
                    y_pred, _ = mynet(x)
                    y_mean_pred = torch.mean(y_pred, dim=[2, 3])
                    loss = loss_fn(y_mean_pred, y)
                    loss.backward()
                    optimizer.step()

                if epoch % params['logstep'] == 0:
                    with torch.no_grad():
                        mynet.eval()
                        predicted_patch, _ = mynet(patch_guide.unsqueeze(0))
                        predicted_patch = predicted_patch.squeeze()
                        if target_img is not None:
                            mse_pred = F.mse_loss(source_img_std * predicted_patch, source_img_std * target_img)
                        consistency = loss_fn(source_img_std * F.avg_pool2d(predicted_patch.unsqueeze(0), D),
                                              source_img_std * patch_source.unsqueeze(0))
                        if target_img is not None:
                            tnr.set_postfix(MSE=mse_pred.item(), consistency=consistency.item())
                        else:
                            tnr.set_postfix(consistency=consistency.item())
                        mynet.train()

        # Save attention maps if available
        if isinstance(mynet, PixTransformNetAttention) and hasattr(mynet, 'attention_weights'):
            attention_maps.append(mynet.attention_weights.cpu().detach().numpy())

        mynet.eval()
        predicted_patch = mynet(patch_guide.unsqueeze(0))
        predicted_patch = source_img_mean + source_img_std * predicted_patch
        predicted_patches.append(predicted_patch.cpu().detach().numpy().squeeze())
        torch.cuda.empty_cache()

    predicted_target_img = np.zeros((hr_height, hr_width))
    overlap_count = np.zeros((hr_height, hr_width))

    idx = 0
    for i in range(0, hr_height - patch_size + 1, stride):
        for j in range(0, hr_width - patch_size + 1, stride):
            predicted_target_img[i:i + patch_size, j:j + patch_size] += predicted_patches[idx]
            overlap_count[i:i + patch_size, j:j + patch_size] += 1
            idx += 1

    # Avoid division by zero
    overlap_count[overlap_count == 0] = 1
    predicted_target_img /= overlap_count

    if attention_maps:
        return (predicted_target_img, attention_maps)

    else:
        return predicted_target_img