import os
import sys
import logging
import traceback
import sys

#sys.path.append('lama') 
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))

import hydra
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
from saicinpainting.evaluation.utils import move_to_device
import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

# Import the FeatureAdaptationLayer and AdaptationLoss from previous examples
 
LOGGER = logging.getLogger(__name__)

import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torchvision.models as models
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import logging
import traceback
import sys

#sys.path.append('lama') 
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))

import hydra
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
from saicinpainting.evaluation.utils import move_to_device
import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

# Import the FeatureAdaptationLayer and AdaptationLoss from previous examples
 
LOGGER = logging.getLogger(__name__)

import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torchvision.models as models
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from tqdm import tqdm
class PostProcessor(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64):
        super(PostProcessor, self).__init__()
        
        # Convolutional layers for feature extraction and enhancement
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        
        out += residual
        return torch.clamp(out, 0, 1)
        
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = vgg16(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:23])
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.features(x)

class AdaptiveLossWeight(nn.Module):
    def __init__(self, num_losses):
        super(AdaptiveLossWeight, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_losses))

    def forward(self, losses):
        return torch.sum(self.weights * losses + torch.log(1 + torch.exp(-self.weights)))

def train_post_processor(pretrained_model, post_processor, train_dataset, val_dataset, config):
    device = torch.device(config.device)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
    
    optimizer = torch.optim.Adam(post_processor.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    feature_extractor = FeatureExtractor().to(device)
    adaptive_weight = AdaptiveLossWeight(num_losses=3).to(device)
    optimizer.add_param_group({'params': adaptive_weight.parameters(), 'lr': 1e-3})

    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    
    best_val_loss = float('inf')
    
    for epoch in range(7):
        post_processor.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{7}"):
            batch = move_to_device(batch, device)
            
            with torch.no_grad():
                pretrained_output = pretrained_model(batch)['predicted_image']
            
            enhanced_output = post_processor(pretrained_output)
            
            # Multiple loss components
            mse_loss = mse_criterion(enhanced_output, batch['image'])
            l1_loss = l1_criterion(enhanced_output, batch['image'])
            
            # Feature matching loss
            pretrained_features = feature_extractor(pretrained_output)
            enhanced_features = feature_extractor(enhanced_output)
            feature_loss = F.mse_loss(enhanced_features, pretrained_features)
            
            # Prevent degradation
            degradation_loss = F.relu(mse_loss - mse_criterion(pretrained_output, batch['image']))
            
            # Combine losses with adaptive weights
            losses = torch.stack([mse_loss, l1_loss, feature_loss])
            loss = adaptive_weight(losses) + degradation_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        val_loss = validate(post_processor, pretrained_model, val_loader, mse_criterion, device)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        

        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #torch.save(post_processor.state_dict(), os.path.join(config.outdir, 'best_post_processor.pth'))
        for param in adaptive_weight.parameters():
            print(param.data)
    return post_processor

def validate(post_processor, pretrained_model, val_loader, criterion, device):
    post_processor.eval()
    val_loss = 0
    pretrained_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = move_to_device(batch, device)
            pretrained_output = pretrained_model(batch)['predicted_image']
            enhanced_output = post_processor(pretrained_output)
            
            val_loss += criterion(enhanced_output, batch['image']).item()
            pretrained_loss += criterion(pretrained_output, batch['image']).item()
    
    val_loss /= len(val_loader)
    pretrained_loss /= len(val_loader)
    
    print(f"Validation - PostProcessor Loss: {val_loss:.4f}, Pretrained Loss: {pretrained_loss:.4f}")
    return val_loss

@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        # Load the pre-trained model
        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        pretrained_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        pretrained_model.freeze()
        pretrained_model.to(device)

        # Create the post-processor
        post_processor = PostProcessor().to(device)

        # Prepare the datasets
        train_dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        new_indir = predict_config.indir.replace("Train", "Test")
        val_dataset = make_default_val_dataset(new_indir, **predict_config.dataset)
        test_dataset = val_dataset#make_default_val_dataset(predict_config.test_indir, **predict_config.dataset)

        # Update the dataset with the new path
        #new_dataset = make_default_val_dataset(new_indir, **predict_config.dataset)
        
        # Train the post-processor
        post_processor = train_post_processor(pretrained_model, post_processor, train_dataset, val_dataset, predict_config)
        # post_processor.load_state_dict(torch.load(os.path.join(predict_config.outdir, 'best_post_processor.pth')))
        ## Inference and save results
        #post_processor = train_post_processor(pretrained_model, post_processor, train_dataset, val_dataset, predict_config)
        
        post_processor.eval()
        out_ext = predict_config.get('out_ext', '.png')
        
        for img_i in tqdm(range(len(test_dataset))):
            mask_fname = test_dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir, 
                os.path.splitext(mask_fname[len(new_indir):])[0] + out_ext
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([test_dataset[img_i]])
            
            with torch.no_grad():
                batch = move_to_device(batch, device)
                pretrained_output = pretrained_model(batch)
                #pretrained_output = pretrained_model(batch)['predicted_image']
                #refined_output = pretrained_output
                refined_output = post_processor(pretrained_output['predicted_image'])
                ###refined_output = pretrained_output['predicted_image']
                
                cur_res = refined_output[0].permute(1, 2, 0).detach().cpu().numpy()

                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)
            
     
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)

if __name__ == '__main__':
    main()
    