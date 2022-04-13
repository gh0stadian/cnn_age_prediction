import wandb
import torch
import datetime
import pandas as pd

from torchvision import transforms
from models.models import get_base_model, get_double_conv, get_resnet, get_adaptive_model
from models.custom_models.resnet_model import ResBlock, ResBatchNormBlock
from torchsummary import summary
from optimizations import MSELoss_age_multiplied

timestamp = datetime.datetime.now().strftime("%d%m_%I%M%S")

wandb = wandb.init(
    project="test-project",
    entity="jbdb",
    config={
        'epoch': 100,
        'batch_size': 128,
        'lr': 0.01,
        'es_patience': 5,
        'scheduler_step_size': 1,
        'scheduler_gamma': 0.5,
        'train_dataset': "datasets/train.csv",
        'test_dataset': "datasets/test.csv",
        'test_log_dataset': "datasets/log.csv",
        'img_root_dir': "imdb_crop/",
        'checkpoint': datetime.datetime.now().strftime("%d%m_%I%M%S"),
        'model_config': {
            'model_name': "resnet",
            'classification_layers': [],
            'conv_layers': [64, 128, 256, 512],
            'num_classes': 1
        }
    }
)

model = get_resnet(ResBlock,
                   wandb.config['model_config']['conv_layers'],
                   num_classes=wandb.config['model_config']['num_classes']
                   )

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])

train_df = pd.read_csv(wandb.config['train_dataset'], low_memory=False)
test_df = pd.read_csv(wandb.config['test_dataset'], low_memory=False)
test_log_df = pd.read_csv(wandb.config['test_log_dataset'], low_memory=False)
criterion = torch.nn.MSELoss()

# criterion = MSELoss_age_multiplied()
# Calibrate loss to reflect class imbalance
# criterion.calibrate(df)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=wandb.config['scheduler_step_size'],
                                            gamma=wandb.config['scheduler_gamma']
                                            )

img_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]
                                                          )]
                                    )

summary(model, (3, 224, 224), device="cpu")
