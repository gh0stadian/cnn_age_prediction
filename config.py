import wandb
import torch
import datetime

from torchvision import transforms
from models.models import get_base_model, get_double_conv
from torchsummary import summary

timestamp = datetime.datetime.now().strftime("%d%m_%I%M%S")

wandb = wandb.init(
    project="test-project",
    entity="jbdb",
    config={
        'epoch': 100,
        'batch_size': 32,
        'lr': 0.01,
        'es_patience': 5,
        'scheduler_patience': 3,
        'scheduler_factor': 0.2,
        'dataset_path': "oversampled.csv",
        'img_root_dir': "imdb_crop/",
        'checkpoint': datetime.datetime.now().strftime("%d%m_%I%M%S"),
        'model_config': {
            'model_name': "double_conv",
            'classification_layers': [],
            'conv_layers': [8, 16, 32, 64, 128],
            'num_classes': 1
        }
    }
)

model = get_double_conv(conv_layers=wandb.config['model_config']['conv_layers'],
                        conv_kernels=None,
                        fc_layers=wandb.config['model_config']['classification_layers'],
                        num_classes=wandb.config['model_config']['num_classes']
                        )
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])

criterion = torch.nn.L1Loss()

img_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]
                                                          )]
                                    )

summary(model, (3, 224, 224), device="cpu")
