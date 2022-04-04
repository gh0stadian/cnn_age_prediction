import wandb
import torch
import datetime

from torchvision import transforms
from models.models import get_baseline_model, get_pretrained_model_resnet18, get_pretrained_model_resnet50
from torchsummary import summary

timestamp = datetime.datetime.now().strftime("%d%m_%I%M%S")

wandb = wandb.init(
    project="test-project",
    entity="jbdb",
    config={
        'epoch': 100,
        'batch_size': 64,
        'lr': 0.001,
        'es_patience': 5,
        'dataset_path': "cured_imdb.csv",
        'img_root_dir': "imdb_crop/",
        'checkpoint': datetime.datetime.now().strftime("%d%m_%I%M%S")
    }
)

# zero_h flag would ignore linear layer sizes
classification_params = {"lin_1_size": 64,
                         'lin_2_size': 32,
                         'num_classes': 1,
                         'zero_h': True
                         }

conv_params = {'conv_1_size': 32,
               'conv_2_size': 64,
               'conv_3_size': 128,
               'conv_4_size': 256}

model = get_baseline_model(**conv_params, **classification_params)

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])

criterion = torch.nn.L1Loss()

img_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]
                                                          )]
                                    )

summary(model, (3, 224, 224), device="cpu")
