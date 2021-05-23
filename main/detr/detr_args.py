num_classes = 67
lr = 1e-04
lr_backbone = 1e-05
weight_decay = 0.0001
lr_drop = 50
clip_max_norm = 0.05
frozen_weights = None
backbone = 'resnet50'
backbone_pretrained = True
return_interm_layers = True
dilation = False
position_embedding = 'learned'
enc_layers = 2
dec_layers = 4
dim_feedforward = 2048
hidden_dim = 512
dropout = 0.1
nheads = 2
num_queries = 68
pre_norm = True
masks = False
dataset_file = 'WS02'
output_dir = '/home/itamar/thesis/outputs/detr'
device = 'cuda'
seed = 42
# losses
multi_dec_loss = True
multi_enc_loss = False
heatmap_regression_via_backbone = False
last_dec_coord_loss = True
