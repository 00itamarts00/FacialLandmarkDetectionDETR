HOME = '/home/itamar/thesis/FacialLandmarkDetectionThesis'

import os

# from top_level import TopLevel
os.chdir(HOME)
from main.top_level import TopLevel
from main.components.trainer import LDMTrain
from main.detr.models.segmentation import DETRsegm
from main.detr.misc import nested_tensor_from_tensor_list

TASK_ID = '65956bb26fb84f37ac70cb75ad27fb0d'
tl = TopLevel()
tl.init(task_id=TASK_ID)
tl.setup_workspace()

lmd_train = LDMTrain(params=tl.params, last_epoch=tl.task.get_last_iteration(), logger=tl.logger)

detr_seg = DETRsegm(lmd_train.model.cpu(), freeze_detr=True).cpu()

item = list(lmd_train.valid_loader)[0]
samples = item['img'].cpu()
samples = nested_tensor_from_tensor_list(samples)
res = detr_seg(samples)
