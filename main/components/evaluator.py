import logging
import os

import torch
from prettytable import PrettyTable
from torch.utils import data

import main.globals as g
from main.components.CLMDataset import CLMDataset, get_data_list
from main.components.trainer import LDMTrain
from main.refactor.evaluation import evaluate_model, analyze_results
from utils.file_handler import FileHandler

torch.cuda.empty_cache()
logger = logging.getLogger(__name__)
import wandb


class Evaluator(LDMTrain):
    def __init__(self, params):
        super().__init__(params)

    @property
    def hm_amp_factor(self):
        return self.tr['hm_amp_factor']

    @property
    def log_interval(self):
        return self.ex['log_interval']

    @property
    def ds(self):
        return self.pr['dataset']

    @property
    def ev(self):
        return self.pr['evaluation']

    @property
    def tr(self):
        return self.pr['train']

    @property
    def ex(self):
        return self.pr['experiment']

    def create_workspace(self):
        workspace_path = self.pr['workspace_path']
        structure = {'workspace': workspace_path,
                     'checkpoint': os.path.join(workspace_path, 'checkpoint'),
                     'args': os.path.join(workspace_path, 'args'),
                     'logs': os.path.join(workspace_path, 'logs'),
                     'stats': os.path.join(workspace_path, 'stats'),
                     'eval': os.path.join(workspace_path, 'evaluation'),
                     'wandb': os.path.join(workspace_path, 'wandb'),
                     'workset': self.workset_path
                     }
        paths = FileHandler.dict_to_nested_namedtuple(structure)
        [os.makedirs(i, exist_ok=True) for i in paths]
        return paths

    def backend_operations(self):
        cuda = self.tr['cuda']
        torch.manual_seed(self.tr['torch_seed'])
        use_cuda = cuda['use'] and torch.cuda.is_available()
        device = torch.device(cuda['device_type'] if use_cuda else 'cpu')
        torch.backends.benchmark = self.tr['backend']['use_torch']
        return device

    def create_test_data_loader(self, dataset):
        use_cuda = self.tr['cuda']['use']
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        batch_size = self.tr['batch_size']
        setnick = dataset.replace('/', '_')
        dflist = get_data_list(self.paths.workset, [dataset], setnick)
        dflist.to_csv(os.path.join(self.paths.workset, f'{setnick}.csv'))
        testset = CLMDataset(self.pr, self.paths, dflist)
        test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=True, **kwargs)
        return test_loader

    def evaluate(self):
        res, dataset_eval = dict(), dict()
        batch_size = self.tr['batch_size']
        self.model.eval()
        self.model.to(self.device)
        for dataset in self.ev['datasets']:
            setnick = dataset.replace('/', '_')
            results_file = os.path.join(self.paths.eval, f'{setnick}.pkl')
            if not os.path.exists(results_file):
                logger.info(f'Evaluating {setnick} testset')
                test_loader = self.create_test_data_loader(dataset=dataset)
                kwargs = {'log_interval': self.log_interval}
                dataset_eval[setnick] = evaluate_model(device=self.device,
                                                       test_loader=test_loader,
                                                       model=self.model,
                                                       **kwargs)
                res.update(dataset_eval)
                FileHandler.save_dict_to_pkl(dict_arg=dataset_eval, dict_path=results_file)
            else:
                logger.info(f'Loading {setnick} testset results')
                dataset_eval = FileHandler.load_pkl(results_file)
                res.update(dataset_eval)

        r300WPub = analyze_results(res, ['helen/testset', 'lfpw/testset', 'ibug', 'AFW'], '300W Public Set')
        r300WPri = analyze_results(res, ['300W'], '300W Private Set')
        rCOFW68 = analyze_results(res, ['COFW68/COFW_test_color'], 'COFW68')
        rWFLW = analyze_results(res, ['WFLW/testset'], 'WFLW')
        #
        p = PrettyTable()
        p.field_names = ["SET NAME", "AUC08", "FAIL08", "NLE"]
        p.add_row([r300WPub['setnick'], r300WPub['auc08'], r300WPub['fail08'], r300WPub['NLE']])
        p.add_row([r300WPri['setnick'], r300WPri['auc08'], r300WPri['fail08'], r300WPri['NLE']])
        p.add_row([rCOFW68['setnick'], rCOFW68['auc08'], rCOFW68['fail08'], rCOFW68['NLE']])
        p.add_row([rWFLW['setnick'], rWFLW['auc08'], rWFLW['fail08'], rWFLW['NLE']])
        logger.info(p)

        wandb.init(project="detr_landmark_detection",
                   id=g.WANDB_INIT,
                   resume='must')
        wandb.log({'r300WPub': r300WPub})
        wandb.log({'r300WPri': r300WPri})
        wandb.log({'rCOFW68': rCOFW68})
        wandb.log({'rWFLW': rWFLW})
