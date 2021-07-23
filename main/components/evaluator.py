import logging
import os
import sys
from functools import partial

import torch
from torch.utils import data
from tqdm import tqdm

from main.components.CLMDataset import CLMDataset, get_data_list
from main.components.trainer import LDMTrain
from main.core.evaluation_functions import analyze_results
from main.core.functions import inference

torch.cuda.empty_cache()
logger = logging.getLogger(__name__)


class Evaluator(LDMTrain):
    def __init__(self, params, logger_cml):
        super().__init__(params, logger=logger_cml)

    @property
    def ev(self):
        return self.pr['evaluation']

    @property
    def tr(self):
        return self.pr['train']

    def create_test_data_loader(self, dataset):
        use_cuda = self.tr.cuda.use
        num_workers = self.tr.cuda.num_workers if sys.gettrace() is None else 0
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
        batch_size = self.tr.batch_size
        setnick = dataset.replace('/', '_')
        dflist = get_data_list(self.paths.workset, [dataset], setnick)
        dflist.to_csv(os.path.join(self.paths.workset, f'{setnick}.csv'))
        testset = CLMDataset(self.pr, self.paths, dflist)
        test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
        return test_loader

    def evaluate(self):
        res, dataset_eval = dict(), dict()
        self.model.eval()
        self.model.to(self.device)
        for dataset in self.ev['datasets']:
            setnick = dataset.replace('/', '_')
            test_loader = self.create_test_data_loader(dataset=dataset)
            kwargs = {'log_interval': self.pr.log_interval}
            if self.tr.heatmaps.train_with_heatmaps:
                kwargs.update({'hm_amp_factor': self.tr.heatmaps.hm_amp_factor})
            kwargs.update({'model_name': self.tr.model})
            kwargs.update({'decoder_head': self.ev.prediction_from_decoder_head})
            self.logger_cml.report_text(
                f'Evaluating {setnick} testset using decoder head: {self.ev.prediction_from_decoder_head}',
                logging.INFO, print_console=True)

            dataset_eval[setnick] = self.evaluate_model(test_loader=test_loader, **kwargs)
            res.update(dataset_eval)
            # FileHandler.save_dict_to_pkl(dict_arg=dataset_eval, dict_path=results_file)

        analyze_results_to_analysis_path = partial(analyze_results,
                                                   output=self.paths.analysis,
                                                   decoder_head=self.ev.prediction_from_decoder_head,
                                                   logger=self.logger_cml)

        iterable = dict()
        iterable.update({"300W Public Set": ["helen/testset", "lfpw/testset", "ibug"]})
        iterable.update({"300W Private Set": ["300W"]})
        iterable.update({"COFW68": ["COFW68/COFW_test_color"]})
        iterable.update({"WFLW": ["WFLW/testset"]})

        for key, val in iterable.items():
            analyze_results_to_analysis_path(datastets_inst=res, datasets=val, eval_name=key)

    def evaluate_model(self, test_loader, **kwargs):
        epts_batch = dict()
        with torch.no_grad():
            for batch_idx, item in enumerate(tqdm(test_loader, position=0, leave=True)):
                input_, tpts = item['img'].cuda(), item['tpts'].cuda()

                output, preds = inference(model=self.model, input_batch=input_, **kwargs)

                item['preds'] = [i.cpu().detach() for i in preds]
                epts_batch[batch_idx] = item
        return epts_batch
