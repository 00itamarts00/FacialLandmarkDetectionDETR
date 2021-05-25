import logging
import os
import sys

import torch
from prettytable import PrettyTable
from torch.utils import data

import main.globals as g
from main.core.CLMDataset import CLMDataset, get_data_list
from main.core.evaluation_functions import analyze_results
from main.core.functions import inference
from main.core.trainer import LDMTrain
from utils.file_handler import FileHandler

torch.cuda.empty_cache()
logger = logging.getLogger(__name__)
import wandb


class Evaluator(LDMTrain):
    def __init__(self, params):
        super().__init__(params)

    @property
    def hm_amp_factor(self):
        return self.tr["hm_amp_factor"]

    @property
    def log_interval(self):
        return self.ex["log_interval"]

    @property
    def ev(self):
        return self.pr["evaluation"]

    @property
    def tr(self):
        return self.pr["train"]

    def create_test_dataloaders(self, datasets):
        use_cuda = self.tr["cuda"]["use"]
        num_workers = self.tr["cuda"]["num_workers"] if sys.gettrace() is None else 0
        batch_size = self.tr["batch_size"]
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
        res = dict()
        for ds in datasets:
            nickname = ds.replace("/", "_")
            dflist = get_data_list(
                worksets_path=self.workset_path,
                datasets=ds,
                nickname=nickname,
                numpts=68,
            )
            dflist.to_csv(os.path.join(self.paths.workset, f"{nickname}.csv"))
            testset = CLMDataset(
                self.pr, self.paths, dflist, is_train=False, transform=None
            )
            test_loader = data.DataLoader(testset, batch_size=batch_size, **kwargs)
            logging.info(f"Test set: {ds}, Number of test images: {len(testset)}")
            res[ds] = test_loader
        return res

    def evaluate(self):
        res, dataset_eval = dict(), dict()
        self.model.eval()
        self.model.to(self.device)
        test_loaders = self.create_test_dataloaders(self.ev["datasets"])
        for dataset, dataloader in test_loaders.items():
            setnick = dataset.replace("/", "_")
            results_file = os.path.join(self.paths.eval, f"{setnick}.pkl")
            logger.info(f"Evaluating {setnick} testset")
            # test_loader = self.create_dataloaders(datasets=dataset)
            kwargs = {"log_interval": self.log_interval}
            kwargs.update({"hm_amp_factor": self.tr["hm_amp_factor"]})
            kwargs.update({"model_name": self.tr["model"]})
            kwargs.update({"decoder_head": self.ev["prediction_from_decoder_head"]})
            logger.info(
                f'Evaluating model using decoder head: {self.ev["prediction_from_decoder_head"]}'
            )
            dataset_eval[setnick] = self.evaluate_model(
                test_loader=dataloader, model=self.model, **kwargs
            )
            res.update(dataset_eval)
            FileHandler.save_dict_to_pkl(dict_arg=dataset_eval, dict_path=results_file)
        r300WPub = analyze_results(
            res,
            ["helen/testset", "lfpw/testset", "ibug"],
            "300W Public Set",
            output=self.paths.analysis,
            decoder_head=self.ev["prediction_from_decoder_head"],
        )
        r300WPri = analyze_results(
            res,
            ["300W"],
            "300W Private Set",
            output=self.paths.analysis,
            decoder_head=self.ev["prediction_from_decoder_head"],
        )
        rCOFW68 = analyze_results(
            res,
            ["COFW68/COFW_test_color"],
            "COFW68",
            output=self.paths.analysis,
            decoder_head=self.ev["prediction_from_decoder_head"],
        )
        rWFLW = analyze_results(
            res,
            ["WFLW/testset"],
            "WFLW",
            output=self.paths.analysis,
            decoder_head=self.ev["prediction_from_decoder_head"],
        )

        p = PrettyTable()
        p.field_names = ["SET NAME", "AUC08", "FAIL08", "NLE"]
        p.add_row(
            [
                r300WPub["setnick"],
                r300WPub["auc08"],
                r300WPub["fail08"],
                r300WPub["NLE"],
            ]
        )
        p.add_row(
            [
                r300WPri["setnick"],
                r300WPri["auc08"],
                r300WPri["fail08"],
                r300WPri["NLE"],
            ]
        )
        p.add_row(
            [rCOFW68["setnick"], rCOFW68["auc08"], rCOFW68["fail08"], rCOFW68["NLE"]]
        )
        p.add_row([rWFLW["setnick"], rWFLW["auc08"], rWFLW["fail08"], rWFLW["NLE"]])
        logger.info(p)

        for ds in [r300WPub, r300WPri, rCOFW68, rWFLW]:
            if ds is not None:
                p = PrettyTable()
                p.field_names = ["DATASET", "AUC08", "AUC10"]
                for dsk, dsv in ds["ds_logger"].items():
                    p.add_row([dsk, dsv["auc08"], dsv["auc10"]])
                logger.info(p)

        wandb.init(project="detr_landmark_detection", id=g.WANDB_INIT, resume="must")
        wandb.log({"r300WPub": r300WPub})
        wandb.log({"r300WPri": r300WPri})
        wandb.log({"rCOFW68": rCOFW68})
        wandb.log({"rWFLW": rWFLW})

    def evaluate_model(self, test_loader, model, **kwargs):
        epts_batch = dict()
        with torch.no_grad():
            for batch_idx, item in enumerate(test_loader):
                input_, tpts = item["img"].cuda(), item["tpts"].numpy()

                output, preds = inference(model, input_batch=input_, **kwargs)

                item["preds"] = preds.detach().cpu().numpy()
                epts_batch[batch_idx] = item
                percent = f" ({100. * (batch_idx + 1) / len(test_loader):.02f}%)]"
                sys.stdout.write(f"\rTesting batch {batch_idx}\t{percent}")
                sys.stdout.flush()
        sys.stdout.write(f"\n")
        return epts_batch
