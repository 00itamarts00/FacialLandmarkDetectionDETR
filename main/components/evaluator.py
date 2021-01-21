import os
import main.globals as g
import torch
import wandb
import logging
torch.cuda.empty_cache()
logger = logging.getLogger(__name__)


class Evaluator(object, params):
    self.pr = params
    self.workset_path = os.path.join(self.ds['dataset_dir'], self.ds['workset_name'])


def evaluate_experiment(workspace_root, worksets_root, workset_name, expname, model, epoch, config):
    batch_size = get_param(config, 'train.batch_size', 5)
    no_cuda = get_param(config, 'experiment.no_cuda', True)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    worksets_path = os.path.join(worksets_root, workset_name)
    trainsetnick = 'trainset_full'

    datasets = ('helen/testset', 'lfpw/testset', 'WFLW/testset', '300W', 'ibug', 'COFW68/COFW_test_color')
    workspace_path = os.path.join(workspace_root, workset_name, trainsetnick, expname)
    results_path = os.path.join(workspace_path, f'results_ep{epoch:04}')
    os.makedirs(results_path, exist_ok=True)
    dfresults = {}
    for item in datasets:
        setnick = item.replace('/', '_')
        dflist = get_data_list(worksets_path, [item], setnick)
        dflist.to_csv(os.path.join(worksets_path, f'{setnick}.csv'))
        # dflist = dflist.sample(n=100)

        testset = CLMDataset(worksets_path, dflist)
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        model = load_model(model, os.path.join(workspace_path, 'nets'), epoch)
        model.to(device)

        results_file = os.path.join(results_path, f'{setnick}.pkl')
        if not os.path.exists(results_file):
            print(f'Evaluating {setnick} testset')
            test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=True, **kwargs)
            dfresults[setnick] = evaluate_model(device, test_loader, model, workspace_path, config)
            dfresults[setnick].to_pickle(results_file)
        else:
            print(f'Loading {setnick} testset results')
            dfresults[setnick] = pd.read_pickle(results_file)

    r300WPub = analyze_results(dfresults, ['helen/testset', 'lfpw/testset', 'ibug'], '300W Public Set')
    r300WPri = analyze_results(dfresults, ['300W'], '300W Private Set')
    rCOFW68 = analyze_results(dfresults, ['COFW68/COFW_test_color'], 'COFW68')
    rWFLW = analyze_results(dfresults, ['WFLW/testset'], 'WFLW')

    print('SET NAME \t\t\t\t AUC08  \t\t NLE ')
    print('----------------------------------------------------')
    print('{} \t\t & {:.03f} \t\t & {:.03f}'.format(r300WPub['setnick'], r300WPub['auc08'], r300WPub['NLE']))
    print('{} \t\t & {:.03f} \t\t & {:.03f}'.format(r300WPri['setnick'], r300WPri['auc08'], r300WPri['NLE']))
    print('{} \t\t\t\t\t & {:.03f} \t\t & {:.03f}'.format(rCOFW68['setnick'], rCOFW68['auc08'], rCOFW68['NLE']))
    print('{} \t\t\t\t\t & {:.03f} \t\t & {:.03f}'.format(rWFLW['setnick'], rWFLW['auc08'], rWFLW['NLE']))

    args_path = os.path.join(workspace_root, workset_name, trainsetnick, expname, 'args')
    flist = get_files_list(args_path, 'json')
    with open(flist[0], 'r') as f:
        config = json.load(f)

    wandb_path = os.path.join(workspace_root, workset_name, trainsetnick, 'wandb')  # expname

    wblog = wandb.init(name=f'{expname}_{epoch:05}', project='landmark-detection', sync_tensorboard=False,
                       dir=str(wandb_path), reinit=True)
    wblog.watch(model, log="all")
    wblog.config.update(config)
    wblog.log({'r300WPub': r300WPub})
    wblog.log({'r300WPri': r300WPri})
    wblog.log({'rCOFW68': rCOFW68})
    wblog.log({'rWFLW': rWFLW})
    wblog.log({'epoch': epoch})
