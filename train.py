import os
import csv
import nni
import json
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.trainer import Trainer, TrainConfig
from models.models import AntiDesigner
from data.dataset import EquiAACDataset, ITAWrapper, AAComplex
from evaluation.rmsd import kabsch
from evaluation.pred_ddg import pred_ddg
from models.evaluator import average_test, set_cdr, rabd_test

from data.utils import set_seed, check_dir, print_log, save_code, valid_check

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def constant_beta(iter_num, beta):
    return beta

class Experiment:
    def __init__(self, args):
        self.args = args

    def _get_data(self, train_path, valid_path):

        train_set = EquiAACDataset(train_path)
        train_set.mode = self.args.mode
        valid_set = EquiAACDataset(valid_path)
        valid_set.mode = self.args.mode

        n_channel = valid_set[0]['X'].shape[1]
        model = AntiDesigner(
                self.args.embed_size, #32
                self.args.hidden_size, #256
                n_channel,  # 4
                n_layers=self.args.n_layers,  # 4
                dropout=args.dropout,
                cdr_type=self.args.cdr_type, 
                args=self.args
        )

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, num_workers=4, shuffle=True, collate_fn=EquiAACDataset.collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=self.args.batch_size, num_workers=4, shuffle=False, collate_fn=EquiAACDataset.collate_fn)
        
        return train_loader, valid_loader, model


    def generate(self, data_dir, save_dir):

        model = torch.load(os.path.join(save_dir, 'checkpoint/best.ckpt')).to(device)
        test_set = EquiAACDataset(os.path.join(data_dir, 'test.json'))
        test_set.mode = self.args.mode
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, num_workers=4, shuffle=False, collate_fn=EquiAACDataset.collate_fn)
        
        model.eval()
        if self.args.optimization == 0 or self.args.optimization == 1:
            report_res = average_test(self.args, model, test_set, test_loader, save_dir, device)
        else:
            report_res = rabd_test(self.args, model, test_set, test_loader, save_dir, device)

        if self.args.output_pdb == True:
            out_dir = os.path.join(save_dir, 'pre_results', 'original')
            check_dir(out_dir)
            for cplx in tqdm(test_set.data):
                pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
                cplx.to_pdb(pdb_path)

        return report_res

    def k_fold_train_eval(self, timestamp):

        res_dict = {'PPL': [], 'RMSD': [], 'TMscore': [], 'AAR': []}

        for k in range(10):
            print_log('CDR {}, Fold {}'.format(self.args.cdr_type, k))

            if self.args.split == -1:
                data_dir = os.path.join('./summaries/cdrh{}'.format(self.args.cdr_type), 'fold_{}'.format(k))
            else:
                data_dir = os.path.join('./summaries/data/split_{}/cdrh{}'.format(self.args.split, self.args.cdr_type), 'fold_{}'.format(k))
            save_dir = os.path.join('./results/cdrh{}'.format(self.args.cdr_type), 'fold_{}'.format(k), timestamp)
            check_dir(save_dir)
            save_code(save_dir)
            
            train_loader, valid_loader, model = self._get_data(os.path.join(data_dir, 'train.json'), os.path.join(data_dir, 'valid.json'))

            trainer = Trainer(model, train_loader, valid_loader, save_dir, args)
            trainer.train()

            report_res = self.generate(data_dir, save_dir)
            for key in res_dict.keys():
                res_dict[key].append(report_res[key])
                print_log('CDR {}, Fold {} | '.format(self.args.cdr_type, k) + f'{key}: {report_res[key]}')
            nni.report_intermediate_result(report_res['AAR'])

        write_buffer = {}
        for key in res_dict.keys():
            vals = res_dict[key]
            val_mean, val_std = np.mean(vals), np.std(vals)
            write_buffer[key] = res_dict[key]
            write_buffer[key+'_mean'] = val_mean
            write_buffer[key+'_std'] = val_std
            print_log('CDR {} Fold-Mean-STD | '.format(self.args.cdr_type) + f'{key}: mean {val_mean}, std {val_std}')

        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(write_buffer, f, indent=2)

        return write_buffer


    def ita_train(self, timestamp, path=None):
        data_dir = './summaries'
        save_dir = os.path.join('./results/skempi', timestamp)
        check_dir(save_dir)
        save_code(save_dir)

        train_loader, valid_loader, model = self._get_data(os.path.join(data_dir, 'train.json'), os.path.join(data_dir, 'valid.json'))
        config = TrainConfig(args, save_dir, args.lr, args.max_epoch, grad_clip=args.grad_clip, early_stop=args.early_stop, anneal_base=args.anneal_base)
        trainer = Trainer(model, train_loader, valid_loader, save_dir, args)
        
        if path is not None:
            model = torch.load(path).to(device)
        else:
            trainer.train()
            model = torch.load(os.path.join(save_dir, 'checkpoint/best.ckpt')).to(device)

        valid_set = EquiAACDataset(os.path.join(data_dir, 'valid.json'))
        valid_set.mode = self.args.mode
        
        model.eval()
        report_res = average_test(self.args, model, valid_set, valid_loader, save_dir, device)

        dataset = EquiAACDataset(os.path.join(data_dir, 'skempi_all.json'))
        dataset.mode = self.args.mode
        itawrapper = ITAWrapper(dataset, self.args.n_samples)

        # writing original structrues
        out_dir = os.path.join(save_dir, 'ita_results/original')
        check_dir(out_dir)
        print_log(f'Writing original structures to {out_dir}')

        origin_cplx_paths = []
        origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]
        
        for cplx in tqdm(origin_cplx):
            pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
            cplx.to_pdb(pdb_path)
            origin_cplx_paths.append(os.path.abspath(pdb_path))


        log_file = open(os.path.join(save_dir, "train_log.txt"), 'a+')
        best_round, best_score = -1, 1e10

        for r in range(self.args.n_iter):
            res_dir = os.path.join(save_dir, 'ita_results/iterative')
            check_dir(res_dir)

            model.eval()
            scores = []
            for i in tqdm(range(len(dataset))):
                origin_input = dataset[i]
                inputs = [origin_input for _ in range(self.args.n_tries)]
                candidates, results = [], []
                with torch.no_grad():
                    batch = dataset.collate_fn(inputs)
                    ppls, seqs, xs, true_xs, aligned = model.infer(batch, device, greedy=False)
                    results.extend([(ppls[i], seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])

                recorded, candidate_pool = {}, []
                for n, (ppl, seq, x, true_x, aligned) in enumerate(results):
                    if seq in recorded:
                        continue
                    recorded[seq] = True
                    if ppl > 10:
                        continue
                    if not valid_check(seq):
                        continue
                    if not aligned:
                        ca_aligned, rotation, t = kabsch(x[:, 1, :], true_x[:, 1, :])
                        x = np.dot(x - np.mean(x, axis=0), rotation) + t
                    candidate_pool.append((ppl, seq, x, n))

                sorted_cand_idx = sorted([j for j in range(len(candidate_pool))], key=lambda j: candidate_pool[j][0])
                for j in sorted_cand_idx:
                    ppl, seq, x, n = candidate_pool[j]
                    new_cplx = set_cdr(origin_cplx[i], seq, x, cdr='H' + str(model.cdr_type))
                    pdb_path = os.path.join(res_dir, new_cplx.get_id() + f'_{n}.pdb')
                    new_cplx.to_pdb(pdb_path)
                    new_cplx = AAComplex(
                        new_cplx.pdb_id, new_cplx.peptides,
                        new_cplx.heavy_chain, new_cplx.light_chain,
                        new_cplx.antigen_chains)
                    try:
                        score = pred_ddg(origin_cplx_paths[i], os.path.abspath(pdb_path))
                    except Exception as e:
                        score = 0
                    if score < 0:
                        candidates.append((new_cplx, score))
                        scores.append(score)
                    if len(candidates) >= self.args.n_samples:
                        break
                while len(candidates) < self.args.n_samples:
                    candidates.append((origin_cplx[i], 0))
                    scores.append(0)
                itawrapper.update_candidates(i, candidates)

            itawrapper.finish_update()

            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_round, best_score = r - 1, mean_score
                check_dir(os.path.join(save_dir, 'checkpoint'))
                # torch.save(model, os.path.join(save_dir, f'checkpoint/iter_{r}.ckpt'))
                torch.save(model, os.path.join(save_dir, 'checkpoint/iter_best.ckpt'))

            print_log(f'{time.strftime("%Y-%m-%d %H-%M-%S")} | model from iteration {r}, ddg mean {mean_score}, std {np.std(scores)}, history best {best_score} at round {best_round}')
            log_file.write(f'{time.strftime("%Y-%m-%d %H-%M-%S")} | model from iteration {r}, ddg mean {mean_score}, std {np.std(scores)}, history best {best_score} at round {best_round}\n')
            log_file.flush()

            # train
            model.train()
            train_loader = DataLoader(itawrapper, batch_size=self.args.batch_size, num_workers=4, shuffle=True, collate_fn=itawrapper.collate_fn)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

            batch_idx = 0
            for e in range(self.args.ita_epoch):
                for batch in train_loader:
                    batch = trainer.to_device(batch, device)
                    loss, _, _ = model(batch['X'], batch['S'], batch['L'], batch['offsets'], opt=True) 
                    loss /= self.args.update_freq
                    loss.backward()

                    batch_idx += 1
                    if batch_idx % self.args.update_freq == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()

            if batch_idx % self.args.update_freq != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        return report_res, best_score

    def ita_eval(self, timestamp, path=None):
        data_dir = './summaries'
        save_dir = os.path.join('./results/skempi', timestamp)
        
        if path is not None:
            model = torch.load(path).to(device)
        else:
            model = torch.load(os.path.join(save_dir, f'checkpoint/iter_best.ckpt')).to(device)

        dataset = EquiAACDataset(os.path.join(data_dir, 'skempi_all.json'))
        dataset.mode = self.args.mode

        # writing original structrues
        out_dir = os.path.join(save_dir, 'ita_results/original')
        check_dir(out_dir)
        print_log(f'Writing original structures to {out_dir}')

        origin_cplx_paths = []
        origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]
        
        for cplx in tqdm(origin_cplx):
            pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
            cplx.to_pdb(pdb_path)
            origin_cplx_paths.append(os.path.abspath(pdb_path))

        log_file = open(os.path.join(save_dir, "train_log.txt"), 'a+')
        res_dir = os.path.join(save_dir, 'ita_results/optimized')
        check_dir(res_dir)

        scores = []
        cdr_list = []

        for i in tqdm(range(len(dataset))):
            origin_input = dataset[i]
            inputs = [origin_input for _ in range(100)]
            cur_scores, results, cur_cdr_list = [], [], []
            with torch.no_grad():
                batch = dataset.collate_fn(inputs)
                ppls, seqs, xs, true_xs, aligned = model.infer(batch, device, greedy=False)
                results.extend([(seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])

            for n, (seq, x, true_x, aligned) in enumerate(results):
                if not aligned:
                    ca_aligned, rotation, t = kabsch(x[:, 1, :], true_x[:, 1, :])
                    x = np.dot(x - np.mean(x, axis=0), rotation) + t
                new_cplx = set_cdr(origin_cplx[i], seq, x, cdr='H' + str(model.cdr_type))
                pdb_path = os.path.join(res_dir, new_cplx.get_id() + f'_{n}.pdb')
                new_cplx.to_pdb(pdb_path)
                try:
                    score = pred_ddg(origin_cplx_paths[i], os.path.abspath(pdb_path))
                except Exception as e:
                    score = 0
                cur_scores.append(score)
                cur_cdr_list.append((seq, x, true_x))

            mean_score = np.mean(cur_scores)
            best_score_idx = min([k for k in range(len(cur_scores))], key=lambda k: cur_scores[k])
            scores.append(cur_scores[best_score_idx])
            cdr_list.append((cur_cdr_list[best_score_idx]))

            print_log(f'{time.strftime("%Y-%m-%d %H-%M-%S")} | pdb {origin_cplx[i].get_id()}, mean ddg {mean_score}, best ddg {cur_scores[best_score_idx]}, sample {best_score_idx}')
            log_file.write(f'{time.strftime("%Y-%m-%d %H-%M-%S")} | pdb {origin_cplx[i].get_id()}, mean ddg {mean_score}, best ddg {cur_scores[best_score_idx]}, sample {best_score_idx}\n')
            log_file.flush()

        with open(os.path.join(save_dir, "ita_results/ita_results.pickle"), "wb") as f:
            pickle.dump(cdr_list, f)

        return scores

    def rabd_train(self, timestamp):
        data_dir = './summaries/cdrh{}'.format(self.args.cdr_type)
        save_dir = os.path.join('./results/rabd/cdrh{}'.format(self.args.cdr_type), timestamp)
        check_dir(save_dir)
        save_code(save_dir)
        
        train_loader, valid_loader, model = self._get_data(os.path.join(data_dir, 'train.json'), os.path.join(data_dir, 'valid.json'))
        trainer = Trainer(model, train_loader, valid_loader, save_dir, args)
        trainer.train()

        write_buffer = self.generate(data_dir, save_dir)
        return write_buffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    #RAAD
    parser.add_argument('--cdr_type', type=str, default='3', help='type of cdr')
    parser.add_argument('--mode', type=str, default='111', help='H/L/Antigen, 1 for include, 0 for exclude')
    parser.add_argument('--node_feats_mode', type=str, default='1111', help='choose what node features to use')
    parser.add_argument('--edge_feats_mode', type=str, default='1111', help='choose what edge features to use')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=20, help='max training epoch')

    parser.add_argument('--embed_size', type=int, default=32, help='embed size of amino acids')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--n_layers', type=int, default=4, help='number of layers')
    parser.add_argument('--alpha', type=float, default=0.8, help='scale mse loss of coordinates')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')

    parser.add_argument('--seed', type=int, default=24, help='Seed to use in training')
    parser.add_argument('--early_stop', type=bool, default=True, help='Whether to use early stop')
    parser.add_argument('--grad_clip', type=float, default=0.5, help='clip gradients with too big norm')
    parser.add_argument('--anneal_base', type=float, default=0.9, help='Exponential lr decay, 1 for not decay')
    parser.add_argument('--output_pdb', type=bool, default=False, help='Whether to use save pdb files')
    parser.add_argument('--interface_only', type=int, default=1, help='antigen interface_only')
    parser.add_argument('--split', type=int, default=-1, help='Which split used to train')

    parser.add_argument('--optimization', type=int, default=0, help='used for antibody optimization')
    parser.add_argument('--ita_epoch', type=int, default=1, help='number of epochs per iteration')
    parser.add_argument('--n_iter', type=int, default=20, help='Number of iterations to run')
    parser.add_argument('--n_tries', type=int, default=50, help='Number of tries each iteration')
    parser.add_argument('--n_samples', type=int, default=8, help='Number of samples each iteration')
    parser.add_argument('--update_freq', type=int, default=4, help='Model update frequency')
    parser.add_argument('--beta', type=float, default=0.5, help='loss weights for specificity constraints')
    parser.add_argument('--ab_mode', type=int, default=0, help='Model update frequency')
    parser.add_argument('--topk', type=int, default=1, help='Number of candidates for evaluation')
    parser.add_argument('--rabd_sample', type=int, default=1, help='Number of samples for RAbD test')
    parser.add_argument('--path', type=str, default=None, help='Number of samples for RAbD test')

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())
    args = argparse.Namespace(**param)
    timestamp = time.strftime("%Y-%m-%d %H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)

    if args.optimization == 0:
        set_seed(args.seed)
        exp = Experiment(args)
        write_buffer = exp.k_fold_train_eval(timestamp)

        nni.report_final_result(write_buffer['AAR_mean'])

        outFile = open('./PerformMetrics_SAbDab.csv','a+', newline='')
        writer = csv.writer(outFile, dialect='excel')

        results = [timestamp]
        for v, k in param.items():
            results.append(k)

        results.append(str(write_buffer['PPL_mean']))
        results.append(str(write_buffer['RMSD_mean']))
        results.append(str(write_buffer['TMscore_mean']))
        results.append(str(write_buffer['AAR_mean']))
        results.append(str(write_buffer['PPL_std']))
        results.append(str(write_buffer['RMSD_std']))
        results.append(str(write_buffer['TMscore_std']))
        results.append(str(write_buffer['AAR_std']))
        results.append(str(write_buffer['PPL']))
        results.append(str(write_buffer['RMSD']))
        results.append(str(write_buffer['TMscore']))
        results.append(str(write_buffer['AAR']))
        writer.writerow(results)
    elif args.optimization == 1:
        args.cdr_type = '3'
        set_seed(args.seed)
        exp = Experiment(args)
        report_res, train_score = exp.ita_train(timestamp,args.path)

        if train_score < -3.0:
            scores = exp.ita_eval(timestamp)
        else:
            scores = [-5.0]

        nni.report_final_result(-np.mean(scores))

        outFile = open('./PerformMetrics_opt.csv','a+', newline='')
        writer = csv.writer(outFile, dialect='excel')

        results = [timestamp]
        for v, k in param.items():
            results.append(k)
        for key in report_res.keys():
            results.append(report_res[key])
        results.append(train_score)
        results.append(str(np.mean(scores)))
        results.append(str(np.std(scores)))
        results.append(str(scores))
        writer.writerow(results)
    elif args.optimization == 2:
        args.cdr_type = '3'
        set_seed(args.seed)
        exp = Experiment(args)
        write_buffer = exp.rabd_train(timestamp)

        nni.report_final_result(write_buffer['AAR'][0])

        outFile = open('./PerformMetrics_rabd.csv','a+', newline='')
        writer = csv.writer(outFile, dialect='excel')

        results = [timestamp]
        for v, k in param.items():
            results.append(k)

        results.append(str(write_buffer['PPL'][0]))
        results.append(str(write_buffer['RMSD'][0]))
        results.append(str(write_buffer['TMscore'][0]))
        results.append(str(write_buffer['AAR'][0]))
        results.append(str(write_buffer['PPL'][1]))
        results.append(str(write_buffer['RMSD'][1]))
        results.append(str(write_buffer['TMscore'][1]))
        results.append(str(write_buffer['AAR'][1]))
        writer.writerow(results)

