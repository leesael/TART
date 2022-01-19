import argparse
import json
import os
import time
from collections import defaultdict

import graphviz

import numpy as np
import torch
from torch import optim

import tart
import data
import baselines
from utils import str2bool


def to_model(args, num_features, num_classes):
    if args.model == 'MLP':
        return baselines.MLP(num_features, num_classes,
                             num_layers=args.layers)
    elif args.model == 'TART-dense':
        return tart.make_network(num_features, num_classes,
                                 depth=args.depth,
                                 style=args.style,
                                 activation=args.activation,
                                 num_nodes=args.nodes)
    elif args.model == 'TART':
        return tart.make_tree(num_features, num_classes,
                              depth=args.depth,
                              decision_units=args.decision_units,
                              style=args.style,
                              activation=args.activation,
                              window=args.window,
                              leaf_layers=args.layers,
                              leaf_units=args.units,
                              temperature=args.temperature,
                              decision_shared=args.decision_shared)
    else:
        raise ValueError(args.model)


def to_device(gpu):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update(model, device, loader, optimizer):
    model.train()
    for x, y in loader:
        x = x.to(device).view(x.size(0), -1)
        y = y.to(device)
        optimizer.zero_grad()
        loss = model.loss(x, y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, device, loader):
    model.eval()
    loss_sum, acc_sum, num_data = 0, 0, 0
    for x, y in loader:
        x = x.to(device).view(x.size(0), -1)
        y = y.to(device)
        loss_sum += model.loss(x, y).item() * x.size(0)
        acc_sum += torch.eq(torch.argmax(model.forward(x), dim=1), y).sum().item()
        num_data += x.size(0)
    return loss_sum / num_data, acc_sum / num_data


@torch.no_grad()
def visualize(model, data_x, dir_out, device, num_samples=100):
    node_idx = defaultdict(lambda: len(node_idx) + 1)

    def to_node_idx(l, i):
        return str(node_idx[(l, i)])

    def to_width(value):
        return str(0.4 + 2.6 * value)

    def draw_nodes(graph_, p_list_):
        graph_.node(to_node_idx(0, 0), penwidth=to_width(1))
        for l, vec_p in enumerate(p_list_[1:]):
            for j in range(vec_p.shape[0]):
                graph_.node(to_node_idx(l + 1, j), penwidth=to_width(vec_p[j]))

    def draw_edges(graph_, p_list_, t_list_):
        for l, (mat_t, vec_p) in enumerate(zip(t_list_, p_list_[:-1])):
            mat_t = mat_t * vec_p
            for i in range(mat_t.shape[1]):
                for j in range(mat_t.shape[0]):
                    if mat_t[j, i] > 0:
                        src_idx = to_node_idx(l, i)
                        dst_idx = to_node_idx(l + 1, j)
                        graph_.edge(src_idx, dst_idx, penwidth=to_width(mat_t[j, i]))

    model.eval()
    data_x = torch.from_numpy(data_x[:num_samples, :]).to(device)
    for sample in range(data_x.size(0)):
        t_list, p_list = model.decision_path(data_x[sample].unsqueeze(0))
        t_list = [t.squeeze(0).cpu().numpy() for t in t_list]
        p_list = [p.squeeze(0).cpu().numpy() for p in p_list]

        graph = graphviz.Digraph(graph_attr=dict(margin=str(0.1),
                                                 nodesep=str(0.1),
                                                 ranksep=str(0.3)),
                                 node_attr=dict(width=str(0.4),
                                                height=str(0.4),
                                                fixedsize=str(True),
                                                style='filled'),
                                 edge_attr=dict(arrowhead='none'))

        draw_nodes(graph, p_list)
        draw_edges(graph, p_list, t_list)

        graph.format = 'pdf'
        graph.filename = str(sample)
        graph.directory = dir_out
        graph.render(view=False)
        os.remove(os.path.join(dir_out, str(sample)))


def parse_args():
    parser = argparse.ArgumentParser()

    # Experimental settings.
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data', type=str, default='breast-tissue')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data-path', type=str, default='../data')
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)

    # Major model settings.
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--units', type=int, default=100)
    parser.add_argument('--decision-units', type=int, default=0)
    parser.add_argument('--decision-shared', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--nodes', type=int, default=16)
    parser.add_argument('--style', type=str, default=None)

    # Model settings for trees.
    parser.add_argument('--shape', type=str, default='custom')
    parser.add_argument('--window', type=int, default=2)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--root-window', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--activation', type=str, default='softmax')

    # Training settings.
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--decay', type=float, default=0.)

    return parser.parse_args()


def check_args_validity(args):
    if args.model == 'TART':
        assert args.depth is not None
        assert args.style is not None


def to_out_path(args):
    if args.model == 'MLP':
        return os.path.join(args.out, f'{args.model}-{args.layers}')
    elif args.model == 'TART':
        model_name = f'{args.model}-D{args.depth}'
        if args.window > 2:
            model_name = f'{model_name}-W{args.window}'
        model_name = f'{model_name}-L{args.layers}-{args.style}'
        return os.path.join(args.out, model_name)
    else:
        raise ValueError(args.model)


def main():
    start_time = time.time()
    args = parse_args()
    check_args_validity(args)
    out_path = to_out_path(args)
    dataset = args.data
    device = to_device(args.gpu)

    fold = args.fold
    np.random.seed(fold)
    torch.manual_seed(fold)

    trn_x, trn_y, test_x, test_y = data.read_data(args.data_path, dataset, fold)
    num_features = trn_x.shape[1]
    num_classes = trn_y.max() + 1
    trn_loader = data.to_loader(trn_x, trn_y, args.batch_size, shuffle=True)
    test_loader = data.to_loader(test_x, test_y, args.batch_size)

    model = to_model(args, num_features, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    log_out = os.path.join(out_path, f'logs/{dataset}/{fold}.txt')
    os.makedirs(os.path.dirname(log_out), exist_ok=True)
    with open(log_out, 'w') as f:
        f.write('epoch\ttrn_loss\ttrn_acc\tis_best\n')

    try:
        model = model.to(device)
        best_loss, best_epoch = np.inf, 0
        for epoch in range(args.epochs + 1):
            if epoch > 0:
                update(model, device, trn_loader, optimizer)
            trn_loss, trn_acc = evaluate(model, device, trn_loader)

            if trn_loss < best_loss:
                best_loss = trn_loss
                best_epoch = epoch

            with open(log_out, 'a') as f:
                f.write(f'{epoch:5d}\t{trn_loss:.4f}\t{trn_acc:.4f}\t')
                if epoch == best_epoch:
                    f.write('\tBEST')
                f.write('\n')

        _, trn_acc = evaluate(model, device, trn_loader)
        _, test_acc = evaluate(model, device, test_loader)

        if args.visualize:
            dir_out = os.path.join(out_path, 'graphviz', args.data, str(args.fold))
            visualize(model, test_x, dir_out, device, num_samples=100)

        if args.save:
            model_out = os.path.join(out_path, 'models-{}/{}.pth'.format(fold, dataset))
            os.makedirs(os.path.dirname(model_out), exist_ok=True)
            torch.save(model.state_dict(), model_out)

    except RuntimeError as e:
        if not str(e).startswith('CUDA out of memory.'):
            raise e
        trn_acc = -np.inf
        test_acc = -np.inf

    out = {arg: getattr(args, arg) for arg in vars(args)}
    out['out_path'] = out_path
    out['result'] = dict(
        trn_acc=trn_acc,
        test_acc=test_acc,
        params=count_parameters(model),
        time=time.time() - start_time)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
