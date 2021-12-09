import os
from pprint import pprint
from tqdm import tqdm
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import utils
from lt_data import train_loader,val_loader
from model import resnet32
import argparse


def get_arguments():

    parser = argparse.ArgumentParser(
        description='PyTorch implementation of the paper: Long-tail Learning via Logit Adjustment')
    parser.add_argument('--dataset', default="cifar10-lt", type=str, help='Dataset to use.',
                        choices=["cifar10", "cifar100", "cifar10-lt", "cifar100-lt"])
    parser.add_argument('--class_names', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1420)
    
    parser.add_argument('--data_home', default="data", type=str,
                        help='Directory where data files are stored.')
    parser.add_argument('--num_workers', default=2, type=int, metavar='N',
                        help='number of workers at dataloader')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4,
                        type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--log_val', help='compute val acc',
                        type=int, default=10)
  
    parser.add_argument('--tro_train', default=1.0,
                        type=float, help='tro for logit adj train')
    
    parser.add_argument('--sign', default=True, type=bool, help='PGD split.')
    parser.add_argument('--thr', default="LA", type=str, help='PGD thr.')
    parser.add_argument('--num_classes', default=100,
                        type=int, help='num_classes')
    parser.add_argument('--pgd_nums', default=40, type=int, help='PGD nums')
    parser.add_argument('--alpha', default=0.01, type=float, help='PGD  alpha.')
    parser.add_argument('--split', default=40, type=int, help='PGD split.')
    

    return parser


def get_step(split: int, classes_num: int, step_size: int, classes_freq: list):
    class_step = []
    for i in range(0, classes_num):
        if i < split:
            step = (classes_freq[i] / classes_freq[0]) * step_size*0.2 - 1
        else:
            step = (classes_freq[-1] / classes_freq[i]) * step_size - 1
        class_step.append(round(step))
    return class_step

class LPLLoss(nn.Module):
    def __init__(self, sign=True, thr="LA", num_classes=100, pgd_nums=50, alpha=0.1, split=27):
        super().__init__()
        self.num_classes = num_classes
        self.pgd_nums = pgd_nums
        self.alpha = alpha
        self.split = split
        self.sign = sign
        self.thr = thr
        self.criterion = nn.CrossEntropyLoss()
        self.CE = nn.CrossEntropyLoss(reduction = 'none')

    def compute_adv_sign(self, logit, y):
        logit_softmax = F.softmax(logit, dim=-1)
        y_onehot = F.one_hot(y, num_classes=self.num_classes)
        # compute sign(nums*nums)  .to(torch.float64)
        sum_class_logit = torch.matmul(y_onehot.permute(1, 0)*1.0, logit_softmax)
        sum_class_num = torch.sum(y_onehot, dim=0)
        # 防止某个类别不存在
        sum_class_num = torch.where(sum_class_num == 0, 100, sum_class_num)
        mean_class_logit = torch.div(
            sum_class_logit, sum_class_num.reshape(-1, 1))

        # compute adv
        grad = mean_class_logit-torch.eye(self.num_classes).cuda()
        grad = torch.div(grad, torch.norm(grad, p=2, dim=0).reshape(-1, 1))

        if self.sign:
            if self.thr == "MEAN":
                mean_class_p = torch.diag(mean_class_logit)
                # 某个类别不存在时mask掉
                mean_mask = sum_class_num > 0
                mean_class_thr = torch.mean(mean_class_p[mean_mask])
                sub = mean_class_thr - mean_class_p
                sign = sub.sign()
            elif self.thr == "LA":
                sign = torch.tensor([-1]*self.split+[1] *
                                    (self.num_classes-self.split)).cuda()
            grad = self.alpha * grad * sign.reshape(-1, 1)

        else:
            grad = self.alpha * grad

        adv_logit = torch.index_select(grad, 0, y)
        return adv_logit, grad

    def forward(self, models, x, y, args):

        logit = models(x)
        init_l = self.CE(logit, y)
        logit_copy = logit.clone()
        logit_steps = torch.zeros(
            [self.pgd_nums, logit.shape[0], self.num_classes]).cuda()
        logit_news = torch.zeros([logit.shape[0], self.num_classes]).cuda()
        for i in range(self.pgd_nums):
            adv_logit, grad = self.compute_adv_sign(logit, y)
            logit = logit + adv_logit
            logit_steps[i] = logit

        for i, freq in enumerate(args.label_freq):
            logit_news += logit_steps[freq] * \
                torch.where(y == i, 1, 0).unsqueeze(-1)
        finl_l = self.CE(logit_news, y)
        loss_adv = self.criterion(logit_news, y)

        return loss_adv, logit_news, logit_copy, init_l, finl_l
    

def main(ord_d):
    """Main script"""

    num_class = args.class_names
    model = torch.nn.DataParallel(resnet32(num_classes=num_class))
    model = model.to(device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)

    criterion_lpl = LPLLoss(sign=args.sign, thr=args.thr, num_classes=args.num_classes,
                            pgd_nums=args.num_classes, alpha=args.alpha, split=args.split)
    
    _, label_freqs = utils.compute_adjustment(train_loader)
    
    # label_freqs = sorted(label_freqs.items(), key=lambda e: e[1], reverse=True)
    label_freq_list = [x[1] for x in label_freqs]
    label_freq = get_step(split=args.split, classes_num=100,
                          step_size=args.pgd_nums, classes_freq=label_freq_list)
    args.label_freq = label_freq
    print(label_freq)    
    
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[80, 120, 160])

    loop = tqdm(range(0, args.epochs), total=args.epochs, leave=False)
    val_loss, val_acc,best_acc = 0, 0, 0
    
    

    for epoch in loop:

        # train for one epoch
        train_loss, train_acc = train(
            train_loader, model, criterion_lpl, optimizer, ord_d, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("train/loss", train_loss, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion)
        if val_acc>best_acc:
            best_acc = val_acc
            writer.add_scalar("val/acc", val_acc, epoch)
            writer.add_scalar("best/acc", best_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)

        loop.set_description(f"Epoch [{epoch}/{args.epochs}")
        loop.set_postfix(train_loss=f"{train_loss:.2f}", val_loss=f"{val_loss:.2f}",
                         train_acc=f"{train_acc:.2f}",
                         val_acc=f"{val_acc:.2f}",
                         best_acc=f"{best_acc:.2f}")
        if (epoch + 1) % 100 == 0:
            np.save('res100_110.npy', ord_d)

    np.save('res.npy', ord_d)
    file_name = 'model.th'
    mdel_data = {"state_dict": model.state_dict()}
    torch.save(mdel_data, os.path.join(model_loc, file_name))

    results = utils.class_accuracy(val_loader, model, args)
    results["OA"] = val_acc
    hyper_param = utils.log_hyperparameter(args, args.tro_train)
    pprint(results)
    writer.add_hparams(hparam_dict=hyper_param, metric_dict=results)
    writer.close()


def train(train_loader, model, criterion, optimizer, ord_d, epoch):
    """ Run one train epoch """

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    model.train()
    labels = []
    init_loss = []
    finl_loss = []
    
    for _, (inputs, target) in enumerate(train_loader):
        target = target.to(device)
        labels.append(target)
        input_var = inputs.to(device)
        target_var = target

        # output = model(input_var)
        # loss, output
        loss, output, _, init_l, finl_l= criterion(model, input_var, target_var,args)
        acc = utils.accuracy(output.data, target)
        init_loss.append(init_l)
        finl_loss.append(finl_l)
        # loss = criterion(output, target_var)

        loss_r = 0
        for parameter in model.parameters():
            loss_r += torch.sum(parameter ** 2)
        loss = loss + args.weight_decay * loss_r

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
    labels = torch.cat(labels).cpu().detach().numpy()
    init_loss = torch.cat(init_loss).cpu().detach().numpy()
    finl_loss = torch.cat(finl_loss).cpu().detach().numpy()
    d = {'labels':labels,
    'init_loss':init_loss,
    'finl_loss':finl_loss}
    ord_d[epoch] = d

    return losses.avg, accuracies.avg


def validate(val_loader, model, criterion):
    """ Run evaluation """

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for _, (inputs, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = inputs.to(device)
            target_var = target.to(device)

            output = model(input_var)
            loss = criterion(output, target_var)

            acc = utils.accuracy(output.data, target)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

    return losses.avg, accuracies.avg


if __name__ == '__main__':
    
    parser = get_arguments()
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    exp_loc, model_loc = utils.log_folders(args)
    writer = SummaryWriter(log_dir=exp_loc)
    ord_d = {}
    main(ord_d)
