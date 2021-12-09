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

import utils
from lt_data import train_loader,val_loader
from model import resnet32
import argparse


def get_arguments():

    parser = argparse.ArgumentParser(
        description='PyTorch implementation of the paper: Long-tail Learning via Logit Adjustment')
    parser.add_argument('--dataset', default="cifar10-lt", type=str, help='Dataset to use.',
                        choices=["cifar10", "cifar100", "cifar10-lt", "cifar100-lt"])
    parser.add_argument('--class_names', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1240)
    
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

    return parser


class LPLLoss(nn.Module):
    def __init__(self, num_classes=10, pgd_nums=10, alpha=0.15, epsilon=1.0, sign=True):
        super().__init__()
        self.num_classes = num_classes
        self.pgd_nums = pgd_nums
        self.alpha = alpha
        self.epsilon = epsilon
        self.sign = sign
        self.criterion = nn.CrossEntropyLoss()

    def compute_adv_sign(self, logit, y):
        logit_softmax = F.softmax(logit, dim=-1)
        y_onehot = F.one_hot(y, num_classes=self.num_classes)
        # compute sign(nums*nums)
        sum_class_logit = torch.matmul(
            y_onehot.permute(1, 0) * 1.0, logit_softmax)
        sum_class_num = torch.sum(y_onehot, dim=0)
        # 防止某个类别不存在
        sum_class_num = torch.where(sum_class_num == 0, 100, sum_class_num)
        mean_class_logit = torch.div(
            sum_class_logit, sum_class_num.reshape(-1, 1))

        # compute adv
        grad = mean_class_logit-torch.eye(self.num_classes).cuda()
        grad = torch.div(grad, torch.norm(grad, p=2, dim=0).reshape(-1, 1))

        if self.sign:
            mean_class_p = torch.diag(mean_class_logit)
            # 某个类别不存在时mask掉
            mean_mask = sum_class_num > 0
            mean_class_thr = torch.mean(mean_class_p[mean_mask])
            # max_class_thr = torch.max(mean_class_p[mean_mask])
            sub = mean_class_thr - mean_class_p 
            # sign = sub.sign()
            sign = torch.tensor([-1]*4+[1]*6).cuda()
            # grad = self.alpha * grad * sign.reshape(-1, 1)
            # 抹平补偿epslion 
            # sign = torch.where(sign>0,1,0)
            grad = self.alpha * grad * sign.reshape(-1, 1)
        else:
            grad = self.alpha * grad

        adv_logit = torch.index_select(grad, 0, y)
        
        return adv_logit, grad

    def forward(self, models, x, y, label_freq):

        logit = models(x)
        logit_copy = logit.clone()
        logit_steps = torch.zeros([self.pgd_nums,logit.shape[0],self.num_classes]).cuda()
        logit_news = torch.zeros([logit.shape[0],self.num_classes]).cuda()
        for i in range(self.pgd_nums):
            adv_logit, grad = self.compute_adv_sign(logit, y)
            logit = logit + adv_logit
            logit_steps[i] = logit
            
        for i,freq in enumerate(label_freq):
            logit_news += logit_steps[freq]*torch.where(y==i,1,0).unsqueeze(-1)
            
        loss_adv = self.criterion(logit_news, y)

        return loss_adv, logit_news, logit_copy

def get_step(split:int,classes_num:int,step_size:int,classes_freq:list):

    class_step = []
    for i in range(0,classes_num):
        if i < split:
            step = (classes_freq[i] / classes_freq[0]) * step_size*0.6 - 1
        else:
            step = (classes_freq[-1] / classes_freq[i]) * step_size - 1
        class_step.append(round(step))
    
    return class_step

def get_label_freq(train_loader):
    label_freq = {}
    for _, (_, target) in enumerate(train_loader):
        target = target.cuda()
        for j in target:
            key = str(j.item())
            label_freq[key] = label_freq.get(key, 0)+1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = list(label_freq.values())
    class_step = get_step(split=4,classes_num=10,step_size=10,classes_freq=label_freq_array)
    return class_step

def main():
    """Main script"""

    num_class = args.class_names
    model = torch.nn.DataParallel(resnet32(num_classes=num_class))
    model = model.to(device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)

    criterion_lpl = LPLLoss()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[604,824])

    loop = tqdm(range(0, args.epochs), total=args.epochs, leave=False)
    val_loss, val_acc,best_acc = 0, 0, 0
    
    label_freq = get_label_freq(train_loader)
    print(label_freq)
    

    for epoch in loop:

        # train for one epoch
        train_loss, train_acc = train(
            train_loader, model, criterion_lpl, optimizer,label_freq)
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


    file_name = 'model.th'
    mdel_data = {"state_dict": model.state_dict()}
    torch.save(mdel_data, os.path.join(model_loc, file_name))

    results = utils.class_accuracy(val_loader, model, args)
    results["OA"] = val_acc
    hyper_param = utils.log_hyperparameter(args, args.tro_train)
    pprint(results)
    writer.add_hparams(hparam_dict=hyper_param, metric_dict=results)
    writer.close()


def train(train_loader, model, criterion, optimizer,label_freq):
    """ Run one train epoch """

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    model.train()

    for _, (inputs, target) in enumerate(train_loader):
        target = target.to(device)
        input_var = inputs.to(device)
        target_var = target

        # output = model(input_var)
        # loss, output
        loss, output, _= criterion(model, input_var, target_var,label_freq)
        acc = utils.accuracy(output.data, target)

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

    main()
