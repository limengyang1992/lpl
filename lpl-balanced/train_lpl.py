import argparse
import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')


###########################################################################
parser.add_argument('--epochs', default=240, type=int,
                    help='number of total epochs to run')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--lplloss', default=True, type=bool)
parser.add_argument('--pgd_nums', default=40, type=int)
parser.add_argument('--alpha', default=0.01, type=float)
##########################################################################
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True,
                    type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=16, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=8, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.3, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-16-8', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')

parser.set_defaults(augment=True)


best_prec1 = 0
args = parser.parse_args()


log_name = args.dataset \
    + "_" + str(args.name) \
    + "_" + str(args.epochs) \
    + "_" + str(args.lplloss) \
    + "_" + str(args.pgd_nums) \
    + "_" + str(args.alpha) \
    + "_" + time.strftime("%Y%m%d%H%M%S", time.localtime()) \
    + ".txt"

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(log_name)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_step(split: int, classes_num: int, pgd_nums: int, classes_freq: list):
    step_size = pgd_nums*0.1
    class_step = []
    for i in range(0, classes_num):
        if i < split:
            step = (classes_freq[i] / classes_freq[0]) * step_size - 1
        else:
            step = (classes_freq[i] / classes_freq[-1]) * step_size - 1
        class_step.append(round(step))
    class_step = [0 if x < 0 else x for x in class_step]
    class_step = [pgd_nums+x for x in class_step]
    return class_step


class LPLLoss(nn.Module):
    def __init__(self, num_classes, pgd_nums, alpha):
        super().__init__()
        self.num_classes = num_classes
        self.pgd_nums = pgd_nums
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def compute_steps(self, logit, y):
        with torch.no_grad():
            _, sub = self.compute_adv_sign(logit, y)
            sort_indexs = torch.sort(sub).indices.detach().cpu().numpy()
            sort_values = torch.sort(sub).values.detach().cpu().numpy()
            split = torch.sum(torch.where(sub < 0, 1, 0)
                              ).detach().cpu().numpy()
            class_step = get_step(split, classes_num=self.num_classes,
                                  pgd_nums=self.pgd_nums, classes_freq=sort_values)
            new_class_step = len(sort_indexs)*[0]
            for i, ind in enumerate(sort_indexs):
                new_class_step[ind] = class_step[i]

        return new_class_step

    def compute_adv_sign(self, logit, y):
        with torch.no_grad():
            logit_softmax = F.softmax(logit, dim=-1)
            y_onehot = F.one_hot(y, num_classes=self.num_classes)
            # compute sign(nums*nums)
            sum_class_logit = torch.matmul(
                y_onehot.permute(1, 0)*1.0, logit_softmax)
            sum_class_num = torch.sum(y_onehot, dim=0)
            # 防止某个类别不存在
            sum_class_num = torch.where(sum_class_num == 0, 100, sum_class_num)
            mean_class_logit = torch.div(
                sum_class_logit, sum_class_num.reshape(-1, 1))
            # compute adv
            grad = mean_class_logit-torch.eye(self.num_classes).cuda()
            grad = torch.div(grad, torch.norm(grad, p=2, dim=0).reshape(-1, 1))

            mean_class_p = torch.diag(mean_class_logit)
            # 某个类别不存在时mask掉
            mean_mask = sum_class_num > 0
            mean_class_thr = torch.mean(mean_class_p[mean_mask])
            sub = mean_class_thr - mean_class_p
            sign = sub.sign()
            grad = self.alpha * grad * sign.reshape(-1, 1)
            adv_logit = torch.index_select(grad, 0, y)
        return adv_logit, sub

    def compute_eta(self, logit, y):

        with torch.no_grad():
            logit_clone = logit
            class_steps = self.compute_steps(logit, y)
            # compute adv logit
            round_pgdnums = round(self.pgd_nums*1.1)
            logit_steps = torch.zeros(
                [round_pgdnums, logit.shape[0], self.num_classes]).cuda()
            logit_news = torch.zeros([logit.shape[0], self.num_classes]).cuda()
            for i in range(round_pgdnums):
                adv_logit, _ = self.compute_adv_sign(logit, y)
                logit = logit + adv_logit
                logit_steps[i] = logit
                logit = logit + adv_logit

            for i, freq in enumerate(class_steps):
                logit_news += logit_steps[freq] * \
                    torch.where(y == i, 1, 0).unsqueeze(-1)

            eta = logit_news - logit_clone
        return eta

    def forward(self, models, x, y):

        logit = models(x)
        eta = self.compute_eta(logit, y)
        logit_news = logit + eta
        loss_adv = self.criterion(logit_news, y)
        return loss_adv, logit_news


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes,
                          out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def main():
    global args, best_prec1
    # if args.tensorboard:
    #     configure("runs/%s" % (args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
                                                transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](
            '../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                       args.widen_factor, dropRate=args.droprate)

    # get the number of model parameters
    logger.info('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion_cat = lplloss(num_classes=args.num_classes, pgd_nums=args.pgd_nums, alpha=args.alpha)
    criterion_cat = LPLLoss(num_classes=args.num_classes,
                            pgd_nums=args.pgd_nums, alpha=args.alpha)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader)*args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.lplloss:
            train(train_loader, model, criterion_cat,
                  optimizer, scheduler, epoch, args.lplloss)
        else:
            train(train_loader, model, criterion,
                  optimizer, scheduler, epoch, args.lplloss)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    logger.info('Best accuracy {best_prec1:.3f}'.format(best_prec1=best_prec1))


def train(train_loader, model, criterion, optimizer, scheduler, epoch, lplloss):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        if lplloss:
            loss, output = criterion(model, input, target)
        else:
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            loss=losses, top1=top1))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('train_loss', losses.avg, epoch)
    #     log_value('train_acc', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1))

    logger.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('val_loss', losses.avg, epoch)
    #     log_value('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' %
                        (args.name) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
