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
    parser.add_argument('--dataset', default="cifar100-lt", type=str, help='Dataset to use.',
                        choices=["cifar10", "cifar100", "cifar10-lt", "cifar100-lt"])
    parser.add_argument('--class_names', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1419)
    
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
  
    parser.add_argument('--tro_train', default=2.0,
                        type=float, help='tro for logit adj train')

    return parser


def main():
    """Main script"""

    num_class = args.class_names
    model = torch.nn.DataParallel(resnet32(num_classes=num_class))
    model = model.to(device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)
    
    logit_adjustment,label_freq = utils.compute_adjustment(train_loader)
    print(logit_adjustment)

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[691, 1059, 1290])

    loop = tqdm(range(0, args.epochs), total=args.epochs, leave=False)
    val_loss, val_acc,best_acc = 0, 0, 0
    
    for epoch in loop:

        # train for one epoch
        train_loss, train_acc = train(
            train_loader, model, criterion, optimizer,logit_adjustment)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("train/loss", train_loss, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion,logit_adjustment)
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


def train(train_loader, model, criterion, optimizer,logit_adjustment):
    """ Run one train epoch """

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    model.train()

    for _, (inputs, target) in enumerate(train_loader):
        target = target.to(device)
        input_var = inputs.to(device)
        target_var = target

        output = model(input_var)
        output = output + logit_adjustment
        # loss, output
        loss = criterion(output, target_var)
        acc = utils.accuracy(output.data, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

    return losses.avg, accuracies.avg


def validate(val_loader, model, criterion,logit_adjustment):
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
