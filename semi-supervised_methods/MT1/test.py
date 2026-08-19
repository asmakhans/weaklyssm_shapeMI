import os
import torch
import argparse
from networks.vnet import VNet
from test_util import test_all_case
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='data/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='model_name', help='model_name')
parser.add_argument('--dataset', type=str,  default='knee', help='dataset to use')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--iteration', type=int,  default=6000, help='GPU to use')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = "./model_" + args.dataset + "/" + args.model + "/"
test_save_path = "./model_" + args.dataset + "/testing/"

print(snapshot_path)
print(test_save_path)

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

if args.dataset == 'la':
    num_classes = 2
    patch_size = (112, 112, 80)
    with open(args.root_path + '/../test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + item.replace('\n', '') + ".h5" for item in image_list]

elif args.dataset == 'femur':
    num_classes = 2
    patch_size = (128, 96, 112)
    with open(args.root_path + 'test_femur.txt', 'r') as f:
        image_list = f.readlines()

    image_list = [args.root_path + item.replace('\n', '') + ".h5" for item in image_list]

elif args.dataset == 'namic':
    num_classes = 2
    patch_size = (112, 80, 80)
    image_list = glob.glob(os.path.join(args.root_path, "train", "*.h5"))
    image_list = glob.glob(os.path.join(args.root_path, "test", "*.h5"))

    with open(args.root_path + 'test_la.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + item.replace('\n', '') + ".h5" for item in image_list]

elif args.dataset == 'knee':
    num_classes = 2
    patch_size = (96, 160, 160)
    with open(args.root_path + 'test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [os.path.join(args.root_path, 'test', item.strip()) for item in image_list]


def test_calculate_metric(epoch_num):
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, args.dataset, image_list, num_classes=num_classes, patch_size=patch_size,
                               save_result=True, stride_xy=18, stride_z=4, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric(args.iteration)
    print(metric)
    with open("./model_" + args.dataset + "prediction.txt", "a") as f:
        f.write(args.model + " - " + str(args.iteration) + ": " + ", ".join(str(i) for i in metric) + "\n")
