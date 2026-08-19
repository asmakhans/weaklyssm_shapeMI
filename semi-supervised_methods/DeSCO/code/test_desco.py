import os
import argparse
import torch
from networks.vnet import VNet
from networks.crn import crnVNet
from test_desco_util import test_all_case,test_all_case_crn

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,  default='laOurs', help='model_name')
parser.add_argument('--gpu', type=str,  default='4', help='GPU to use')
parser.add_argument('--dataset',type=str,default="la",help='dataset to use')
parser.add_argument('--max_iteration', type=int,  default=6000, help='GPU to use')
parser.add_argument('--iteration_step', type=int,  default=2000, help='GPU to use')
parser.add_argument('--split', type=str,  default='test', help='testlist to use')
parser.add_argument('--min_iteration', type=int,  default=6000, help='GPU to use')
parser.add_argument('--data_root', type=str, required=True, help='Directory containing test HDF5 files')
parser.add_argument('--split_file', type=str, required=True, help='Text file listing test case IDs')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2
data_root = os.path.abspath(os.path.expanduser(FLAGS.data_root))
split_file = os.path.abspath(os.path.expanduser(FLAGS.split_file))
with open(split_file, 'r') as f:
    cases = [line.strip().split(',')[0] for line in f if line.strip()]
image_list = [os.path.join(data_root, case + '.h5') for case in cases]


def test_calculate_metric(epoch_num):
    if 'CRN' in FLAGS.model:
        net = crnVNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
        save_mode_path1 = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
        print(save_mode_path1)
        print("init weight from {}".format(save_mode_path1))
        net.eval()
        if FLAGS.dataset == 'la':
            net.load_state_dict(torch.load(save_mode_path1), strict=True)
            avg_metric,dice_list = test_all_case_crn(net, image_list, num_classes=num_classes,
                                       patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                       save_result=True, test_save_path=test_save_path, dataset=FLAGS.dataset)
        elif FLAGS.dataset == 'kits_kidney':
            net.load_state_dict(torch.load(save_mode_path1), strict=True)
            avg_metric,dice_list = test_all_case_crn(net, image_list, num_classes=num_classes,
                                       patch_size=(128, 128, 64), stride_xy=18, stride_z=4,
                                       save_result=True, test_save_path=test_save_path, dataset=FLAGS.dataset)
        elif FLAGS.dataset == 'lits_liver':
            net.load_state_dict(torch.load(save_mode_path1), strict=True)
            avg_metric,dice_list = test_all_case_crn(net, image_list, num_classes=num_classes,
                                       patch_size=(176, 176, 64), stride_xy=18, stride_z=4,
                                       save_result=True, test_save_path=test_save_path, dataset=FLAGS.dataset)
        return avg_metric,dice_list
    else:
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
        save_mode_path1 = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
        print(save_mode_path1)
        print("init weight from {}".format(save_mode_path1))
        net.eval()
        if FLAGS.dataset == 'la':
            net.load_state_dict(torch.load(save_mode_path1), strict=True)
            avg_metric, dice_list = test_all_case(net, image_list, num_classes=num_classes,
                                                  patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                                  save_result=True, test_save_path=test_save_path,
                                                  dataset=FLAGS.dataset)
        elif FLAGS.dataset == 'kits_kidney':
            net.load_state_dict(torch.load(save_mode_path1), strict=True)
            avg_metric, dice_list = test_all_case(net, image_list, num_classes=num_classes,
                                                  patch_size=(128, 128, 64), stride_xy=18, stride_z=4,
                                                  save_result=True, test_save_path=test_save_path,
                                                  dataset=FLAGS.dataset)

        elif FLAGS.dataset == 'lits_liver':
            net.load_state_dict(torch.load(save_mode_path1), strict=True)
            avg_metric, dice_list = test_all_case(net, image_list, num_classes=num_classes,
                                                  patch_size=(176, 176, 64), stride_xy=18, stride_z=4,
                                                  save_result=True, test_save_path=test_save_path,
                                                  dataset=FLAGS.dataset)


        return avg_metric,dice_list


if __name__ == '__main__':
    path=os.path.join(snapshot_path, 'test.txt')
    for i in range(FLAGS.min_iteration,FLAGS.max_iteration+1,FLAGS.iteration_step):
        metric,dice_list=test_calculate_metric(i)
        strmetric = 'net'+str(FLAGS.modeleffe)+": iter"+str(i)+":"+str(metric)+'\n'
        with open (path,"a") as f:
            f.writelines(strmetric)

