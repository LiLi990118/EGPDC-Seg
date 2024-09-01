import gc
import os
import pickle
import logging
from copy import deepcopy, copy
from .discriminator import Discriminator
import numpy as np
from torch.utils.data import DataLoader
from local.fusiontrans import UFusionNet
from src.loss import DiceLoss, MultiClassDiceLoss, Focal_Loss, FocalLoss_Ori, Marginal_Loss,GANLoss
from src.utils import save_on_batch, sigmoid_rampup
from PIL import Image
import torch
import torch.nn.functional as F
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, local_valdata, device,save_img_path):
        """Client object is initiated by the center server."""
        self.id = client_id
        #本地训练数据
        self.data = local_data
        self.valdata = local_valdata
        self.device = device
        self.__model = None
        self.save_img_path = save_img_path


    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        #本地数据集
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.valdataloader = DataLoader(self.valdata, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        #self.criterion = WeightedDiceLoss
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]



    def client_update(self,idx):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)
        if idx==0:
            self.ema_model = deepcopy(self.model)
            for param in self.ema_model.parameters():
                param.detach_()
            self.ema_model.to(self.device)
            self.ema_model.eval()
            #D
            self.Dis_model = Discriminator(5, 64)
            self.Dis_model.train()
            self.Dis_model.to(self.device)
            Dis_optimizer = eval(self.optimizer)(self.Dis_model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=5e-4)
            Dis_optimizer.zero_grad()
        client_train_losses =[]
        optimizer = eval(self.optimizer)(self.model.parameters(), lr=2e-4,betas=(0.9, 0.999), weight_decay=5e-4)
        loss_g = GANLoss(gan_type='vanilla').to('cuda')
        for e in range(self.local_epoch):
            for dataset, labelName in self.dataloader:
                data = dataset['image'].float().to(self.device)
                labels = dataset['label'].float().to(self.device)
                if idx==0:
                    with torch.no_grad():
                        ema_output = self.ema_model(data.clone())
                        ema_output_soft = torch.softmax(ema_output, dim=1)
                        mask_ema = torch.argmax(ema_output, dim=1)
                optimizer.zero_grad()
                outputs = self.model(data)
                outputssoft = torch.softmax(outputs, dim=1)
                mask_pre = torch.argmax(outputs, dim=1)
                if idx==0:
                    weights = [0, 2, 0, 0, 0]
                    ema_l_dice, ema_mDice = MultiClassDiceLoss()(outputssoft, labels, idx,weights)
                    print(ema_l_dice,'ema_l_dice')
                    #consistency_dist
                    #与专家生成的标签做损失
                    #consistency_loss = F.mse_loss(outputssoft, ema_output_soft)
                    c_weights = [1, 2, 1, 2, 1]
                    consistency_loss,_ = MultiClassDiceLoss()(outputssoft, mask_ema.unsqueeze(1),idx,c_weights)
                    print(mask_ema,'mask_ema')
                    print(consistency_loss,'consistency_loss')
                    fake_g_pred = self.Dis_model(outputssoft)
                    print(fake_g_pred, 'fake_g_pred')
                    #生成器中为真
                    l_g_real = loss_g(fake_g_pred, True, is_disc=True)
                    print(l_g_real,'l_g_real')
                    #floss = 0
                    # w_consistence = (e+0.001)/(self.local_epoch*2)
                    #
                    # threshold = (0.75 + 0.25 * sigmoid_rampup(e, self.local_epoch)) * np.log(2)
                    # mask = (uncertainty < threshold).float()
                    # consistency_dist = torch.sum(mask * consistency_loss) / (2 * torch.sum(mask) + 1e-16)
                    mDiceLoss = 1 * ema_l_dice + consistency_loss + l_g_real

                    img_path = 'C:/dataset/Fed_data/300slices/quebiaoqian/'

                    for i in range(mask_pre.shape[0]):
                        labels_arr = labels[i][0].cpu().numpy()
                        mask_arr = mask_pre[i].cpu().numpy()
                        # 定义颜色映射表
                        color_map = {
                            0: (0, 0, 0),  # 类别0为黑色
                            1: (221, 160, 221),  # 类别1为梅红色
                            2: (0, 128, 0),  # 类别2为深绿色
                            3: (128, 128, 0),  # 类别3为深黄色
                            4: (240, 255, 255),  # 类别4为深蓝色
                            5: (128, 0, 128),  # 类别5为深紫色
                            6: (0, 255, 255),  # 类别6为青色
                            7: (128, 128, 128),  # 类别7为灰色
                            8: (255, 0, 0),  # 类别8为红色
                            9: (0, 255, 0),  # 类别9为绿色
                            10: (255, 255, 0),  # 类别10为黄色
                            11: (0, 0, 255),  # 类别11为蓝色
                        }

                        # 将像素值映射成颜色
                        colors = np.zeros((*mask_arr.shape, 3), dtype=np.uint8)
                        for j, color in color_map.items():
                            colors[mask_arr == j, :] = color

                        img = Image.fromarray(colors)
                        client_dir_path = img_path + labelName[i][:-4]
                        if not os.path.exists(client_dir_path):
                            os.makedirs(client_dir_path)
                        img.save(client_dir_path + ".png")

                    for i in range(mask_ema.shape[0]):
                        labels_arr = labels[i][0].cpu().numpy()
                        mask_ema_arr = mask_ema[i].cpu().numpy()
                        # 定义颜色映射表
                        color_map = {
                            0: (0, 0, 0),  # 类别0为黑色
                            1: (221, 160, 221),  # 类别1为梅红色
                            2: (0, 128, 0),  # 类别2为深绿色
                            3: (128, 128, 0),  # 类别3为深黄色
                            4: (240, 255, 255),  # 类别4为深蓝色
                            5: (128, 0, 128),  # 类别5为深紫色
                            6: (0, 255, 255),  # 类别6为青色
                            7: (128, 128, 128),  # 类别7为灰色
                            8: (255, 0, 0),  # 类别8为红色
                            9: (0, 255, 0),  # 类别9为绿色
                            10: (255, 255, 0),  # 类别10为黄色
                            11: (0, 0, 255),  # 类别11为蓝色
                        }
                    # 将像素值映射成颜色
                        colors = np.zeros((*mask_ema_arr.shape, 3), dtype=np.uint8)
                        for j, color in color_map.items():
                            colors[mask_ema_arr == j, :] = color
                        # 保存预测标签
                        img = Image.fromarray(colors)
                        client_dir_path = img_path + labelName[i][:-4]
                        if not os.path.exists(client_dir_path):
                            os.makedirs(client_dir_path)
                        img.save(client_dir_path + "_ema.png")
                    # 保存标签
                    # 将像素值映射成颜色
                        gt_colors = np.zeros((*labels_arr.shape, 3), dtype=np.uint8)
                        for j, color in color_map.items():
                            gt_colors[labels_arr == j, :] = color
                        img = Image.fromarray(gt_colors)
                        img.save(img_path + labelName[i][:-4] + "_gt.png")

                else:
                    weights = [1, 2, 1, 2, 1]
                    #weights = [1, 1, 1, 1, 1]
                    mDiceLoss, mDice = MultiClassDiceLoss()(outputssoft, labels,idx,weights)
                    #floss = FocalLoss_Ori()(outputs,labels)
                # loss = mDiceLoss+floss
                loss = mDiceLoss
                #loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                client_train_losses.append(loss.item())
                # Dis
                if idx == 0:
                    # # real
                    real_d_pred = self.Dis_model(ema_output_soft)
                    l_d_real = loss_g(real_d_pred, True, is_disc=True)
                    print(l_d_real, 'l_d_real')
                    l_d_real.backward()
                    # fake
                    fake_d_pred = self.Dis_model(outputssoft.detach().clone())
                    #print('fake_d_pred:',fake_d_pred)
                    l_d_fake = loss_g(fake_d_pred, False, is_disc=True)
                    print(l_d_fake, 'l_d_fake')
                    l_d_fake.backward()
                    Dis_optimizer.step()
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        client_avg_loss = np.average(client_train_losses)
        message = f"\t[Client {str(self.id).zfill(4)}] ...finished training!\
            \n\t=> Tain loss: {client_avg_loss:.4f}\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()
        return client_avg_loss,optimizer

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)
        client_id = self.id
        test_loss, dice_pred = 0, 0
        val_losses = []
        val_Dice = []
        with torch.no_grad():
            for dataset, labelName in self.valdataloader:
                #data, labels = data.float().to(self.device), labels.long().to(self.device)
                data = dataset['image'].float().to(self.device)
                labels = dataset['label'].float().to(self.device)
                outputs = self.model(data)
                outputssoft = torch.softmax(outputs,dim=1)
                mask = torch.argmax(outputs, dim=1)

                img_path = self.save_img_path + 'client_img/'+str(client_id)+'/'
                for i in range(labels.shape[0]):
                    labels_arr = labels[i][0].cpu().numpy()
                    mask_arr = mask[i].cpu().numpy()
                    # 定义颜色映射表
                    color_map = {
                        0: (0, 0, 0),  # 类别0为黑色
                        1: (221, 160, 221),  # 类别1为梅红色
                        2: (0, 128, 0),  # 类别2为深绿色
                        3: (128, 128, 0),  # 类别3为深黄色
                        4: (240, 255, 255),  # 类别4为深蓝色
                        5: (128, 0, 128),  # 类别5为深紫色
                        6: (0, 255, 255),  # 类别6为青色
                        7: (128, 128, 128),  # 类别7为灰色
                        8: (255, 0, 0),  # 类别8为红色
                        9: (0, 255, 0),  # 类别9为绿色
                        10: (255, 255, 0),  # 类别10为黄色
                        11: (0, 0, 255),  # 类别11为蓝色
                    }

                    # 将像素值映射成颜色
                    colors = np.zeros((*mask_arr.shape, 3), dtype=np.uint8)
                    for j, color in color_map.items():
                        colors[mask_arr == j, :] = color

                    # 将颜色数组保存为图片
                    # img = Image.new('RGB', (colors.shape[1], colors.shape[0]))
                    # data = colors.reshape(-1, colors.shape[2]).tolist()
                    # img.putdata(data)
                    # img.save(img_path + labelName[i][:-4] + ".png")
                    img = Image.fromarray(colors)
                    # 检查目录是否存在，如果不存在则创建
                    dir_path = img_path + labelName[i][:-4]
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    img.save(dir_path + ".png")


                    # 将像素值映射成颜色
                    gt_colors = np.zeros((*labels_arr.shape, 3), dtype=np.uint8)
                    for j, color in color_map.items():
                        gt_colors[labels_arr == j, :] = color

                    img = Image.fromarray(gt_colors)
                    img.save(img_path + labelName[i][:-4] + "_gt.png")
                #save_on_batch(data, labels, outputssoft, labelName, img_path)
                if client_id==0:
                    weights=[0,2,0,0,0]
                else:
                    weights=[1,2,1,2,1]
                mDiceLoss, mDice = MultiClassDiceLoss()(outputssoft, labels,client_id,weights)
                # floss = FocalLoss_Ori()(outputs, labels)
                test_loss = mDiceLoss

                val_losses.append(test_loss.item())
                val_Dice.append(mDice.item())
                # predicted = outputs.argmax(dim=1, keepdim=True)
                # correct += predicted.eq(labels.view_as(predicted)).sum().item()

                # dice_pred_t = dice_show(outputs, labels)
                # dice_pred += dice_pred_t

                #iou_pred += iou_pred_t
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = np.average(val_losses)
        #test_dice = dice_pred / len(self.valdataloader)
        test_dice = np.average(val_Dice)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> test_dice: {100. * test_dice:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()
        return test_loss, test_dice





