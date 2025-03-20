import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import cv2
import kornia.utils as KU
import torch.nn as nn
from PIL import Image
import math
import numpy as np
from torchvision.transforms import transforms
import os

img_path = '/root/autodl-tmp/unsupervised deraining/datasets/CSD/test/Snow'
targeet_path = '/root/autodl-tmp/unsupervised deraining/datasets/CSD/test/Gt'

img_list = sorted(os.listdir(img_path))
num_img = len(img_list)

def psnr(pred, gt):

    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)

def imsave(img, filename):
    img = img.squeeze().cpu()
    img = KU.tensor_to_image(img) * 255.
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@MODEL_REGISTRY.register()
class CLIPNAFNetModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(CLIPNAFNetModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models

        load_path2 = self.opt['path'].get('pretrain_network_g', None)
        if load_path2 is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path2, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):

        self.net_g.train()

        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')

            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path2 = self.opt['path'].get('pretrain_network_g', None)

            if load_path2 is not None:
                self.load_network(self.net_g_ema, load_path2, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight

            self.net_g_ema.eval()
            # self.net_ir_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('bgm_opt'):
            self.cri_bgm = build_loss(train_opt['bgm_opt']).to(self.device)
        else:
            self.cri_bgm = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)


    def feed_data(self, data1, data2):
        self.lq1 = data1['lq'].to(self.device)
        self.gt1 = data1['gt'].to(self.device)
        if 'gt' in data2:
            self.lq2 = data2['lq'].to(self.device)
            self.gt2 = data2['gt'].to(self.device)

    def feed_val_data(self, data):
        self.lq_val = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt_val = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # optimize net_g

        self.optimizer_g.zero_grad()

        #LR和HR融合以后输入到生成器中
        self.output, _ = self.net_g(self.lq1)
        _, self.mid = self.net_g(self.lq2)

        l_g_total = 0
        # l_ir_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output, self.gt1) + self.cri_pix(self.mid, self.gt2)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

        if self.cri_ssim:
            l_g_ssim= self.cri_ssim(self.output, self.gt1) + self.cri_ssim(self.mid, self.gt2)
            l_g_total += l_g_ssim
            loss_dict['l_g_ssim'] = l_g_ssim

        # perceptual loss
        if self.cri_perceptual:
            l_g_percep, l_ir_style = self.cri_perceptual(self.output, self.gt1) + self.cri_perceptual(self.mid, self.gt2)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            # if l_g_style is not None:
            #     l_g_total += l_g_style
            #     loss_dict['l_g_style'] = l_g_style


            l_g_total.backward(retain_graph=True)
            self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_ir_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, _  = self.net_g_ema(self.lq_val)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, _  = self.net_g(self.lq_val)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_val_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt_val

            # tentative for out of GPU memory
            del self.lq_val
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq_val.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt_val.detach().cpu()
        return out_dict

    def ceshi(self):
        print("laileao")
        device_ids = [i for i in range(torch.cuda.device_count())]
        net_ir = nn.DataParallel(self.net_g, device_ids=device_ids)
        net_ir.cuda()
        net_ir.eval()
        transform = transforms.ToTensor()
        PSNR = 0
        for img in img_list:
            # print(img)
            image = Image.open(img_path + '/' + img).convert('RGB')
            target = Image.open(targeet_path + '/' + img).convert('RGB')
            image = transform(image)
            target = transform(target)
            image = image.cuda()
            target = target.cuda()
            [A, B, C] = image.shape
            image = image.reshape([1, A, B, C])
            [A, B, C] = target.shape
            target = target.reshape([1, A, B, C])
            with torch.set_grad_enabled(False):
                B,C,H,W = image.size()
                _, _, h_old, w_old = image.size()
                h_pad = (h_old // 16 + 1) * 16 - h_old
                w_pad = (w_old // 16+ 1) * 16 - w_old
                img_lq = torch.cat([image, torch.flip(image, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                pre = net_ir(img_lq)
                pre= pre[:,:,:H,:W]
            psnr_out = psnr(pre, target)
            PSNR += psnr_out
        print("PSNR =", PSNR / num_img)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        # self.save_network(self.net_d, 'net_d', current_iter)
        # self.save_training_state(epoch, current_iter)