import torch
from torch.autograd import Variable
from dataset.build import build_train_dataset, build_test_dataset
from modeling.net import build_network
from modeling.criterion import build_criterions
from modeling.output import build_output_method
from modeling.input_preprocessing import build_test_input

from solver.build import make_optimizer, make_lr_scheduler
import os
import os.path as osp
import time
from lib.afm_op import afm
from lib.squeeze_to_lsg import lsgenerator
from util.progbar import progbar
from lib.afm_op import afm
import cv2

class AFM(object):
    def __init__(self,cfg):
        self.input_method = build_test_input(cfg)
        self.train_dataset = build_train_dataset(cfg)
        self.test_dataset  = build_test_dataset(cfg)
        self.model = build_network(cfg).cuda()
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                        lr = cfg.SOLVER.BASE_LR,
                                        momentum=cfg.SOLVER.MOMENTUM,
                                        weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        self.lr_schedulr = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                        milestones=cfg.SOLVER.STEPS,
                                        gamma=cfg.SOLVER.GAMMA)
        # self.criterion = build_criterions(cfg)
        # self.optimizer = make_optimizer(cfg, self.model)
        # self.lr_schedulr = make_lr_scheduler(cfg,self.optimizer)

        self.saveDir = cfg.SAVE_DIR
        self.weightDir = osp.join(self.saveDir, 'weight')
        self.resultDir = osp.join(self.saveDir,'results')

        self.output_method = build_output_method(cfg)

        if osp.isdir(self.weightDir) is not True:
            os.makedirs(self.weightDir)

        if osp.isdir(self.resultDir) is not True:
            os.makedirs(self.resultDir)

        self.logger = {'train': open(osp.join(self.saveDir,'train.log'),'a+')}
        self.current_epoch = 0


    def load_weight_by_epoch(self, epoch):
        assert isinstance(epoch, int)

        if epoch>0:
            modelFile = 'model_{}.pth.tar'.format(epoch)
            optimFile = 'optimState_{}.pth.tar'.format(epoch)
        else:
            modelFile = 'model_final.pth.tar'
            optimFile = 'optimState_final.pth.tar'

        self.current_epoch = epoch
        self.model.load_state_dict(torch.load(osp.join(self.weightDir, modelFile),map_location='cpu'))

        try:
            self.optimizer.load_state_dict(torch.load(osp.join(self.weightDir, optimFile),map_location='cpu'))
        except:
            pass
        
    def save_weight_by_epoch(self, epoch):
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.get(0)
        modelFile = 'model_{}.pth.tar'.format(epoch)
        optimFile = 'optimState_{}.pth.tar'.format(epoch)
        
        torch.save(self.model.state_dict(), os.path.join(self.weightDir, modelFile))
        torch.save(self.optimizer.state_dict(), osp.join(self.weightDir, optimFile))

    def train(self, cfg, current_epoch = 0):
        def step(epoch):
            self.model.train()
            bar = progbar(len(self.train_dataset), width=10)
            self.lr_schedulr.step(epoch=epoch)
            print('\n Training AT epoch = {}'.format(epoch))
            print('current learning rate = {}\n'.format(self.lr_schedulr.get_lr()))
            avgLoss = 0
            for i, (image, afmap) in enumerate(self.train_dataset):  
                self.optimizer.zero_grad()            
                image_var = Variable(image).cuda()
                afmap_var = Variable(afmap).cuda()                
                afmap_pred = self.model(image_var)
                loss = self.criterion(afmap_pred, afmap_var)
                loss.backward()
                self.optimizer.step()                
                avgLoss = (avgLoss*i + loss.item()) / (i+1)

                log = 'Epoch: [%d][%d/%d] Err %1.4f\n' % (epoch, i, len(self.train_dataset), avgLoss)
                self.logger['train'].write(log)

                bar.update(i, [('avgLoss', avgLoss)])
            
            log = '\n * Finished training epoch # %d     Loss: %1.4f\n' % (epoch, avgLoss)
            self.logger['train'].write(log)
            print(log)
            return avgLoss

        self.current_epoch = 0
        if current_epoch > 0:
            self.load_weight_by_epoch(current_epoch)
        
        # self.model.cuda()
        # self.model.train()

        for epoch in range(self.current_epoch+1, cfg.SOLVER.NUM_EPOCHS+1):

            avgLoss = step(epoch)
            if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                self.save_weight_by_epoch(epoch)
            
    def test(self, cfg, epoch = -1):
        self.model.eval()
        
        self.load_weight_by_epoch(epoch)
        
        # self.model.cuda()

        for name, dataset in zip(cfg.DATASETS.TEST, self.test_dataset):

            print('Testing on {} dataset'.format(name.upper()))
            
            bar = progbar(target = len(dataset))
            start_time = time.time()
            for i, (image, lines, shape_info, fname) in enumerate(dataset):
                image_var = Variable(image).cuda()
                lines_var = Variable(lines).cuda()
                shape_info = Variable(shape_info).cuda()

                # image_var = self.input_method(image_var)
                afmap_pred = self.model(image_var)

                lines_pred,xx,yy = lsgenerator(afmap_pred[0].cpu().data.numpy())
                afmap_gt, label = afm(lines_var,shape_info, image_var.shape[2],image_var.shape[3])
                image_raw = cv2.imread(osp.join(dataset.dataset.data_root,'images',fname[0]))
                # import pdb
                # pdb.set_trace()
                output_dict = {
                    'image': image_raw,
                    'image_resized': image_var[0].cpu().data.numpy(),
                    'lines_pred_resized': lines_pred,
                    'lines_gt': lines.numpy(),
                    'afmap_pred': afmap_pred[0].cpu().data.numpy(),
                    'afmap_gt': afmap_gt[0].cpu().data.numpy(),
                    'fname': fname[0],
                    'output_dir': osp.join(self.resultDir,
                            name),

                }
                self.output_method(output_dict, cfg)
                bar.update(i)

            end_time = time.time()

            print('Total images: {}'.format(len(dataset)))
            print('Total time: {} ellapsed for {}'.format(end_time-start_time, cfg.TEST.OUTPUT_MODE))
            print('Frames per Second: {}'.format(len(dataset)/(end_time-start_time)))




