#encoding: utf-8
from fastai.conv_learner import *


class DPNConvnetBuilder():
    """
    # Args
        ps: float list, dropout rate of dense layers
    """
    def __init__(self, f,
                 n_classes,
                 is_multi,
                 is_reg,
                 ps=None,
                 xtra_fc=None,
                 xtra_cut=0,
                 custom_head=None,
                 pretrained=True):
        
        self.f, self.n_classes = f, n_classes
        self.is_multi = is_multi
        self.is_reg = is_reg
        
        if xtra_fc is None:
            xtra_fc = [512]
        
        if ps is None:
            self.ps = [0.5]
        else:
            self.ps = [0.]
        
        # 提取网络每一层重新构造网络
        dpn_model = f(pretrained='imagenet+5k')
        dpn = cut_model(dpn_model, 2)[0]
        # 初始化第一层卷积层
        blocks = []
        blocks = list(dpn.children())
        self.lr_cut = len(blocks)
        print('backbone has ', len(blocks), ' blocks')
        input_block = list(blocks[0].children())
        w = input_block[0] .weight
        input_block[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        input_block[0] .weight = nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
        input_layer = nn.Sequential(*input_block)

        blocks[0] = input_layer
        
        # backbone
        self.top_model = to_gpu(nn.Sequential(*blocks))

        if custom_head:
            fc_layers = [custom_head]
        else:
            fc_layers = self.get_fc_layers()

        self.n_fc = len(fc_layers)
        # classifier
        self.fc_model = nn.Sequential(*fc_layers)
        
#         if not custom_head:
#             apply_init(self.fc_model, kaiming_normal)
        
        self.model = to_gpu(nn.Sequential(*(blocks+fc_layers)))

    @property
    def name(self):
        return f'{self.f.__name__}_{self.xtra_cut}'

    def get_fc_layers(self):
        layers = [nn.Conv2d(2688, 256, kernel_size=1, stride=1, bias=False), 
                  AdaptiveConcatPool2d(),
                  Flatten()]
        if self.ps[0] == 0:
            layers += [nn.BatchNorm1d(512), 
                       nn.Dropout(self.ps[0]), 
                       nn.Linear(512, self.n_classes)]
        else:
            layers += [nn.BatchNorm1d(512), 
                       nn.Linear(512, self.n_classes)]
            
        return layers

    def get_layer_groups(self, do_fc=False):
        if do_fc:
            return [self.fc_model]
        self.lr_cut = 32
        idxs = [self.lr_cut]
        c = children(self.top_model)
        if len(c) == 32:
            c = children(c[0])+c[1:]
        lgs = list(split_by_idxs(c, idxs))
        print('num of layer groups of backbone: ', len(lgs))
        return lgs+[self.fc_model]


class ConvLearner(Learner):
    """学习器，继承库的Learner类

    # Args
        data: databunch
        models: models
        precompute: 
    """

    def __init__(self, data, models, precompute=False, **kwargs):
        self.precompute = False
        super().__init__(data, models, **kwargs)

        if hasattr(data, 'is_multi') and not data.is_reg and self.metrics is None:
            self.metrics = [accuracy_thresh(0.5)] if self.data.is_multi else [accuracy]
        
        if precompute:
            self.save_fc1()
        self.freeze()
        self.precompute = precompute

    def _get_crit(self, data):
        if not hasattr(data, 'is_multi'):
            return super()._get_crit(data)

        return F.l1_loss if data.is_reg else F.binary_cross_entropy if data.is_multi else F.nll_loss

    @classmethod
    def pretrained(cls, f, data, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, precompute=False,
                   pretrained=True, **kwargs):
        models = DPNConvnetBuilder(f, data.c, data.is_multi, data.is_reg,
                                   ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut,
                                   custom_head=custom_head, pretrained=pretrained)

        return cls(data, models, precompute, **kwargs)

    @classmethod
    def lsuv_learner(cls, f, data, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, precompute=False,
                     needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=False, **kwargs):
        models = ConvnetBuilder(f, data.c, data.is_multi, data.is_reg,
                                ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut, 
                                custom_head=custom_head, pretrained=False)
        convlearn = cls(data, models, precompute, **kwargs)
        convlearn.lsuv_init()
        return convlearn

    @property
    def model(self):
        return self.models.fc_model if self.precompute else self.models.model
    
    def half(self):
        """将模型转化为版精度，减小运算量
        """
        if self.fp16:
            return
        self.fp16 = True
        if type(self.model) != FP16:
            self.models.model = FP16(self.model)
        if not isinstance(self.models.fc_model, FP16):
            self.models.fc_model = FP16(self.models.fc_model)
    
    def float(self):
        """将模型转换为全精度
        """
        if not self.fp16:
            return
        self.fp16 = False
        if type(self.models.model) == FP16:
            self.models.model = self.model.module.float()
        if type(self.models.fc_model) == FP16:
            self.models.fc_model = self.models.fc_model.module.float()

    @property
    def data(self):
        return self.fc_data if self.precompute else self.data_

    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0, n), np.float32), chunklen=1, mode='w', rootdir=name)

    def set_data(self, data, precompute=False):
        super().set_data(data)
        if precompute:
            self.unfreeze()
            self.save_fc1()
            self.freeze()
            self.precompute = True
        else:
            self.freeze()

    def get_layer_groups(self):
        return self.models.get_layer_groups(self.precompute)

    def summary(self):
        precompute = self.precompute
        self.precompute = False
        res = super().summary()
        self.precompute = precompute

        return res

    def get_activations(self, force=False):
        tmpl = f'_{self.models.name}_{self.data.sz}.bc'

        names = [os.path.join(self.tmp_path, p+tmpl)
                 for p in ('x_act', 'x_act_val', 'x_act_test')]
        if os.path.exists(names[0]) and not force:
            self.activations = [bcolz.open(p) for p in names]
        else:
            self.activations = [self.create_empty_bcolz(
                self.models.nf, n) for n in names]

    def save_fc1(self):
        self.get_activations()
        act, val_act, test_act = self.activations
        m = self.models.top_model
        if len(self.activations[0]) != len(self.data.trn_ds):
            predict_to_bcolz(m, self.data.fix_dl, act)
        if len(self.activations[1]) != len(self.data.val_ds):
            predict_to_bcolz(m, self.data.val_dl, val_act)
        if self.data.test_dl and (len(self.activations[2]) != len(self.data.test_ds)):
            if self.data.test_dl:
                predict_to_bcolz(m, self.data.test_dl, test_act)

        self.fc_data = ImageClassifierData.from_arrays(self.data.path,
                                                       (act, self.data.trn_y), 
                                                       (val_act, self.data.val_y), 
                                                       self.data.bs, 
                                                       classes=self.data.classes,
                                                       test=test_act if self.data.test_dl else None,
                                                       num_workers=8)

    def freeze(self):
        self.freeze_to(-1)

    def unfreeze(self):
        self.freeze_to(0)
        self.precompute = False

    def predict_array(self, arr):
        precompute = self.precompute
        self.precompute = False
        pred = super().predict_array(arr)
        self.precompute = precompute

        return pred