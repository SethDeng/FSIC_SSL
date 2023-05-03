import torch
import torch.nn as nn
import torch.nn.functional as F
from .emd_utils import *
from .resnet import ResNet

class DeepEMD_CL(nn.Module):

    def __init__(self, args, mode='meta'):
        super().__init__()

        self.mode = mode
        self.args = args

        self.encoder = ResNet(args=args)

        # contrastive settings
        self.encoder_m = ResNet(args=args)
        self.add_nonlinear = nn.Linear(640, self.args.mlp_dim)
        self.add_nonlinear_m = nn.Linear(640, self.args.mlp_dim)

        # create the queue
        self.register_buffer("queue", torch.randn(self.args.mlp_dim, self.args.queue_len))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if self.mode == 'pre_train':
            self.fc = nn.Linear(640, self.args.num_class)
        
        # 4 angle classifier
        self.trans_num = 4
        self.Rotation_classifier = nn.Sequential(nn.Linear(640, self.trans_num), nn.ReLU())

    def forward(self, input):
        if self.mode == 'meta':
            support, query = input
            return self.emd_forward_1shot(support, query)

        elif self.mode == 'encoder':
            if self.args.deepemd == 'fcn':
                dense = True
            else:
                dense = False
            return self.encode(input, dense)

        elif self.mode == 'pre_train':
            return self.pre_train_forward(input)

        elif self.mode == 'rot':
            return self.rot_train_forward(input)

        elif self.mode == 'contrastive':
            return self.contrastive_train_forward(input)

        else:
            raise ValueError('Unknown mode')

    def pre_train_forward(self, input):
        return self.fc(self.encode(input, dense=False).squeeze(-1).squeeze(-1))

    
    def rot_train_forward(self, input):
        
        rot_label = torch.arange(self.trans_num, dtype=torch.int8).repeat(input.size(0)).long().cuda()

        images = input.view(input.size(0)*input.size(1), input.size(2), input.size(3), input.size(4))

        rot_pred = self.Rotation_classifier(self.encode(images, dense=False).squeeze(-1).squeeze(-1))
        rot_loss = F.cross_entropy(rot_pred, rot_label)
        return rot_loss, rot_pred, rot_label

    def get_weight_vector(self, A, B):

        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def emd_forward_1shot(self, proto, query):
        proto = proto.squeeze(0)

        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)

        similarity_map = self.get_similiarity_map(proto, query)
        if self.args.solver == 'opencv' or (not self.training):
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
        else:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')
        return logits

    def get_sfc(self, support):
        support = support.squeeze(0)

        # init the proto
        SFC = support.view(self.args.shot, -1, 640, support.shape[-2], support.shape[-1]).mean(dim=0).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([SFC], lr=self.args.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

        # crate label for finetune
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)

        with torch.enable_grad():
            for k in range(0, self.args.sfc_update_step):
                rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
                for j in range(0, self.args.way * self.args.shot, self.args.sfc_bs):
                    selected_id = rand_id[j: min(j + self.args.sfc_bs, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.emd_forward_1shot(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        if solver == 'opencv':  # use openCV solver

            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

                    similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

            temperature=(self.args.temperature/num_node)
            logitis = similarity_map.sum(-1).sum(-1) *  temperature
            return logitis

        elif solver == 'qpth':
            weight_2 = weight_2.permute(1, 0, 2)
            similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
            weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])

            _, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2,form=self.args.form, l2_strength=self.args.l2_strength)

            logitis=(flows*similarity_map).view(num_query, num_proto,flows.shape[-2],flows.shape[-1])
            temperature = (self.args.temperature / num_node)
            logitis = logitis.sum(-1).sum(-1) *  temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x


    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.args.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if self.args.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map
        
    def encode(self, x, dense=True):

        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            x = x.squeeze(-1)
            return x

        else:
            x = self.encoder(x)
            if dense == False:
                x = F.adaptive_avg_pool2d(x, 1)
                return x
            if self.args.feature_pyramid is not None:
                x = self.build_feature_pyramid(x)
        return x

    def build_feature_pyramid(self, feature):
        feature_list = []
        for size in self.args.feature_pyramid:
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param, param_m in zip(self.encoder.parameters(), self.encoder_m.parameters()):
            param_m.data = param_m.data * self.args.con_momentum + param.data * (1. - self.args.con_momentum)
        for param, param_m in zip(self.add_nonlinear.parameters(), self.add_nonlinear_m.parameters()):
            param_m.data = param_m.data * self.args.con_momentum + param.data * (1. - self.args.con_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # print(batch_size)
        assert self.args.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.args.queue_len  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle(self, x):
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size).cuda() # Returns a 1D tensor of random permutation of integers from 0 to n - 1.
        idx_unshuffle = torch.argsort(idx_shuffle)  # Returns the indices that sort a tensor along a given dimension in ascending order by value.
        x = x[idx_shuffle]
        return x, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    def contrastive_train_forward(self, input):
        im1, im2 = input

        # compute query features
        batch_size = im1.size(0)
        q_1, q_2 = torch.split(im1, batch_size//2)
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        q_1 = self.encoder(q_1)
        q_2 = self.encoder(q_2)
        q = torch.cat((q_1,q_2), dim = 0)
        q = F.adaptive_avg_pool2d(q, (1,1)).squeeze(-1).squeeze(-1)
        q = self.add_nonlinear(q)
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im2, idx_unshuffle = self._batch_shuffle(im2)
            im_two_1, im_two_2 = torch.split(im2, batch_size//2)
            k_1 = self.encoder_m(im_two_1)
            k_2 = self.encoder_m(im_two_2)
            k = torch.cat((k_1,k_2), dim = 0)
            k = F.adaptive_avg_pool2d(k, (1,1)).squeeze(-1).squeeze(-1)
            k = self.add_nonlinear_m(k)
            k = F.normalize(k, dim=1)
            k = self._batch_unshuffle(k, idx_unshuffle)

        # n:batch_size, c:mlp_dim, k:queue_len    
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits_con = torch.cat([l_pos, l_neg], dim=1)
        logits_con /= self.args.temparature
        
        self._dequeue_and_enqueue(k)
        return logits_con
