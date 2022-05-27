class Contrastive(torch.nn.Module):
    def __init__(self,m):
        super(Contrastive, self).__init__()
        self.m = m

    def ed(self,a,b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt((sum_sq_a+sum_sq_b-2*a.mm(bt))/a.shape[1])

    def pair(self, labels):
        labels_np = labels.cpu().numpy()
        total_num = labels_np.shape[0]
        if total_num % 2 != 0: total_num -= 1
        #pair_idx [[idx1,idx2,i(=0 when diff/ =1 when same)],...]
        pair_idx = []
        idx = []
        for i in range(total_num):
            idx.append(i)

        labels_list = labels_np.tolist()
        dct = defaultdict(list)
        for key, value in [(v, i) for i, v in enumerate(labels_list)]:
            dct[key].append(value)
        for label in dct:
            if len(dct[label]) > 1:
                a = random.sample(dct[label], 2)
                if a[0] in idx and a[1] in idx:
                    for i in a:
                        idx.remove(i)
                    a.append(1)
                    pair_idx.append(a)

        while idx != []:
            a = random.sample(idx, 2)
            for i in a:
                idx.remove(i)
            a.append(0)
            pair_idx.append(a)
            
        return pair_idx
        
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        device = labels.device
        pair_idx = self.pair(labels)
        num = len(pair_idx)
        total_loss = 0
        for line in pair_idx:
            em1,em2,i= embeddings[line[0]],embeddings[line[1]],line[2]
            d =self.ed(em1.reshape(1, -1),em2.reshape(1, -1))
            loss = (1-i) * (max(0,self.m-d))**2 + i*d**2
            total_loss += loss
        loss = total_loss / num
        
        return torch.FloatTensor(loss).to(device)
