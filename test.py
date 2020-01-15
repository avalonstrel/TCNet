import torch



def test(model, data, config, verbose=False):
    config.obj = config.obj.split('_')[0]

    if verbose:
        print('test: simple')

    if config.obj in ('shoes', 'chairs') and config.distance in ('sq', 'cos'):
        accu_simple = test_simple(model, data, config)

    else:

        if config.obj in ('shoes', 'chairs') and not (config.distance in ('sq', 'cos')):
            sep = 50

        if config.obj in ('shoes_v2', 'hairstyle') and config.distance in ('sq', 'cos'):
            sep = 300

        if config.obj in ('shoes_v2', 'hairstyle') and not (config.distance in ('sq', 'cos')):
            sep = 10

        if config.obj == 'sketchy' and config.distance in ('sq', 'cos'):
            sep = 200

        if config.obj == 'sketchy' and not (config.distance in ('sq', 'cos')):
            sep = 1

        accu_simple = test_simple_manidata(model, data, config, sep, verbose)


    if verbose:
        print('test: complex')

    data.feat_imgs = None

    if config.obj in ('shoes', 'chairs') and config.distance in ('sq', 'cos'):
        sep = 50

    if config.obj in ('shoes', 'chairs') and not (config.distance in ('sq', 'cos')):
        sep = 5

    if config.obj == 'shoes_v2' and config.distance in ('sq', 'cos'):
        sep = 30

    if config.obj == 'shoes_v2' and not (config.distance in ('sq', 'cos')):
        sep = 1

    if config.obj in ('sketchy', 'hairstyle') and config.distance in ('sq', 'cos'):
        #sep = 10
        sep = 10

    if config.obj in ('sketchy', 'hairstyle') and not (config.distance in ('sq', 'cos')):
        sep = 0

    accu_complex = test_complex_manidata(model, data, config, sep, verbose)

    return accu_simple, accu_complex







class TestData:
    def __init__(self, test_skts, test_imgs, test_idxs):
        self.test_skts = test_skts
        self.test_imgs = test_imgs
        self.test_idxs = test_idxs
        self.feat_imgs = None

    def get_test(self, flag):
        return self.test_skts, self.test_imgs, self.test_idxs


def test_simple(model, data, config):

    # args
    device = config.device
    distance = config.distance

    # get data
    test_skts, test_imgs, test_idxs = data.get_test(False)
    #test_skts, test_imgs, test_idxs = test_skts.to(device), test_imgs.to(device), test_idxs.to(device)
    ns, np = test_skts.size(0), test_imgs.size(0)
    num_sep = 300

    # get feature
    model.eval()
    with torch.no_grad():

        feat_skts = model(test_skts.to(device)).cpu()

        if hasattr(data, 'feat_imgs') and data.feat_imgs is not None:
            feat_imgs = data.feat_imgs
        else:
            feat_imgs_list = []
            curr_idx = 0
            while curr_idx < np:
                feat_imgs_list.append(model(test_imgs[curr_idx:curr_idx+num_sep].to(device)).cpu())
                curr_idx += num_sep

            feat_imgs = torch.cat(feat_imgs_list)
            data.feat_imgs = feat_imgs

        if not config.fix_bn:
            model.train()

        # compute distance
        feat_skts = feat_skts.unsqueeze(1).repeat(1, np, 1)
        feat_imgs = feat_imgs.unsqueeze(0).repeat(ns, 1, 1)

        if distance == 'cos':
            res = -torch.nn.functional.cosine_similarity(feat_skts, feat_imgs, dim=2)
        elif distance == 'sq':
            res = torch.norm(feat_skts - feat_imgs, dim=2).pow(2)
        else:
            res = distance(feat_skts, feat_imgs)

    # compute top-1, top-10 accuracy
    retrieval_idxs = res.sort(dim=1)[1][:,:10]
    test_idxs = test_idxs.type(retrieval_idxs.type())
    accu1 = (retrieval_idxs[:,0] == test_idxs).float().mean().item()
    accu10 = accu1
    for i in range(1, 10):
        accu = (retrieval_idxs[:,i] == test_idxs).float().mean().item()
        accu10 += accu
    return {'top-1':accu1, 'top-10':accu10}


def test_simple_manidata(model, data, config, sep, verbose):

    if sep <= 0:
        return None

    # get data
    test_skts, test_imgs, test_idxs = data.get_test(False)
    ns = test_skts.size(0)
    niter = ns // sep + int(ns % sep > 0)

    curr_idx = 0
    right1, right10, num, i = 0, 0, 0, 0
    while curr_idx < ns:
        n = min(sep, ns - curr_idx)
        data = TestData(test_skts[curr_idx:curr_idx+n],
                        test_imgs,
                        test_idxs[curr_idx:curr_idx+n])
        accu = test_simple(model, data, config)
        curr_idx += sep
        accu1, accu10 = accu['top-1'], accu['top-10']


        num += n
        right1 += accu1 * n
        right10 += accu10 * n
        i += 1
        if verbose:
            print('\r%2d/%3d, right1:{:.0f}, right10:{:.0f}, num:%6d'.format(right1, right10)%(i,niter, num), end='')
    if verbose:
        print()

    return {'top-1':right1/num, 'top-10':right10/num}



def test_complex(model, data, config):
    # args
    device = config.device
    distance = config.distance

    # get data
    test_skts, test_imgs, test_idxs = data.get_test(True)
    #test_skts, test_imgs, test_idxs = test_skts.to(device), test_imgs.to(device), test_idxs.to(device)
    ns, np = test_skts.size(0), test_imgs.size(0)
    num_sep = 300

    # get feature
    model.eval()
    with torch.no_grad():

        feat_skts = model(test_skts.to(device)).cpu()

        if hasattr(data, 'feat_imgs') and data.feat_imgs is not None:
            feat_imgs = data.feat_imgs
        else:
            feat_imgs_list = []
            curr_idx = 0
            while curr_idx < np:
                feat_imgs_list.append(model(test_imgs[curr_idx:curr_idx + num_sep].to(device)).cpu())
                curr_idx += num_sep

            feat_imgs = torch.cat(feat_imgs_list)
            data.feat_imgs = feat_imgs

        if not config.fix_bn:
            model.train()

        # compute distance
        ns, np = ns//10, np//10
        feat_skts = feat_skts.view(10, ns, -1).transpose(0, 1).cpu()
        feat_imgs = feat_imgs.view(10, np, -1).transpose(0, 1).cpu()

        feat_skts = feat_skts.unsqueeze(1).repeat(1, np, 1, 1)
        feat_imgs = feat_imgs.unsqueeze(0).repeat(ns, 1, 1, 1)

        if distance == 'cos':
            res = -torch.nn.functional.cosine_similarity(feat_skts, feat_imgs, dim=3)
        elif distance == 'sq':
            res = torch.norm(feat_skts - feat_imgs, dim=3).pow(2)
        else:
            res = distance(feat_skts, feat_imgs)
        res = res.mean(dim = 2)

    # compute top-1, top-10 accuracy
    retrieval_idxs = res.sort(dim=1)[1][:,:10]
    test_idxs = test_idxs.type(retrieval_idxs.type())
    accu1 = (retrieval_idxs[:,0] == test_idxs).float().mean().item()
    accu10 = accu1
    for i in range(1, 10):
        accu = (retrieval_idxs[:,i] == test_idxs).float().mean().item()
        accu10 += accu
    return {'top-1':accu1, 'top-10':accu10}


def test_complex_manidata(model, data, config, sep, verbose):

    if sep <= 0:
        return None

    # get data
    test_skts, test_imgs, test_idxs = data.get_test(True)
    ns = test_skts.size(0) // 10
    niter =  ns // sep + int(ns % sep > 0)

    curr_idx = 0
    right1, right10, num, i = 0, 0, 0, 0
    while curr_idx < ns:

        n = min(sep, ns - curr_idx)

        idxs = []
        for k in range(10):
            idxs.extend(range(curr_idx+k*ns, curr_idx+k*ns+n))
        
        data = TestData(test_skts[idxs],
                        test_imgs,
                        test_idxs[curr_idx:curr_idx+sep])
        curr_idx += sep
        accu = test_complex(model, data, config)
        accu1, accu10 = accu['top-1'], accu['top-10']


        num += n
        right1 += accu1 * n
        right10 += accu10 * n
        i += 1
        
        if verbose:
            print('\r%2d/%3d, right1:{:.0f}, right10:{:.0f}, num:%6d'.format(right1, right10) % (i, niter, num),
                  end='')
    if verbose:
        print()

    return {'top-1':right1/num, 'top-10':right10/num}
