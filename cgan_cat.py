import os
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from skimage import io
import time
import cgan

# Define the Dataset
# TODO: Move to the main cgan file so no duplicates
class SketchPhotoDataset(Dataset):
    def __init__(self, sketch_photo_pair, transform=None):
        self.sketch_photo_pair = sketch_photo_pair
        self.transform = transform
  
    def __len__(self):
        return len(self.sketch_photo_pair)

    def __getitem__(self, index):
        sketch, photo = self.sketch_photo_pair[index]
        image_sketch = io.imread(sketch)
        image_photo = io.imread(photo)

        return image_sketch, image_photo

def ProcessData(photo_url, sketch_url):
    photos_dic = {}
    sketches_dic = {}
    for root, dirs, files in os.walk(photo_url):
        # set up the keys
        for dir in dirs:
            photos_dic[photo_url + "/" +  dir] = []
        for photo in files:
            photos_dic[root].append(photo)

    for root, dirs, files in os.walk(sketch_url):
        # set up the keys
        for dir in dirs:
            sketches_dic[sketch_url + "/" + dir] = []
        for sketch in files:
            sketches_dic[root].append(sketch)

    sketch_photo_pair = {}
    # for each sketch category
    for sketch_dir in sketches_dic:
        sketches = sketches_dic[sketch_dir]
        for sketch in sketches:
            photo_name = sketch[:sketch.find('-')]
            photo_name = photo_name + ".jpg"
            category = sketch_dir[sketch_dir.rfind('/')+1:]
            if photo_name in photos_dic[photo_url + "/" + category]:
                photo_path = photo_url + "/" + category + "/" + photo_name
                sketch_path = sketch_dir + "/" + sketch
                if category not in sketch_photo_pair:
                    sketch_photo_pair[category] = []
                sketch_photo_pair[category].append((sketch_path, photo_path))
    return sketch_photo_pair

def training(sketch_photo_pair, G, D, batch_size):
    # Hyperparameters
    NUM_EPOCHS = 1
    G_LEARNING_RATE = 0.0002
    D_LEARNING_RATE = 0.0002
    BETA1 = 0.5
    BETA2 = 0.999
    L1_LAMBDA = 100

    # Data setup    
    split = int(len(sketch_photo_pair) * 0.8)
    

    G.cuda()
    D.cuda()
    G.train()
    D.train()

    # loss
    BCE_loss = nn.BCELoss().cuda()
    L1_loss = nn.L1Loss().cuda()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=G_LEARNING_RATE, betas=(BETA1, BETA2))
    D_optimizer = optim.Adam(D.parameters(), lr=D_LEARNING_RATE, betas=(BETA1, BETA2))

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    it = 0
    # do this for each category
    for pair in sketch_photo_pair:
        transformed_dataset = SketchPhotoDataset(sketch_photo_pair[pair])
        data_train = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

        print('training start!')
        start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            D_losses = []
            G_losses = []
            epoch_start_time = time.time()
            num_iter = 0

            for batch_idx, (x_, y_) in enumerate(tqdm(data_train, position=0, leave=True)):
            #for x_, y_ in data_train:
                # train discriminator D
                D.zero_grad()
                x_, y_ = x_.permute(0, 3, 1, 2), y_.permute(0, 3, 1, 2)  # rearrange to (batches, channels, height, width)
                x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())  # send to device
                x_, y_ = x_.float(), y_.float() # make x and y floats instead of bytes

                D_result = D(x_, y_).squeeze()
                D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))

                G_result = G(x_)
                D_result = D(x_, G_result).squeeze()
                D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).cuda()))

                D_train_loss = (D_real_loss + D_fake_loss) * 0.5
                D_train_loss.backward()
                D_optimizer.step()

                train_hist['D_losses'].append(D_train_loss.item())

                D_losses.append(D_train_loss.item())

              # train generator G
                G.zero_grad()

                G_result = G(x_)
                D_result = D(x_, G_result).squeeze()

                G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) + L1_LAMBDA * L1_loss(G_result, y_)
                G_train_loss.backward()
                G_optimizer.step()

                train_hist['G_losses'].append(G_train_loss.data.item())

                G_losses.append(G_train_loss.data.item())

                num_iter += 1

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
          
            # Saving the generator
            torch.save(G, '/data2/limill01/dl/sketchy-v2/models/cat/generator/gen' + str(epoch))

            # Saving the discriminator
            torch.save(D, '/data2/limill01/dl/sketchy-v2/models/cat/discriminator/disc' + str(epoch))

            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), NUM_EPOCHS, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                    torch.mean(torch.FloatTensor(G_losses))))
          # fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
          # util.show_result(G, Variable(fixed_x_.cuda(), volatile=True), fixed_y_, (epoch+1), save=True, path=fixed_p)
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        # Saving the generator
        torch.save(G, '/data2/limill01/dl/sketchy-v2/models/cat/generator/gen' + str(it))

        # Saving the discriminator
        torch.save(D, '/data2/limill01/dl/sketchy-v2/models/cat/discriminator/disc' + str(it))

        it += 1

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
photo_url = '/data2/limill01/dl/sketchy/photo'
sketch_url = '/data2/limill01/dl/sketchy/sketch'

sketch_photo_pair = ProcessData(photo_url, sketch_url)
split = int(len(sketch_photo_pair) * 0.8)

sketch_photo_pair_train = {}
sketch_photo_pair_test = {}

curr = 0
for key in sketch_photo_pair:
    if curr < split:
        sketch_photo_pair_train[key] = sketch_photo_pair[key]
    else:
        sketch_photo_pair_test[key] = sketch_photo_pair[key]
    curr += 1

# Params
batch_size = 16

# Instantiate themodel
G = cgan.generator()
D = cgan.discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
training(sketch_photo_pair_train, G, D, batch_size)
