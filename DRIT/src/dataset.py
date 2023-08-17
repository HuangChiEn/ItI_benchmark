import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
from pathlib import Path
from collections import defaultdict

_weather2id = {
        'rainy':0, 
        'snowy':1,
        'clear':2,
        'overcast':3,
        'foggy':4,
        'partly cloudy': 5,
        'undefined': 6
}

_exclude_wea_cond = [
    'undefined', 
    'partly cloudy',
    'foggy'
]

_exclude__time = [
    'undefined',
    'night'
]

## BDD dataset
class bdd_dataset_single(data.Dataset):
  def __init__(self, opts, input_dim):
    self.input_dim = input_dim
    subset = opts.phase

    ds_rt = Path(opts.dataroot)
    js_path = ds_rt / 'labels' / f'bdd100k_labels_images_{subset}.json'
    with js_path.open('r') as f_ptr:
      js_dict = json.load(f_ptr)

    ds_path_prefix = ds_rt / 'images/100k' / subset
    self.im_dict = defaultdict(list)
    for im in js_dict:
      wea_cond = im['attributes']['weather']
      time = im['attributes']['timeofday']
      # weather condition in exclude
      if (wea_cond in _exclude_wea_cond) or (time in _exclude__time):
        continue

      im_path = str(ds_path_prefix / im["name"])
      self.im_dict[wea_cond].append(im_path)

    # setup image transformation
    self.transforms = Compose([
      Resize((opts.resize_size, opts.resize_size), Image.BICUBIC),
      CenterCrop(opts.crop_size)),
      ToTensor(),
      Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    self.set_domain()
    return
  
  # public interface for user adjust the domain
  def set_domain(self, domain_name='clear'):
    self.img = self.im_dict[domain_name]
    self.size = len(self.img)
    print(f"set domain : {domain_name} ; images : {self.size}\n")

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size


class bdd_dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.input_dim = input_dim
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b
    subset = opts.phase

    ds_rt = Path(opts.dataroot)
    js_path = ds_rt / 'labels' / f'bdd100k_labels_images_{subset}.json'
    with js_path.open('r') as f_ptr:
      js_dict = json.load(f_ptr)

    ds_path_prefix = ds_rt / 'images/100k' / subset
    self.im_dict = defaultdict(list)
    for im in js_dict:
      wea_cond = im['attributes']['weather']
      time = im['attributes']['timeofday']
      # weather condition in exclude
      if (wea_cond in _exclude_wea_cond) or (time in _exclude__time):
        continue

      im_path = str(ds_path_prefix / im["name"])
      self.im_dict[wea_cond].append(im_path)

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)

    # default domain trfs.
    self.set_domains()
    return

  # public interface to setup the domain transfer pair
  def set_domains(self, domainA='clear', domainB='rainy'):
    # A
    self.A = self.im_dict[domainA]

    # B
    self.B = self.im_dict[domainB]

    self.A_size = len(self.A)
    self.B_size = len(self.B)

    print(f"set source domain : {domainA} ; images : {self.A_size}")
    print(f"set reference domain : {domainB} ; images : {self.B_size}\n")

  def __getitem__(self, index):
    data_A = self.load_img(self.A[index], self.input_dim_A)
    data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size


###########################################################
## Alps dataset

_season2id = {
    'spring':0, 
    'summer':1,
    'autumn':2, 
    'winter':3
}

class alps_dataset_single(data.Dataset):
  def __init__(self, opts, input_dim):
    self.input_dim = input_dim
    self.im_dict = defaultdict(list)

    ds_rt = Path(opts.dataroot) / opts.phase
    for sea_cond in os.listdir(ds_rt):
      sea_fold = ds_rt / sea_cond
      for im_files in os.listdir(sea_fold):
          self.im_dict[sea_cond].append( str(sea_fold / im_files) )
  
    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)

    # default domain trfs.
    self.set_domain()
    return

  def set_domain(self, domain_name='spring'):
    self.img = self.im_dict[domain_name]
    self.size = len(self.img)
    print(f"set domain : {domain_name} ; images : {self.size}\n")

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size

class alps_dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b
    self.im_dict = defaultdict(list)
    ds_rt = Path(opts.dataroot) / opts.phase
    for sea_cond in os.listdir(ds_rt):
      sea_fold = ds_rt / sea_cond
      for im_files in os.listdir(sea_fold):
          self.im_dict[sea_cond].append( str(sea_fold / im_files) )

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)

    # default domain trfs.
    self.set_domains()
    return

  # public interface to setup the domain transfer pair
  def set_domains(self, domainA='spring', domainB='summer'):
    # A
    self.A = self.im_dict[domainA]

    # B
    self.B = self.im_dict[domainB]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    
    print(f"set source domain : {domainA} ; images : {self.A_size}")
    print(f"set reference domain : {domainB} ; images : {self.B_size}\n")

  def __getitem__(self, index):
    data_A = self.load_img(self.A[index], self.input_dim_A)
    data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size


###########################################################
## CelebAHQ dataset
_gender2id = {
    'male':0, 
    'female':1
}

class celebahq_dataset_single(data.Dataset):
  def __init__(self, self, opts, input_dim):
    # Note that we ignore subset arg for competiblility
    self.im_dict = defaultdict(list)

    ds_rt = Path(opts.dataroot) / opts.phase 
    for gen_type in os.listdir(ds_rt):
      gen_fold = Path(ds_rt) / gen_type
      for im_files in os.listdir(gen_fold):
        self.im_dict[gen_type].append( str(gen_fold / im_files) )

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)

    # default domain trfs.
    self.set_domain()
    return

  def set_domain(self, domain_name='male'):
    self.img = self.im_dict[domain_name]
    self.size = len(self.img)
    print(f"set domain : {domain_name} ; images : {self.size}\n")

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size

class celebahq_dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b
    self.im_dict = defaultdict(list)
    ds_rt = Path(opts.dataroot) / opts.phase 
    for gen_type in os.listdir(ds_rt):
      gen_fold = Path(ds_rt) / gen_type
      for im_files in os.listdir(gen_fold):
        self.im_dict[gen_type].append( str(gen_fold / im_files) )

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)

    # default domain trfs.
    self.set_domains()
    return

  # public interface to setup the domain transfer pair
  def set_domains(self, domainA='male', domainB='female'):
    # A
    self.A = self.im_dict[domainA]

    # B
    self.B = self.im_dict[domainB]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    
    print(f"set source domain : {domainA} ; images : {self.A_size}")
    print(f"set reference domain : {domainB} ; images : {self.B_size}\n")

  def __getitem__(self, index):
    data_A = self.load_img(self.A[index], self.input_dim_A)
    data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size



###################################################################################
###################################################################################
## default dataset
class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot

    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size

## Weather dataset (disabled)

# The code repo bootstrap from yosemite dataset,
# In the data structure, subfolder A (summer), B (winter), 
# image path sample : trainA/2005-06-26 14_04_52.jpg
#                  -> f'{phase}{subset}/*.jpg'
class wea_dataset_single(data.Dataset):
  # we only replace the __init__ func, while use the glob to prepare the im_path
  # the other part is unchanged to compitetive with the original setup.. 
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    im_dict = {}
    im_path = Path(self.dataroot) / 'train_images'
    for path in im_path.glob('*.jpg'):
      wea_cond = os.path.basename(path).split('_')[0]
      if not (wea_cond in im_dict.keys()):
          im_dict[wea_cond] = []
      im_dict[wea_cond].append(path)

    self.img = im_dict[setname]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size

class wea_dataset_unpair(data.Dataset):
  # we only replace the __init__ func, while use the glob to prepare the im_path
  # the other part is unchanged to compitetive with the original setup.. 
  def __init__(self, opts):
    self.dataroot = opts.dataroot
    im_dict = {}
    im_path = Path(self.dataroot) / 'train_images'
    for path in im_path.glob('*.jpg'):
      wea_cond = os.path.basename(path).split('_')[0]
      if not (wea_cond in im_dict.keys()):
          im_dict[wea_cond] = []
      im_dict[wea_cond].append(path)
        
    # A
    self.A = im_dict['cloudy']

    # B
    self.B = im_dict['sunny']

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size
