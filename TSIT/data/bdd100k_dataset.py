import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from pathlib import Path
import json

_weather2id = {
    'rainy':0,     # 2850
    'snowy':1,     # 3298
    'clear':2,     # 14458
    'overcast':3,  # 8698

    'foggy':4,
    'partly cloudy': 5,
    'undefined': 6
}

_exclude_wea_cond = [
    'foggy',
    'partly cloudy',
    'undefined'
]

_exclude__time = [
    'undefined',
    'night'
]

# The way to arrange the training data will follow the official script : 
# /data/joseph/wea_trfs_benchmark/TSIT/data/sunny2diffweathers_dataset.py
class BDD100KDataset(Pix2pixDataset):
    # deeply code review required!!
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        # --test_mode, doesn't support, we use ast-style! 
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(aspect_ratio=2.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt, subset='val'):  # train
        def record_path(wea, wea_lst): 
            file_path = []
            for raw_path in wea_lst:
                path_lst = raw_path.split(os.sep)
                idx = path_lst.index('images')
                file_path.append( os.sep.join(path_lst[idx:]) + '\n' )  # '\n' for control write line

            with open(f'./datasets/bdd100k_lists/clear2diffweathers/{wea}_test.txt', 'w+') as f_ptr:
                f_ptr.writelines(file_path)

        # setup domains
        self.c_domain = opt.c_domain
        self.s_domain = opt.s_domain
        
        self.dataroot = Path(opt.croot) 
        js_tmplt = f'bdd100k_labels_images_{subset}.json'
        self.js_path = self.dataroot / 'labels' / js_tmplt

        with self.js_path.open('r') as f_ptr:
            js_dict = json.load(f_ptr)

        im_dict = {}
        ds_path_prefix = self.dataroot / 'images/100k' / subset
        for im in js_dict:
            wea_cond = im['attributes']['weather']
            time = im['attributes']['timeofday']
            # weather condition in exclude
            if (wea_cond in _exclude_wea_cond) or (time in _exclude__time):
                continue
            if not (wea_cond in im_dict.keys()):
                im_dict[wea_cond] = []

            im_path = str(ds_path_prefix / im["name"])
            im_dict[wea_cond].append(im_path)

        # AOP get src path
        #for wea in im_dict.keys():
        #    record_path(wea, im_dict[wea])
        #breakpoint()
        
        # make_dataset(.) return the list of image_path traversal all of sub-folders
        # we also feed list & sorted by filename to c_image_paths
        c_image_paths = sorted(im_dict[self.c_domain])
        s_image_paths = sorted(im_dict[self.s_domain])

        # bdd ds, clear have 2~3 times of sample then the other categories,
        # we apply the same way to increase the style image set
        while len(s_image_paths) < len(c_image_paths):
            s_image_paths = s_image_paths + s_image_paths

        # ret empty lst, set no_instance flag
        instance_paths = []

        # alignment the num of sample of both set
        length = min(len(c_image_paths), len(s_image_paths))
        c_image_paths = c_image_paths[:length]
        s_image_paths = s_image_paths[:length]

        #       content img,    style img,         0
        return c_image_paths, s_image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True
