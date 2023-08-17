import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset

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

class BDD100KDataset(Pix2pixDataset):

    # deeply code review required!!
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(aspect_ratio=1.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    # call BDD100KDataset.set_domains('clear', 'rainy') before calling get_paths!!
    def set_domains(self, c_domain='clear', s_domain='rainy'):
        self.c_domain = c_domain
        self.s_domain = s_domain

    def get_paths(self, opt, subset='train'):
        self.set_domains()
        
        self.dataroot = Path(opt.croot) 
        js_tmplt = f'bdd100k_labels_images_{subset}.json'
        self.js_path = Path(self.dataroot) / 'labels' / js_tmplt

        with self.js_path.open('r') as f_ptr:
            js_dict = json.load(f_ptr)

        im_dict = {}
        for im in js_dict:
            wea_cond = im['attributes']['weather']
            time = im['attributes']['timeofday']
            # weather condition in exclude
            if (wea_cond in _exclude_wea_cond) or (time in _exclude__time):
                continue
            if not (wea_cond in im_dict.keys()):
                im_dict[wea_cond] = []

            im_path = str(self.im_path / im["name"])
            im_dict[wea_cond].append(im_path)

        # make_dataset(.) return the list of image_path traversal all of sub-folders
        # we also feed list & sorted by filename to c_image_paths
        c_image_paths = sorted(im_dict[self.c_domain])
        s_image_paths = sorted(im_dict[self.s_domain])

        # unchanged the default setup for TSIT dataset..
        if opt.phase == 'train':
            s_image_paths = s_image_paths + s_image_paths

        # ret empty lst, set no_instance flag
        instance_paths = []

        length = min(len(c_image_paths), len(s_image_paths))
        c_image_paths = c_image_paths[:length]
        s_image_paths = s_image_paths[:length]

        #       content img,    style img,         0
        return c_image_paths, s_image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True
