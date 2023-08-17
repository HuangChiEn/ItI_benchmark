import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class Cloudy2SunnyWeatherDataset(Pix2pixDataset):

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

    def get_paths(self, opt):
        im_dict = {}
        im_path = Path(opt.croot) / 'train_images'
        for path in im_path.glob('*.jpg'):
            wea_cond = os.path.basename(path).split('_')[0]
            if not (wea_cond in im_dict.keys()):
                im_dict[wea_cond] = []
            im_dict[wea_cond].append(path)

        # make_dataset(.) return the list of image_path traversal all of sub-folders
        # we also feed list & sorted by filename to c_image_paths
        c_image_paths = sorted(im_dict['cloudy'])
        s_image_paths = sorted(im_dict['sunny'])

        # unchanged the default setup for TSIT dataset..
        if opt.phase == 'train':
            s_image_paths = s_image_paths + s_image_paths

        instance_paths = []

        length = min(len(c_image_paths), len(s_image_paths))
        c_image_paths = c_image_paths[:length]
        s_image_paths = s_image_paths[:length]
        return c_image_paths, s_image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True
