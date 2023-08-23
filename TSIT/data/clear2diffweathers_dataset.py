import os
from data.pix2pix_dataset import Pix2pixDataset


class Clear2DiffWeathersDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--test_mode', type=str, default='all',
                            help='specify style mode to control multi-modal image synthesis (MMIS) during test phase:'
                                 'overcast | rainy | snowy | all')
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(aspect_ratio=2.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser


    def get_paths(self, opt):
        # path modification
        tmp_rt = './datasets' if 'data1' in opt.croot else opt.croot
        croot, sroot = opt.croot, opt.sroot
        
        with open(os.path.join(tmp_rt, 'bdd100k_lists/clear2diffweathers/clear_%s.txt' % opt.phase)) as c_list:
            c_image_paths_read = c_list.read().splitlines()
            c_image_paths = [os.path.join(croot, p) for p in c_image_paths_read if p != '']

        if opt.phase == 'train' or opt.test_mode == 'all':
            mode_list = ['overcast', 'rainy', 'snowy']
        else:
            mode_list = [opt.test_mode]
        s_image_paths = []
        for mode in mode_list:
            with open(os.path.join(tmp_rt, 'bdd100k_lists/clear2diffweathers/%s_%s.txt' % (mode, opt.phase))) as s_list:
                s_image_paths_read = s_list.read().splitlines()
                s_image_paths_mode = [os.path.join(sroot, p) for p in s_image_paths_read if p != '']
            s_image_paths.extend(s_image_paths_mode)

        while len(s_image_paths) < len(c_image_paths):
            s_image_paths = s_image_paths + s_image_paths

        instance_paths = []

        # while + min(len(.)) -> ( length == c_image_paths, since 'clear' have 14458 samples )
        length = min(len(c_image_paths), len(s_image_paths))
        c_image_paths = c_image_paths[:length]
        s_image_paths = s_image_paths[:length]

        #       source img    reference img   no_inst, nil
        return c_image_paths, s_image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True