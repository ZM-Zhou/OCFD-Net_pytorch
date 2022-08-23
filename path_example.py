class Path(object):
    @staticmethod
    def get_path_of(name):
        if name == "kitti":
            return '/zhouzm/Datasets/kitti'
        elif name == 'make3d':
            return '/zhouzm/Datasets/Make3D'
        elif name == 'cityscapes':
            return '/zhouzm/Datasets/NYU_v2/cityscapes'
        else:
            raise NotImplementedError