from cv2 import *
from Searching import *
args = make_parser().parse_args()
args.ckpt = '../YOLOX/assets/yolox_s.pth'
args.exp_file = '../YOLOX/exps/default/yolox_s.py'
args.path = "../YOLOX/assets/"
args.save_result = True
args.demo = 'image'
args.trt = True
#image = main(get_exp(args.exp_file), args)

# os.makedirs('test_image')

# cv2.imwrite('./test_image.jpg', image)

main_search(get_exp(args.exp_file), args, ["../YOLOX/assets/dog.jpg", "../YOLOX/assets/demo.jpg"])

# rgs: Namespace(demo='demo', experiment_name='yolox_s', name=None, path='test/frame_0.jpg', camid=0, save_result=False, exp_file='../YOLOX/exps/default/yolox_s.py', ckpt='../YOLOX/assets/yolox_s.pth', device='cpu', conf=0.3, nms=0.3, tsize=None, fp16=False, legacy=False, fuse=False, trt=False)
# Args: Namespace(demo='image', experiment_name='yolox_s', name=None, path='test/frame_9.jpg', camid=0, save_result=True, exp_file='../YOLOX/exps/default/yolox_s.py', ckpt='../YOLOX/assets/yolox_s.pth', device='[cpu/gpu]', conf=0.25, nms=0.45, tsize=640, fp16=False, legacy=False, fuse=False, trt=False)