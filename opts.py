import argparse


def parse_opts():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str,
						default='uf101', help='dataset type')
	parser.add_argument(
		'--root_path',
		default='/root/data/ActivityNet',
		type=str,
		help='Root directory path of data')
	parser.add_argument(
		'--video_path',
		default='video_kinetics_jpg',
		type=str,
		help='Directory path of Videos')
	parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
	parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')
	parser.add_argument(
		'--annotation_path',
		default='kinetics.json',
		type=str,
		help='Annotation file path')
	parser.add_argument(
		'--gpu',
		default=0,
		type=int)
	parser.add_argument(
		'--sample_size',
		default=150,
		type=int,
		help='Height and width of inputs')
	parser.add_argument(
		'--log_interval',
		default=10,
		type=int,
		help='Log interval for showing training loss')
	parser.add_argument(
		'--save_interval',
		default=2,
		type=int,
		help='Model saving interval')
	parser.add_argument(
        '--model',
        default='cnnlstm',
        type=str,
        help=
        '(cnnlstm | cnnlstm_attn |')
	parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)'
    )
	parser.add_argument(
		'--lr_rate',
		default=1e-3,
		type=float,
		help='Initial learning rate (divided by 10 while training by lr scheduler)')
	parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
	parser.add_argument(
		'--dampening', default=0.9, type=float, help='dampening of SGD')
	parser.add_argument(
		'--weight_decay', default=1e-3, type=float, help='Weight Decay')
	parser.add_argument(
		'--no_mean_norm',
		action='store_true',
		help='If true, inputs are not normalized by mean.')
	parser.set_defaults(no_mean_norm=False)
	parser.add_argument(
        '--mean_dataset',
        default='activitynet',
        type=str,
        help=
        'dataset for mean values of mean subtraction (activitynet | kinetics)')
	parser.add_argument(
		'--use_cuda',
		action='store_true',
		help='If true, use GPU.')
	parser.set_defaults(std_norm=False)
	parser.add_argument(
		'--nesterov', action='store_true', help='Nesterov momentum')
	parser.set_defaults(nesterov=False)
	parser.add_argument(
		'--optimizer',
		default='sgd',
		type=str,
		help='Currently only support SGD')
	parser.add_argument(
		'--lr_patience',
		default=10,
		type=int,
		help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
	)
	parser.add_argument(
		'--batch_size', default=32, type=int, help='Batch Size')
	parser.add_argument(
		'--n_epochs',
		default=10,
		type=int,
		help='Number of total epochs to run')
	parser.add_argument(
		'--start_epoch',
		default=1,
		type=int,
		help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
	)
	parser.add_argument(
		'--resume_path',
		type=str,
		help='Resume training')
	parser.add_argument(
		'--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
	parser.add_argument(
		'--num_workers',
		default=4,
		type=int,
		help='Number of threads for multi-thread loading')
	parser.add_argument(
		'--norm_value',
		default=1,
		type=int,
		help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
	parser.add_argument(
		'--std_norm',
		action='store_true',
		help='If true, inputs are normalized by standard deviation.')

	args = parser.parse_args()

	return args