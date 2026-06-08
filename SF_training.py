import numpy as np
import os
import glob
import h5py
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense, Reshape, Dropout, Add, ReLU, GlobalAveragePooling2D, Multiply, Lambda, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATH CONFIGURATION — edit only this section
# ============================================================
TRAINSET_DIR = '/data/dust/user/zhaiyufe/TrainSet/ML_GISAXS_Yuxin'   # root of all *_aux.tfrecord files
RESULTS_DIR  = '/data/dust/user/zhaiyufe/Models/ML_GISAXS_Yuxin/Results_SF_4paras'  # all outputs go here
MODEL_NAME   = 'yuxin_sf_model_4para'                                 # base name for all output files
MASK_PATH    = 'mask.npy'                                          # mask file path

# Derived paths (no need to edit below)
os.makedirs(RESULTS_DIR, exist_ok=True)
PATH_MODEL       = os.path.join(RESULTS_DIR, f'{MODEL_NAME}.keras')
PATH_CHECKPOINT  = os.path.join(RESULTS_DIR, f'{MODEL_NAME}.checkpoint.keras')
PATH_LOSS_FIG    = os.path.join(RESULTS_DIR, f'{MODEL_NAME}_training_loss.png')
PATH_LOSS_CSV    = os.path.join(RESULTS_DIR, f'{MODEL_NAME}_loss_history.csv')
PATH_PRED_FIG    = os.path.join(RESULTS_DIR, f'{MODEL_NAME}_predictions.png')
PATH_PRED_DIR    = os.path.join(RESULTS_DIR, 'epoch_comparisons')
os.makedirs(PATH_PRED_DIR, exist_ok=True)
# ============================================================

parser = argparse.ArgumentParser(description='Train Yuxin Structure Factor Model')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--input_mode', choices=['preprocessed', 'clean'], default='preprocessed', help='Interpret TFRecord inputs as already-preprocessed images or clean images that need online preprocessing')
parser.add_argument('--noise_mode', choices=['none', 'gaussian', 'poisson'], default='none', help='Training-time noise mode used only with --input_mode clean')
parser.add_argument('--snr_min', type=float, default=80.0, help='Lower SNR bound for gaussian training noise')
parser.add_argument('--snr_max', type=float, default=110.0, help='Upper SNR bound for gaussian training noise')
parser.add_argument('--poisson_scale_min', type=float, default=1.0, help='Lower poisson scale used for training-time noise')
parser.add_argument('--poisson_scale_max', type=float, default=20.0, help='Upper poisson scale used for training-time noise')
parser.add_argument('--mask_path', type=str, default=MASK_PATH, help='Mask file path used for the mask channel and optional masking of the image channel')
parser.add_argument('--apply_mask', action='store_true', help='Apply the fixed mask to the image channel when using clean inputs')
parser.add_argument('--apply_random_edge_mask', action='store_true', help='Apply random edge masking during training when using clean inputs')
parser.add_argument('--edge_mask_max_width', type=int, default=20, help='Maximum random edge-mask width in pixels')
parser.add_argument('--random_mask_points_min', type=int, default=5, help='Minimum number of random masked pixels added during training')
parser.add_argument('--random_mask_points_max', type=int, default=10, help='Maximum number of random masked pixels added during training')
parser.add_argument('--log_eps', type=float, default=1e-8, help='Numerical epsilon used in online log-normalization')
parser.add_argument('--comparison_interval', type=int, default=5, help='Save predicted vs ground-truth comparison plots every N epochs')
parser.add_argument('--comparison_samples', type=int, default=4, help='Number of validation samples used for comparison plots')
args = parser.parse_args()

print('Training Configuration:')
print(f'- Epochs: {args.epochs}')
print(f'- Batch size: {args.batch_size}')
print(f'- Learning rate: {args.learning_rate}')
print(f'- Input mode: {args.input_mode}')
if args.input_mode == 'clean':
	print(f'- Noise mode: {args.noise_mode}')
	print(f'- Apply fixed mask: {args.apply_mask}')
	print(f'- Apply random edge mask: {args.apply_random_edge_mask}')
print()

TARGET_PARAM_NAMES = ['t_Cu', 't_polymer', 'D', 'sigma']
TARGET_MIN = np.array([0.0, 10.0, 4.0, 0.2], dtype=np.float32)
TARGET_MAX = np.array([25.0, 50.0, 20.0, 4.0], dtype=np.float32)
TARGET_MIN_TF = tf.constant(TARGET_MIN, dtype=tf.float32)
TARGET_MAX_TF = tf.constant(TARGET_MAX, dtype=tf.float32)


def normalize_targets_tf(targets):
	targets = tf.cast(targets, tf.float32)
	return (targets - TARGET_MIN_TF) / (TARGET_MAX_TF - TARGET_MIN_TF)


def denormalize_targets_np(targets):
	targets = np.asarray(targets, dtype=np.float32)
	return targets * (TARGET_MAX - TARGET_MIN) + TARGET_MIN

# def custom_attention(x):
#     """
#     自定义注意力机制：为输入特征图生成一个权重矩阵，横向从左到右先增加到中间，再减小，
#     并加入一个从左到右递增的指数项。
#     """
#     # 动态获取输入特征图的形状
#     shape = tf.shape(x)  # 动态获取形状
#     batch_size, h, w, c = shape[0], shape[1], shape[2], shape[3]

#     # 创建一个对称的权重矩阵，横向从左到右先增加到中间，再减小
#     left = tf.linspace(0.0, 1.0, w // 2)  # 左半部分权重从 0 增加到 1
#     right = tf.linspace(1.0, 0.0, w // 2)  # 右半部分权重从 1 减小到 0
#     weight_row = tf.concat([left, right], axis=0)  # 合并左右部分

#     # 加入从左到右递增的指数项
#     exp_weight = tf.exp(tf.linspace(-7.0, 0.0, w))  # 指数项从 1 到 e^2
#     weight_row = weight_row * exp_weight  # 将对称权重与指数项相乘

#     # 将权重矩阵扩展到整个高度
#     weight_matrix = tf.tile(tf.expand_dims(weight_row, axis=0), [h, 1])  # 将权重矩阵扩展到整个高度
#     weight_matrix = tf.expand_dims(weight_matrix, axis=-1)  # 添加通道维度

#     # 将权重矩阵扩展到批量维度，并广播到与 x 的形状匹配
#     weight_matrix = tf.expand_dims(weight_matrix, axis=0)  # 添加批量维度
#     weight_matrix = tf.tile(weight_matrix, [batch_size, 1, 1, c])  # 广播到与 x 的形状一致

#     # 将输入特征与权重矩阵相乘
#     x = Multiply()([x, weight_matrix])
#     return x
def custom_attention(x):
	"""
	自定义注意力：横向倒V+指数衰减，再把指定矩形区强制 attention=1。
	"""
	shape = tf.shape(x)
	batch, h, w, c = shape[0], shape[1], shape[2], shape[3]

	# 1) 横向倒V+衰减
	eps = 1e-5
	half = w // 2
	left  = tf.linspace(0.0, eps, half)
	right = tf.linspace(eps, 0.0, w - half)
	ramp  = tf.concat([left, right], axis=0)              # [w]
	decay = tf.exp(tf.linspace(-7.0, 0.0, w))             # [w]
	row   = ramp * decay                                 # [w]
	row   = tf.reshape(row, [1,1,w,1])
	weights = tf.tile(row, [batch, h, 1, c])             # [B,H,W,C]

	# 2) 布尔掩码 via 比较
	# rectangles = list of (r0,c0,r1,c1)
	rectangles = [(200,50,240,200)]
	# make 2D grid
	rows = tf.range(h)[:,None]      # [H,1]
	cols = tf.range(w)[None,:]      # [1,W]
	mask2d = tf.zeros([h,w], tf.bool)
	for (r0,c0,r1,c1) in rectangles:
		mask_rect = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
		mask2d = mask2d | mask_rect

	mask4d = tf.reshape(mask2d, [1,h,w,1])
	mask4d = tf.tile(mask4d, [batch,1,1,c])

	# 3) 在掩码区域强制 weight=1
	weights = tf.where(mask4d, tf.ones_like(weights), weights)

	return Multiply()([x, weights])

# snr = int(sys.argv[1])
# Read the TFRecord Files

# Collect all *_aux.tfrecord files from all subdirectories
file_list = sorted(glob.glob(os.path.join(TRAINSET_DIR, '**', '*_aux.tfrecord'), recursive=True))
print(f"Found {len(file_list)} aux TFRecord files.")
if not file_list:
    raise RuntimeError("No *_aux.tfrecord files found. Check TRAINSET_DIR.")


def has_readable_record(path):
	"""Fast sanity check: file can be opened and at least one record can be read."""
	try:
		for _ in tf.data.TFRecordDataset(path).take(1):
			return True
	except Exception:
		return False
	return False


good_files = [p for p in file_list if has_readable_record(p)]
bad_files = [p for p in file_list if p not in good_files]
if bad_files:
	print("Skipped unreadable TFRecord files:")
	for bad_path in bad_files:
		print(f"  - {bad_path}")
if not good_files:
	raise RuntimeError("No readable *_aux.tfrecord files found after validation.")

# Split by files to avoid counting every record (expensive and memory-heavy)
if len(good_files) == 1:
	train_files = good_files
	val_files = good_files
else:
	split_idx = max(1, int(0.8 * len(good_files)))
	if split_idx >= len(good_files):
		split_idx = len(good_files) - 1
	train_files = good_files[:split_idx]
	val_files = good_files[split_idx:]

print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")


def infer_shapes_from_tfrecord(filename):
	"""Read the first record to infer input/output shapes automatically."""
	if isinstance(filename, (list, tuple)):
		filename = filename[0]
	input_len = None
	output_len = None
	for raw in tf.data.TFRecordDataset(filename).take(1):
		example = tf.train.Example()
		example.ParseFromString(raw.numpy())
		input_len = len(example.features.feature['input'].float_list.value)
		output_len = len(example.features.feature['output'].float_list.value)
	if input_len is None or output_len is None:
		raise RuntimeError(f"Could not infer shapes from empty TFRecord: {filename}")
	input_side = int(round(input_len ** 0.5))
	input_shape = (input_side, input_side, 1)
	output_shape = (output_len,)
	print(f"[infer_shapes] input: {input_shape}, output: {output_shape}")
	return input_shape, output_shape

input_data_shape, output_data_shape = infer_shapes_from_tfrecord(train_files)


def load_mask_tensors(mask_path, image_shape):
	if not mask_path:
		raise ValueError('mask_path is required for SF training because the model expects a mask channel')
	if not os.path.exists(mask_path):
		raise FileNotFoundError(f'Mask file not found: {mask_path}')
	if mask_path.endswith('.npz'):
		npz = np.load(mask_path)
		mask = np.array(npz['mask']) if 'mask' in npz else np.array(npz[npz.files[0]])
	else:
		mask = np.load(mask_path)
	if mask.ndim == 2:
		mask = mask[..., None]
	if mask.shape != image_shape:
		raise ValueError(f'Mask shape {mask.shape} does not match input shape {image_shape}')
	mask_float = mask.astype(np.float32)
	mask_bool = mask.astype(bool)
	return tf.constant(mask_float, dtype=tf.float32), tf.constant(mask_bool)


_fixed_mask_tf, _fixed_mask_bool_tf = load_mask_tensors(args.mask_path, input_data_shape)


def add_gaussian_noise_tf(image):
	snr = tf.random.uniform([], minval=args.snr_min, maxval=args.snr_max, dtype=tf.float32)
	signal_power = tf.reduce_mean(tf.square(image))
	snr_linear = tf.pow(tf.constant(10.0, dtype=tf.float32), snr / 10.0)
	noise_power = tf.math.divide_no_nan(signal_power, tf.maximum(snr_linear, tf.constant(1e-6, dtype=tf.float32)))
	noise = tf.random.normal(tf.shape(image), stddev=tf.sqrt(noise_power), dtype=image.dtype)
	return tf.maximum(image + noise, 0.0)


def add_poisson_noise_tf(image):
	scale = tf.random.uniform([], minval=args.poisson_scale_min, maxval=args.poisson_scale_max, dtype=tf.float32)
	clipped = tf.maximum(image, 0.0)
	noisy = tf.random.poisson([], clipped * scale, dtype=tf.float32)
	return tf.maximum(noisy / scale, 0.0)


def log_normalize_tf(image):
	max_val = tf.reduce_max(image)
	scale = tf.math.divide_no_nan(tf.constant(np.e, dtype=image.dtype), max_val + tf.constant(args.log_eps, dtype=image.dtype))
	scaled = image * scale
	logged = tf.math.log(scaled + tf.constant(args.log_eps, dtype=image.dtype))
	return tf.where(tf.math.is_finite(logged), logged, tf.fill(tf.shape(logged), tf.constant(-1.0, dtype=image.dtype)))


def apply_fixed_mask_tf(image):
	return tf.where(_fixed_mask_bool_tf, tf.fill(tf.shape(image), tf.constant(-1.0, dtype=image.dtype)), image)


def apply_random_edge_mask_tf(image):
	max_width = max(0, int(args.edge_mask_max_width))
	widths = tf.random.uniform([4], minval=0, maxval=max_width + 1, dtype=tf.int32)
	height = tf.shape(image)[0]
	width = tf.shape(image)[1]
	rows = tf.range(height)[:, None]
	cols = tf.range(width)[None, :]
	edge_mask = (
		(rows < widths[0])
		| (rows >= height - widths[1])
		| (cols < widths[2])
		| (cols >= width - widths[3])
	)
	edge_mask = edge_mask[..., None]
	return tf.where(edge_mask, tf.fill(tf.shape(image), tf.constant(-1.0, dtype=image.dtype)), image)


def apply_random_point_mask_tf(image):
	min_points = max(0, int(args.random_mask_points_min))
	max_points = max(min_points, int(args.random_mask_points_max))
	num_points = tf.random.uniform([], minval=min_points, maxval=max_points + 1, dtype=tf.int32)
	height = tf.shape(image)[0]
	width = tf.shape(image)[1]
	rows = tf.random.uniform([num_points], minval=0, maxval=height, dtype=tf.int32)
	cols = tf.random.uniform([num_points], minval=0, maxval=width, dtype=tf.int32)
	indices = tf.stack([rows, cols, tf.zeros_like(rows)], axis=1)
	updates = tf.fill([num_points], tf.constant(-1.0, dtype=image.dtype))
	return tf.tensor_scatter_nd_update(image, indices, updates)


def make_clean_input_preprocessor(training):
	def preprocess_record(record):
		image = tf.cast(record['input'], tf.float32)
		if training and args.noise_mode == 'gaussian':
			image = add_gaussian_noise_tf(image)
		elif training and args.noise_mode == 'poisson':
			image = add_poisson_noise_tf(image)
		image = log_normalize_tf(image)
		if args.apply_mask:
			image = apply_fixed_mask_tf(image)
		if training and args.apply_random_edge_mask:
			image = apply_random_point_mask_tf(image)
			image = apply_random_edge_mask_tf(image)
		record['input'] = image
		return record

	return preprocess_record

# def parse_example(example_proto):
#     feature_description = {
#         'input': tf.io.FixedLenFeature(input_data_shape, tf.float32),
#         'output': tf.io.FixedLenFeature(output_data_shape, tf.float32)
#     }
	# return tf.io.parse_single_example(example_proto, feature_description)

# def parse_example(example_proto):

# 	feature_description = {
# 		'input': tf.io.FixedLenFeature(input_data_shape, tf.float32),
# 		'output': tf.io.FixedLenFeature(output_data_shape, tf.float32)
# 	}
# 	parsed_example = tf.io.parse_single_example(example_proto, feature_description)
	
# 	# parsed_example['input'] = input_data

# 	# Use only the first two elements of 'output'
# 	parsed_example['output'] = parsed_example['output'][:2]
	
# 	return parsed_example

def parse_example(example_proto):
	feature_description = {
		'input': tf.io.FixedLenFeature(input_data_shape, tf.float32),
		'output': tf.io.FixedLenFeature(output_data_shape, tf.float32)
	}
	parsed_example = tf.io.parse_single_example(example_proto, feature_description)

	# 4-parameter SF model: keep only [t_Cu, t_polymer, D, sigma].
	parsed_example['output'] = parsed_example['output'][:4]
	return parsed_example

def filter_example(parsed_example):
	# 过滤逻辑：跳过 output[0] > 15 的数据
	return tf.less_equal(parsed_example['output'][0], 30.0)


def filter_valid_output(parsed_example):
	output = parsed_example['output'][:4]
	return tf.reduce_all(output >= 0.0)

def add_mask_channel(record):
	"""
	在每个样本后面拼接一个 mask 通道：
	 - record['input'] 原本是 (H,W,1)
	 - _fixed_mask_tf 是 (H,W,1)
	 => 拼接后变成 (H,W,2)
	"""
	x = record['input']                  # (H,W,1)
	# mask 通道直接复用固定 mask
	mask_ch = _fixed_mask_tf             # (H,W,1)
	record['input'] = tf.concat([x, mask_ch], axis=-1)  # (H,W,2)
	return record

def random_mask_columns(record):
	"""
	对 record['input'] 的前 n 列置 -1，
	n 在 [0, 50] 内随机选取。
	"""
	x = record['input']                      # x.shape == (H, W, 1)
	H = tf.shape(x)[0]
	W = tf.shape(x)[1]
	
	# 随机选一个 n ∈ [0, 50]
	n = tf.random.uniform([], minval=30, maxval=51, dtype=tf.int32)
	
	# 构造两部分：前 n 列全 -1，后面保留原值
	left  = tf.fill([H, n, 1], -1.0)
	right = x[:, n:, :]                       # shape == (H, W-n, 1)
	
	# 拼回去
	x_masked = tf.concat([left, right], axis=1)
	
	record['input'] = x_masked
	return record

def replace_nan_with_zero(parsed_record):
	input_data = parsed_record['input']
	output_data = parsed_record['output']
	
	# Replace NaN values with 0
	input_data = tf.where(tf.math.is_nan(input_data), tf.fill(tf.shape(input_data), tf.constant(float('1e-10'), dtype=input_data.dtype)), input_data)
	output_data = tf.where(tf.math.is_nan(output_data), tf.fill(tf.shape(output_data), tf.constant(float('1e-10'), dtype=output_data.dtype)), output_data)
	
	# # Set middle vertical pixels to zero
	# indices = tf.constant([[i, j, 0] for i in range(128) for j in range(58, 70)], dtype=tf.int32)
	# updates = tf.zeros([128 * 12], dtype=input_data.dtype)
	# input_data = tf.tensor_scatter_nd_update(input_data, indices, updates)

	parsed_record['input'] = input_data
	parsed_record['output'] = output_data
	return parsed_record


def normalize_output_record(parsed_record):
	parsed_record['output'] = normalize_targets_tf(parsed_record['output'])
	return parsed_record

def load_dataset(filename, training=False):
	raw_dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=tf.data.AUTOTUNE)
	raw_dataset = raw_dataset.ignore_errors()
	parsed_dataset = raw_dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
	parsed_dataset = parsed_dataset.filter(filter_valid_output)
	parsed_dataset = parsed_dataset.map(replace_nan_with_zero, num_parallel_calls=tf.data.AUTOTUNE)
	parsed_dataset = parsed_dataset.map(normalize_output_record, num_parallel_calls=tf.data.AUTOTUNE)
	if args.input_mode == 'clean':
		parsed_dataset = parsed_dataset.map(make_clean_input_preprocessor(training=training), num_parallel_calls=tf.data.AUTOTUNE)
	# parsed_dataset = parsed_dataset.map(random_mask_columns)
	parsed_dataset = parsed_dataset.map(add_mask_channel, num_parallel_calls=tf.data.AUTOTUNE)
	# normalized_dataset = parsed_dataset.map(normalize_output_elements)
	return parsed_dataset

# file_list = ['/data/dust/user/zhaiyufe/TrainSet/Jungui_Build/Au_sf_snr80to110_noRough_01.tfrecord',
# 			 '/data/dust/user/zhaiyufe/TrainSet/Jungui_Build/Au_sf_snr80to110_noRough_02.tfrecord',
# 			 '/data/dust/user/zhaiyufe/TrainSet/Jungui_Build/Au_sf_snr80to110_noRough_03.tfrecord']


train_dataset = load_dataset(train_files, training=True)
val_dataset = load_dataset(val_files, training=False)


for data in train_dataset.take(1):
	print(data['input'].shape)
	print(data['output'].shape)

for data in val_dataset.take(1):
	print(data['input'].shape)
	print(data['output'].shape)




# model = tf.keras.Sequential([
# 	tf.keras.layers.InputLayer(input_shape=(256, 256, 1), name='input'),
# 	tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.Activation('relu'),
# 	tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.Activation('relu'),
# 	tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.Activation('relu'),
# 	tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.Activation('relu'),
# 	tf.keras.layers.Flatten(),
# 	tf.keras.layers.Dense(2048, activation='relu'),
# 	# tf.keras.layers.Dropout(0.5),
# 	tf.keras.layers.Dense(30 * 30, activation='relu'),
# 	tf.keras.layers.Dense(2, name='output')
# ])

# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(256, 256, 1), name='input'),
	
#     tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Activation('relu'),
	
#     tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.Dropout(0.2),
	
#     tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.Dropout(0.2),
	
#     tf.keras.layers.Flatten(),
	
#     tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
#     tf.keras.layers.Dropout(0.5),
	
#     # For structure factor extraction, we focus on regression of parameters such as D and sigma.
#     tf.keras.layers.Dense(2, name='output')
# ])


def prepare_for_training(dataset, batch_size=128, shuffle_buffer_size=1000):
	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	return dataset


train_dataset = prepare_for_training(train_dataset, batch_size=args.batch_size)
val_dataset = prepare_for_training(val_dataset, batch_size=args.batch_size)

def split_input_output(data):
	return data['input'], data['output']


# Two output heads:
# branch_thickness: y[0:2] = [t_Cu, t_polymer]
# branch_size:      y[2:4] = [D, sigma]
def to_two_branch_labels(x, y):
	labels = {
		'branch_thickness': y[..., 0:2],
		'branch_size':      y[..., 2:4],
	}
	return x, labels

train_dataset = train_dataset.map(split_input_output).map(to_two_branch_labels)
val_dataset   = val_dataset.map(split_input_output).map(to_two_branch_labels)


def collect_preview_samples(dataset, sample_count):
	preview_dataset = dataset.unbatch().take(sample_count)
	return list(preview_dataset.as_numpy_iterator())


def flatten_sf_outputs(prediction_dict):
	return np.concatenate([
		prediction_dict['branch_thickness'],
		prediction_dict['branch_size'],
	], axis=-1)


def flatten_sf_labels(label_dict):
	return np.concatenate([
		label_dict['branch_thickness'],
		label_dict['branch_size'],
	], axis=-1)


def save_prediction_comparison_figure(model, preview_samples, save_path, title=None):
	if not preview_samples:
		print('No preview samples available; skipping comparison figure.')
		return

	param_names = TARGET_PARAM_NAMES
	n_samples = len(preview_samples)
	fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4 * n_samples), squeeze=False)

	for i, (input_image, label_dict) in enumerate(preview_samples):
		input_batch = np.asarray(input_image).reshape(1, *input_image.shape)
		prediction_dict = model.predict(input_batch, verbose=0)
		predicted = denormalize_targets_np(flatten_sf_outputs(prediction_dict))[0]
		ground_truth = denormalize_targets_np(flatten_sf_labels(label_dict))

		axes[i, 0].imshow(np.asarray(input_image)[..., 0], cmap='hot')
		axes[i, 0].set_title(f'Input #{i}')
		axes[i, 0].axis('off')

		x = np.arange(len(param_names))
		width = 0.38
		axes[i, 1].bar(x - width / 2, ground_truth, width=width, label='Ground Truth')
		axes[i, 1].bar(x + width / 2, predicted, width=width, label='Predicted')
		axes[i, 1].set_xticks(x)
		axes[i, 1].set_xticklabels(param_names, rotation=30, ha='right')
		axes[i, 1].set_title(f'Parameters #{i}')
		axes[i, 1].legend()

	if title:
		fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(save_path)
	plt.close(fig)


class PredictionComparisonCallback(tf.keras.callbacks.Callback):
	def __init__(self, preview_samples, output_dir, interval):
		super().__init__()
		self.preview_samples = preview_samples
		self.output_dir = output_dir
		self.interval = max(1, int(interval))

	def on_epoch_end(self, epoch, logs=None):
		epoch_num = epoch + 1
		if epoch_num % self.interval != 0:
			return
		save_path = os.path.join(self.output_dir, f'epoch_{epoch_num:03d}.png')
		save_prediction_comparison_figure(
			self.model,
			self.preview_samples,
			save_path,
			title=f'Predicted vs Ground Truth - Epoch {epoch_num}',
		)


def append_loss_history(loss_file_path, history):
	if os.path.exists(loss_file_path):
		try:
			df = pd.read_csv(loss_file_path)
			last_epoch = int(df['epoch'].max()) if not df.empty else 0
		except Exception as exc:
			print(f"Error reading loss history file: {exc}")
			last_epoch = 0
	else:
		last_epoch = 0

	with open(loss_file_path, mode='a', newline='') as file:
		writer = csv.writer(file)
		if last_epoch == 0:
			writer.writerow(['epoch', 'loss', 'val_loss'])

		for index, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss']), start=1):
			writer.writerow([last_epoch + index, loss, val_loss])


def plot_loss_from_csv(loss_file_path, figure_path, title):
	if not os.path.exists(loss_file_path):
		print(f'Loss history file not found: {loss_file_path}')
		return

	df = pd.read_csv(loss_file_path)
	if df.empty:
		print(f'Loss history file is empty: {loss_file_path}')
		return

	plt.figure(figsize=(8, 5))
	plt.plot(df['epoch'], df['loss'], label='Training Loss')
	plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
	plt.title(title)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.tight_layout()
	plt.savefig(figure_path)
	plt.close()
preview_samples = collect_preview_samples(val_dataset, args.comparison_samples)



from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model



# Set up ReduceLROnPlateau callback, 
reduce_lr_callback = ReduceLROnPlateau(
	monitor='val_loss',  # Monitor validation loss
	factor=0.2,          # Factor by which the learning rate will be reduced
	patience=5,          # Number of epochs with no improvement after which learning rate will be reduced
	min_delta=1e-10,    # Threshold for measuring the new optimum
	min_lr=0.0000001        # Lower bound on the learning rate
)

# Set up ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
	filepath=PATH_CHECKPOINT,
	save_freq='epoch',  # Save every epoch
	monitor='val_loss',  # Monitor validation loss
	mode='min',
	save_best_only=True
)

# model.compile(optimizer= optimizer, loss='mse', metrics=['mse'])


strategy = tf.distribute.MirroredStrategy()
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

def res_block(x, mask_tensor, filters):
	shortcut = x
	y = Conv2D(filters, 3, padding='same', activation=None)(x)
	y = BatchNormalization()(y)
	y = ReLU()(y)
	y = Conv2D(filters, 3, padding='same', activation=None)(y)
	y = BatchNormalization()(y)
	y = Multiply()([y, Lambda(lambda m: 1.0 - m)(mask_tensor)])
	y = Add()([shortcut, y])
	y = ReLU()(y)
	return y

def gated_conv_block(x, mask_tensor, filters):
	x = Multiply()([x, Lambda(lambda m: 1.0 - m)(mask_tensor)])
	x = Conv2D(filters, 3, strides=2, padding='same', activation=None)(x)
	x = BatchNormalization()(x)
	x = ReLU()(x)
	mask_tensor = MaxPooling2D(pool_size=2, padding='same')(mask_tensor)
	return x, mask_tensor

with strategy.scope():
	optimizer = Adam(learning_rate=args.learning_rate)

	inp = Input(shape=(256,256,2), name='img_and_mask')
	img  = inp[:, :, :, 0:1]
	mask = inp[:, :, :, 1:2]

	# Block 1: 256→128
	x, mask = gated_conv_block(img, mask, filters=32)
	x = res_block(x, mask, 32)
	x = res_block(x, mask, 32)

	# Block 2: 128→64
	x, mask = gated_conv_block(x, mask, filters=64)
	x = res_block(x, mask, 64)
	x = res_block(x, mask, 64)

	# Block 3: 64→32
	x, mask = gated_conv_block(x, mask, filters=128)
	x = res_block(x, mask, 128)
	x = res_block(x, mask, 128)

	# Block 4: 32→16
	x, mask = gated_conv_block(x, mask, filters=256)
	x = Dropout(0.5)(x)

	# Head: 2 independent branches for the 4-parameter target
	x = GlobalAveragePooling2D()(x)
	x_shared = Dense(128, activation='relu')(x)

	b1 = Dense(64, activation='relu')(x_shared)
	branch_thickness = Dense(2, activation=None, name='branch_thickness')(b1)

	b2 = Dense(64, activation='relu')(x_shared)
	branch_size = Dense(2, activation=None, name='branch_size')(b2)

	model = Model(inp, {'branch_thickness': branch_thickness,
	                    'branch_size':      branch_size}, name='gated_two_branch')

	model.compile(
		optimizer=optimizer,
		loss={
			'branch_thickness': 'mse',
			'branch_size':      'mse',
		},
		metrics={
			'branch_thickness': ['mae'],
			'branch_size':      ['mae'],
		}
	)

	resume_path = None
	if os.path.exists(PATH_CHECKPOINT):
		resume_path = PATH_CHECKPOINT
	elif os.path.exists(PATH_MODEL):
		resume_path = PATH_MODEL

	if resume_path is not None:
		model = load_model(resume_path, safe_mode=False)
		print(f"Loaded existing model from: {resume_path}")
	else:
		print("Created a new model.")

	# 训练
	history = model.fit(
		train_dataset,
		validation_data=val_dataset,
		epochs=args.epochs,
		callbacks=[
			reduce_lr_callback,
			checkpoint_callback,
			PredictionComparisonCallback(preview_samples, PATH_PRED_DIR, args.comparison_interval),
		],
		verbose=2
	)

append_loss_history(PATH_LOSS_CSV, history)
plot_loss_from_csv(PATH_LOSS_CSV, PATH_LOSS_FIG, 'Total Loss')

save_prediction_comparison_figure(
	model,
	preview_samples,
	PATH_PRED_FIG,
	title='Predicted vs Ground Truth - Final Model',
)

model.save(PATH_MODEL)
