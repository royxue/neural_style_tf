import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import tensorflow as tf
import vgg

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
VGG_19_PATH = './vgg19.mat'

CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
CONTENT_LAYER_WEIGHT = 1
STYLE_LAYER_WEIGHT_EXP = 1
ITERATIONS = 1001

def _tensor_size(tensor):
	size = 1
	for d in tensor.get_shape():
		size *= d.value
	return size

def ns_net(content, style):
	img_shape = (300, 300)
	content_image = scipy.misc.imread(content).astype(np.float)
	if content_image.shape[2] == 4:
		content_image = content_image[:,:,:3]

	content = scipy.misc.imresize(content_image, img_shape)

	style_image = scipy.misc.imread(style).astype(np.float)
	if style_image.shape[2] == 4:
		style_image = style_image[:,:,:3]
	style = scipy.misc.imresize(style_image, img_shape)

	shape = (1,) + content.shape

	# Load VGG19 Network
	vgg_weights, vgg_mean_pixel = vgg.load_net(VGG_19_PATH)
	print("VGG19 Loaded")

	# Generate content and style weights
	content_features = {}
	style_features = {}

	content_layers_weights = {}
	content_layers_weights['relu4_2'] = CONTENT_LAYER_WEIGHT
	content_layers_weights['relu5_2'] = 1.0 - CONTENT_LAYER_WEIGHT

	layer_weight = 1.0
	style_layers_weights = {}
	for style_layer in STYLE_LAYERS:
		style_layers_weights[style_layer] = layer_weight
		layer_weight *= STYLE_LAYER_WEIGHT_EXP

	layer_weights_sum = 0
	for style_layer in STYLE_LAYERS:
		layer_weights_sum += style_layers_weights[style_layer]
	for style_layer in STYLE_LAYERS:
		style_layers_weights[style_layer] /= layer_weights_sum

	# extract content feature
	g = tf.Graph()
	with g.as_default(), tf.Session() as sess:
		image = tf.placeholder('float', shape=shape)
		net = vgg.net_preloaded(vgg_weights, image, 'max')
		content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
		for layer in CONTENT_LAYERS:
			content_features[layer] = net[layer].eval(feed_dict={image: content_pre})
	print("Content Feature Extracted")

	# extract style feature
	g = tf.Graph()
	with g.as_default(), tf.Session() as sess:
		image = tf.placeholder('float', shape=shape)
		net = vgg.net_preloaded(vgg_weights, image, 'max')
		style_pre = np.array([vgg.preprocess(style, vgg_mean_pixel)])
		for layer in STYLE_LAYERS:
			features = net[layer].eval(feed_dict={image: style_pre})
			features = np.reshape(features, (-1, features.shape[3]))
			gram = np.matmul(features.T, features) / features.size
			style_features[layer] = gram
	print("Style Feature Extracted")


	g = tf.Graph()
	with g.as_default():
		# starting from a random image
		initial = tf.random_normal(shape) * 0.256

		image = tf.Variable(initial)
		net = vgg.net_preloaded(vgg_weights, image, 'max')

		# Calculat the loss function
		content_loss = 0.
		for content_layer in CONTENT_LAYERS:
			content_loss += (content_layers_weights[content_layer]*tf.nn.l2_loss(net[content_layer]-
				content_features[content_layer])*2/content_features[content_layer].size)
		content_loss *= CONTENT_WEIGHT

		style_loss = 0.
		for style_layer in STYLE_LAYERS:
			layer = net[style_layer]
			_, height, width, number = map(lambda i: i.value, layer.get_shape())
			size = height * width * number
			feats = tf.reshape(layer, (-1, number))
			gram = tf.matmul(tf.transpose(feats), feats) / size
			style_gram = style_features[style_layer]
			style_loss += (style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
		style_loss *= STYLE_WEIGHT

		tv_y_size = _tensor_size(image[:,1:,:,:])
		tv_x_size = _tensor_size(image[:,:,1:,:])
		tv_loss = TV_WEIGHT * 2 * (
			(tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
				tv_y_size) +
			(tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
				tv_x_size))

		# total loss
		loss = content_loss + style_loss# + tv_loss

		# Use default value for the optimizer
		train_step = tf.train.AdamOptimizer(learning_rate=1e1).minimize(loss)

		print("Start Iterations")
		steps = 100
		i = 0
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(ITERATIONS):
				train_step.run()
				if(i%steps == 0 and i>0):
					print(str(i)+"iterations done")
					res = image.eval()
					img_out = vgg.unprocess(res.reshape(shape[1:]), vgg_mean_pixel)
					img_out = np.clip(img_out, 0, 255).astype(np.uint8)
					mpimg.imsave('./test_%d.png'%(i), img_out)

if __name__ == '__main__':
	ns_net('./content_1.png', './starry.jpg')