from utils.nsc import NotSoCleverCreator as NSCCreator
from matplotlib import pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--canvas_size', metavar='canvas_size',
                    help='Canvas Size; default: 64, 64', type=int,
                    nargs='+', default=(64, 64))
parser.add_argument('--center_region', metavar='center_region',
                    help='Center coordinates region; default: 56, 56', type=int,
                    nargs='+', default=(56, 56))
parser.add_argument('--square_size', metavar='square_size',
                    help='Square Size; default: 9, 9', type=int,
                    nargs='+', default=(9, 9))

args = vars(parser.parse_args())


nsc = NSCCreator(**args)

# Uniform distribution
#all_coord_imgs = nsc.one_hot_coords(args['canvas_size'], nsc.coords)
#np.random.shuffle(all_coord_imgs)
#split = int(0.8 * all_coord_imgs.shape[0])
#train_imgs = all_coord_imgs[:split]
#test_imgs = all_coord_imgs[split:]

train_coords, test_coords = nsc.uniform_split()
train_imgs = nsc.one_hot_coords(args['canvas_size'], train_coords)
test_imgs = nsc.one_hot_coords(args['canvas_size'], test_coords)

summation_train = np.sum(train_imgs, axis=0)
summation_test = np.sum(test_imgs, axis=0)

# quadrant distribution
train_quad_coords, test_quad_coords = nsc.quadrant_split()
train_quad_imgs = nsc.one_hot_coords(args['canvas_size'], train_quad_coords)
test_quad_imgs = nsc.one_hot_coords(args['canvas_size'], test_quad_coords)

summation_train_quad = np.sum(train_quad_imgs, axis=0)
summation_test_quad = np.sum(test_quad_imgs, axis=0)

plt.figure(figsize=(7, 7))
plt.subplot(2, 3, 1)
plt.title('Sum of all\ntrain points')
plt.imshow(summation_train[..., 0], cmap='gray')
plt.axis('off')
#plt.text(-20, args['canvas_size'][0] // 2, 'Uniform\nSplit')

plt.subplot(2, 3, 2)
plt.title('Sum of all\ntest points')
plt.imshow(summation_test[..., 0], cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.text(0, 0.5, 'Uniform\nSplit')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(summation_train_quad[..., 0], cmap='gray')
plt.axis('off')
#plt.text(-20, args['canvas_size'][0] // 2, 'Quadrant\nSplit')

plt.subplot(2, 3, 5)
plt.imshow(summation_test_quad[..., 0], cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.text(0, 0.5, 'Quadrant\nSplit')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

plt.tight_layout()
plt.show()