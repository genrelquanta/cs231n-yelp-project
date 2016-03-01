----------------------------------------------------------------------
-- This script defines the network architecture which will be used for
-- extracting features from the images and then train the multi-instance
-- multi-label classifier. This script is adapted from Clement Farabet's
-- script

-- Pulkit Agrawal
----------------------------------------------------------------------

require 'nn'
require 'loadcaffe'
require 'xlua'
require 'optim'

----------------------------------------------------------------------

-- parameters
nstates = {16,256,128}
bagsize = 10
hidden_dim = 256 -- hidden dimension of the fully connected layer
num_classes = 10

----------------------------------------------------------------------


-- specify the feature extraction model to be loaded.
-- This script currently loads VGG-16

prototxt = 'VGG_ILSVRC_16_layers_deploy.prototxt'   -- specify the prototxt path
binary = 'VGG_ILSVRC_16_layers.caffemodel'  -- specify the binary path
-- backend = 'ccn2' -- specify the backend

-- this will load the network and print it's structure
caffe_net = loadcaffe.load(prototxt, binary)

-- remove the fully connected layers from the VGG net

local num_remove = 8    -- number of layers to be removed from the top

for i=1,num_remove do
    caffe_net:remove()
end

----------------------------------------------------------------------

-- we extract/define the trainable part of the caffe network here

model = nn.Sequential() -- container for the entire trainable architecture
feat_train_pipe = nn.Sequential() -- the trainable feature pipe

-- extract the top layers which need to be trained
-- index of the first and last layers from the top
-- which need to be inserted into the trainable net

local idx_start = 25
local idx_end = 32
feat_train_layers = idx_end-idx_start+1

for i=idx_start,idx_end do
    feat_train_pipe:add(caffe_net:get(i))
end

for i=1,num_feat_layers do
    caffe_net:remove()
end

----------------------------------------------------------------------
-- Replicate the caffe_net to generate a feature extractor

local num_feat_layers = 24 -- Number of layers in the feature extractor


feature_net = nn.Parallel(1,1) -- container for the feature extractor

for i=1,bagsize do
    mlp = nn.Sequential()
    for j=1,num_feat_layers do
        mlp:add(caffe_net:get(j))
    end
    mlp:share(caffe_net,'bias','weight')
    feature_net:add(mlp)
end

----------------------------------------------------------------------

-- add layers to the trainable model

-- stage 1 : input -> trainable extractor
feat_train_net = nn.Parallel(1,1)
for i=1,bagsize do
    mlp = nn.Sequential()
    for j=1,feat_train_layers do
        mlp:add(feat_train_pipe:get(j))
    end
    mlp:share(feat_train_pipe,'bias','weight','gradWeight','gradBias')
    feat_train_net:add(mlp)
end
model:add(feat_train_net)
model:add(nn.Reshape(1,bagsize,512*7*7))

-- stage 2 : convl -> relu
model:add(nn.SpatialConvolution(1, hidden_dim, 1, 512*7*7, 1, 512*7*7))
model:add(nn.ReLU())
model:add(nn.Reshape(1,bagsize,hidden_dim))

-- stage 3: convl -> relu
model:add(nn.SpatialConvolution(1, num_classes, 1, hidden_dim, 1, hidden_dim))
model:add(nn.ReLU())

--stage 4: max
model:add(nn.Max(2))








