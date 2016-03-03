----------------------------------------------------------------------
-- This script defines the network architecture which will be used for
-- extracting features from the images and then train the multi-instance
-- multi-label classifier. This script is adapted from Clement Farabet's
-- script

-- Pulkit Agrawal
----------------------------------------------------------------------

require 'nn'
require 'loadcaffe'

----------------------------------------------------------------------

-- parameters
local hidden_dim = 64 -- hidden dimension of the fully connected layer
local num_classes = 9
local num_max_pooling = 5
local input_size = 224

local model_dir = '/home/raghav/Downloads/Yelp/'

----------------------------------------------------------------------


-- specify the feature extraction model to be loaded.
-- This script currently loads VGG-16

prototxt = 'VGG_ILSVRC_16_layers_deploy.prototxt'   -- specify the prototxt path
binary = 'VGG_ILSVRC_16_layers.caffemodel'  -- specify the binary path

-- this will load the network and print it's structure
feature_net = loadcaffe.load(model_dir..prototxt, model_dir..binary)

-- remove the fully connected layers from the VGG net

local num_remove = 9    -- number of layers to be removed from the top

for i=1,num_remove do
    feature_net:remove()
end

----------------------------------------------------------------------

-- we extract/define the trainable part of the feature network here

model = nn.Sequential() -- container for the entire trainable architecture

-- extract the top layers which need to be trained
-- index of the first and last layers from the top
-- which need to be inserted into the trainable net

local idx_start = 25
local idx_end = 31
feat_train_layers = idx_end-idx_start+1

for i=idx_start,idx_end do
    model:add(feature_net:get(i))
end

for i=1,feat_train_layers do
    feature_net:remove()
end

----------------------------------------------------------------------

-- add layers to the trainable model

local feat_out_size = input_size/(2^num_max_pooling)

-- stage 2 : linear -> relu
model:add(nn.View(-1):setNumInputDims(3))
model:add(nn.Linear(512*feat_out_size*feat_out_size, hidden_dim))
model:add(nn.ReLU())

-- stage 3: linear -> relu
model:add(nn.Linear(hidden_dim, num_classes))
model:add(nn.ReLU())

--stage 4: max
model:add(nn.Max())








