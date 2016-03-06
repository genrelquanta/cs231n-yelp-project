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
local num_classes = 9

local model_dir = '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/Code/'

----------------------------------------------------------------------


-- specify the feature extraction model to be loaded.
-- This script currently loads VGG-16

prototxt = 'VGG_ILSVRC_16_layers_deploy.prototxt'   -- specify the prototxt path
binary = 'VGG_ILSVRC_16_layers.caffemodel'  -- specify the binary path

-- this will load the network and print it's structure
feature_net = loadcaffe.load(model_dir..prototxt, model_dir..binary)

-- remove the softmax layer and the last fully connected layer from the VGG net

local num_remove = 3    -- number of layers to be removed from the top

for i=1,num_remove do
    feature_net:remove()
end








