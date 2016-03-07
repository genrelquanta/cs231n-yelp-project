----------------------------------------------------------------------
-- This script defines the training procedure to be used with the network
-- architecture defined in the model script. (Adapted from script by
-- Clement Farabet)
-- Pulkit Agrawal
----------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'image'
require 'csvigo'

----------------------------------------------------------------------
-- parameters


numTrain = 800 -- Number of training instances
batchSize = 16 -- Bag or batch size
featureDim = 4096 -- Size of features
trainDir = '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_features/'


   -- shuffle at each epoch
   shuffle = torch.randperm(numTrain)

   trainData = torch.load('trainDataIds.dat')
   trainTargets = torch.load('trainTargets.dat')
   trainBiz = torch.load('trainBiz.dat')

   bizFeats = {}
   bizTar = {}

   for t = 1,numTrain do
      print(t)
      local inputs = torch.Tensor(batchSize,featureDim)
      local targets = trainTargets[trainBiz[shuffle[t]]]
      local photo_ids = trainData[trainBiz[shuffle[t]]]
      local numImgs = photo_ids:size(1)
      for i = 1,numImgs do
         local img = torch.load(trainDir..tostring(photo_ids[i])..'.dat')
         inputs[{{i,i},{1,featureDim}}] = img:clone()
       end
       bizFeats[t] = inputs:mean(1)
       bizTar[t] = targets:clone()
    end

matio.save('bizFeatsTrain.mat',bizFeats)
matio.save('bizTarTrain.mat',bizTar)

