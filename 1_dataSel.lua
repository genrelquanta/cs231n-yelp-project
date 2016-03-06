----------------------------------------------------------------------
-- This script selects a subset of data for training and testing

-- Pulkit Agrawal
----------------------------------------------------------------------

require 'torch'

----------------------------------------------------------------------
-- parameters

local numTrain = 800 -- Number of training datapoints
local numTest = 200 -- Number of test datapoints
local batchSize = 16 -- Bag or batch size


trainTableInputs = {}
trainTableTargets = {}
testTableInputs = {}
testTableTargets = {}
testBiz = {}
trainBiz = {}

----------------------------------------------------------------------
print '==> selecting training and test datasets'


   -- local vars
   local time = sys.clock()


   -- shuffle at each epoch
   shuffle = torch.randperm(numTrain+numTest)

   -- load id data

   biz_to_targets = torch.load('biz_to_targets.dat')
   biz_to_photo_ids = torch.load('biz_to_photo_ids.dat')
   biz_ids = torch.load('biz_ids.dat')

   for t = 1,numTrain do
      print('Iteration number: '..t)

      -- create mini batch
      trainTableTargets[biz_ids[shuffle[t]]] = biz_to_targets[biz_ids[shuffle[t]]]
      local photo_ids = biz_to_photo_ids[biz_ids[shuffle[t]]]
      local photo_shuffle = torch.randperm(photo_ids:size(1))
      local numImgs = math.min(batchSize,photo_ids:size(1))
      local inputs = torch.Tensor(numImgs)
      trainBiz[t] = biz_ids[shuffle[t]]


      for i = 1,numImgs do
         inputs[{{i,i}}] = photo_ids[photo_shuffle[i]]
      end

      trainTableInputs[biz_ids[shuffle[t]]] = inputs:clone()

    end

   for t = numTrain+1,numTest+numTrain do
      print('Iteration number: '..t)

      -- create mini batch
      testTableTargets[biz_ids[shuffle[t]]] = biz_to_targets[biz_ids[shuffle[t]]]
      local photo_ids = biz_to_photo_ids[biz_ids[shuffle[t]]]
      local photo_shuffle = torch.randperm(photo_ids:size(1))
      local numImgs = math.min(batchSize,photo_ids:size(1))
      local inputs = torch.Tensor(numImgs)
      testBiz[t-numTrain] = biz_ids[shuffle[t]]


      for i = 1,numImgs do
         inputs[{{i,i}}] = photo_ids[photo_shuffle[i]]
      end

      testTableInputs[biz_ids[shuffle[t]]] = inputs:clone()

    end

    torch.save('trainTargets.dat',trainTableTargets)
    torch.save('trainDataIds.dat',trainTableInputs)
    torch.save('testTargets.dat',testTableTargets)
    torch.save('testDataIds.dat',testTableInputs)
    torch.save('testBiz.dat',testBiz)
    torch.save('trainBiz.dat',trainBiz)


   -- time taken
   time = sys.clock() - time
   print("\n==> time to select data = " .. (time*1000) .. 'ms')