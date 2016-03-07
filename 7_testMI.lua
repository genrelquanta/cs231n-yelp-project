----------------------------------------------------------------------
-- This script defines the training procedure to be used with the network
-- architecture defined in the model script. (Adapted from script by
-- Clement Farabet)
-- Pulkit Agrawal
----------------------------------------------------------------------

require 'torch'
require 'image'

----------------------------------------------------------------------
-- parameters

local opt = {}
plot = false
opt.save ='test_results' -- name of subdirectory to save the results in
numTest = 800 -- Number of test instances
batchSize = 16 -- Bag or batch size
featureDim = 4096 -- Size of features
testDir = '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_features/'

----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end
----------------------------------------------------------------------
print '==> defining some tools'

-- Log results to files
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

----------------------------------------------------------------------
print '==> defining test procedure'

function test()

   -- accuracy tracker
   accuracy = 0

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- do one epoch
   print('==> testing accuracy on training data:')

   testData = torch.load('trainDataIds.dat')
   testTargets = torch.load('trainTargets.dat')
   testBiz = torch.load('trainBiz.dat')

   for t = 1,numTest do
      -- create mini batch
      local inputs = torch.Tensor(batchSize,featureDim)
      local targets = testTargets[testBiz[t]]
      local photo_ids = testData[testBiz[t]]
      local numImgs = photo_ids:size(1)
      for i = 1,numImgs do
         -- load new sample
         local img = torch.load(testDir..tostring(photo_ids[i])..'.dat')
         if opt.type == 'double' then input = img:double()
         elseif opt.type == 'cuda' then input = img:cuda() end
         inputs[{{i,i},{1,featureDim}}] = img:clone()
       end
      local acc = torch.ne(torch.gt(model:forward(inputs),0):double(),targets):sum()/9
      accuracy = accuracy + acc
   end

   print("\n==> Accuracy = " ..accuracy/numTrain)
   -- update logger/plot
   testLogger:add{['% Test Accuracy'] = accuracy/numTrain}
   if plot then
      trainLogger:style{['% Test Accuracy'] = '-'}
      trainLogger:plot()
   end

end
