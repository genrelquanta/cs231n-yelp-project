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

----------------------------------------------------------------------
-- parameters

local opt = {}
opt.type = 'double' -- can change to 'cuda'
opt.save ='train_results' -- name of subdirectory to save the results in
opt.optimization = 'SGD' -- optimization method, can change to 'ADAM'
opt.learningRate = 1e-2 -- learning rate
opt.beta1 = 1e-3 -- beta1 for ADAM
opt.beta2 = 1e-3 -- beta2 for ADAM
opt.epsilon = 1e-3 -- epsilon for ADAM
opt.weightDecay = 0 -- regularization or weight decay parameter (only in SGD)
opt.momentum = 0 -- momentum for SGD
opt.plot = false -- live plot of results, can change to true
numTrain = 800 -- Number of training instances
batchSize = 16 -- Bag or batch size
featureDim = 4096 -- Size of features
trainDir = '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_features/'

----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   feature_net:cuda()
   criterion:cuda()
end
----------------------------------------------------------------------
print '==> defining some tools'

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the model
-- into a 1-dim vector

if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'ADAM' then
   optimState = {
      learningRate = opt.learningRate,
      beta1 = opt.beta1,
      beta2 = opt.beta2,
      epsilon = opt.epsilon
   }
   optimMethod = optim.adam

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd
end

----------------------------------------------------------------------



----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- loss tracker
   local loss = 0

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(numTrain)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

   trainData = torch.load('trainDataIds.dat')
   trainTargets = torch.load('trainTargets.dat')
   trainBiz = torch.load('trainBiz.dat')

   for t = 1,numTrain do
      -- print('Iteration number: '..t)

      -- local timeSample = sys.clock()

      -- create mini batch
      local inputs = torch.Tensor(batchSize,featureDim)
      local targets = trainTargets[trainBiz[shuffle[t]]]
      local photo_ids = trainData[trainBiz[shuffle[t]]]
      local numImgs = photo_ids:size(1)
      for i = 1,numImgs do
         -- load new sample
         local img = torch.load(trainDir..tostring(photo_ids[i])..'.dat')
         if opt.type == 'double' then input = img:double()
         elseif opt.type == 'cuda' then input = img:cuda() end
         inputs[{{i,i},{1,featureDim}}] = img:clone()
       end
      --print(inputs.type)

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       local output = model:forward(inputs)
                       local err = criterion:forward(output, targets)
                       f = f + err

                       -- estimate df/dW
                       local df_do = criterion:backward(output, targets)
                       model:backward(inputs, df_do)

                       loss = f

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
         optimMethod(feval, parameters, optimState)
         -- timeSample = sys.clock() - timeSample
         -- print("\n==> Loss = " ..loss)
   end

   -- time taken
   time = sys.clock() - time
   print("\n==> time taken for one epoch = " .. (time*1000) .. 'ms')
   print("\n==> Loss = " ..loss)
   -- update logger/plot
   trainLogger:add{['% Train Loss'] = loss}
   if opt.plot then
      trainLogger:style{['% Train Loss'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end