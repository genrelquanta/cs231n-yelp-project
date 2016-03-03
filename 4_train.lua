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
opt.type = 'cpu' -- can change to 'cuda'
opt.save ='train_results' -- name of subdirectory to save the results in
opt.optimization = 'SGD' -- optimization method, can change to 'ADAM'
opt.learningRate = 1e-3 -- learning rate
opt.beta1 = 1e-3 -- beta1 for ADAM
opt.beta2 = 1e-3 -- beta2 for ADAM
opt.epsilon = 1e-3 -- epsilon for ADAM
opt.batchSize = 8 -- equivalent to bag size here
opt.weightDecay = 0 -- regularization or weight decay parameter (only in SGD)
opt.momentum = 0 -- momentum for SGD
opt.plot = false -- live plot of results, can change to true
local numTrain = 1600 -- Number of training instances
local batchSize = 16 -- Bag or batch size
local fileDir = '~/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_photos/'
local imgSize = 224

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


-- return 224x224 random crop/resize of an image
function crop_or_resize(input_image_path)
  h = 224
  w = 224
  img = image.load(input_image_path)
  -- Directly scaled image
  scaled_img = image.scale(img, w, h)
  -- Cropped image
  _h = img:size()[2]
  _w = img:size()[3]

  if _h < h then _h = h end
  if _w < w then _w = w end

  img = image.scale(img, _w, _h)

  crop_h = 1
  crop_w = 1
  if _w > w then crop_w = torch.random(1, _w - w) end
  if _h > h then crop_h = torch.random(1, _h - h) end

  cropped_img = image.resize(img, crop_w, crop_h)

  return scaled_img
end

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
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   for t = 1,numTrain do

      -- create mini batch
      local inputs = torch.Tensor(batchSize,3,imgSize,imgSize)
      local targets = biz_to_targets[tostring(shuffle[t])]
      local photo_ids = biz_to_photo_ids[tostring(shuffle[t])]
      local photo_shuffle = torch.randperm(#photo_ids)
      local numImgs = math.min(batchSize,#photo_ids)
      for i = 1,numImgs do
         -- load new sample
         local img = crop_or_resize(fileDir..tostring(photo_ids[photo_shuffle[i]])..'.jpg')
         if opt.type == 'double' then input = img:double()
         elseif opt.type == 'cuda' then input = img:cuda() end
         inputs[{{i,i},{1,3},{1,imgSize},{1,imgSize}}] = img:clone()
      end

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
                       for i = 1,#inputs do
                          -- estimate f
                          local features = feature_net:forward(inputs)
                          local output = model:forward(features)
                          local err = criterion:forward(output, targets)
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets)
                          model:backward(features, df_do)

                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs
                       loss = f

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
         optimMethod(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

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