----------------------------------------------------------------------
-- This script defines the training procedure to be used with the network
-- architecture defined in the model script. (Adapted from script by
-- Clement Farabet)
-- Pulkit Agrawal
----------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'optim'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('MIML Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ADAM')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-beta1', 1e-3, 'beta1 at t=0 for adam')
   cmd:option('-beta2', 1e-3, 'beta2 at t=0 for adam')
   cmd:option('-epsilon', 1e-3, 'epsilon for adam')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:text()
   opt = cmd:parse(arg or {})
end
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
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

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
      learningRate = opt.learningRate
      beta1 = opt.beta1
      beta2 = opt.beta2
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

else
   error('unknown optimization method')
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
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         if opt.type == 'double' then input = input:double()
         elseif opt.type == 'cuda' then input = input:cuda() end
         table.insert(inputs, input)
         table.insert(targets, target)
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
                          local features = feature_net:forward(inputs[i])
                          local output = model:forward(features)
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
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