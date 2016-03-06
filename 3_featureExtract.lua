----------------------------------------------------------------------
-- This script extracts features from the images using VGG Net

-- Pulkit Agrawal
----------------------------------------------------------------------

require 'torch'
require 'image'

----------------------------------------------------------------------
-- parameters

local type = 'double' -- can change to CUDA
local numTrain = 800 -- Number of training datapoints
local numTest = 200 -- Number of test datapoints
local batchSize = 16 -- Bag or batch size
local photoDir = '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_photos/'
local writeDirTrain= '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_features/'
local writeDirTest = '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/test_features/'
local imgSize = 224
----------------------------------------------------------------------

-- load the feature extractor

dofile '2_modelFeature.lua'


----------------------------------------------------------------------
-- CUDA?
if type == 'cuda' then
   feature_net:cuda()
end
----------------------------------------------------------------------
print '==> defining some tools'

-- return 224x224 random crop/resize of an image
function crop_or_resize(input_image_path)
  h = imgSize
  w = imgSize
  img = image.load(input_image_path)
  -- Directly scaled image
  scaled_img = image.scale(img, w, h)
  --[[
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
  ]]--

  return scaled_img
end

----------------------------------------------------------------------
print '==> extracting features'



   -- load train and test data indices
   trainData = torch.load('trainDataIds.dat')
   testData = torch.load('testDataIds.dat')
   trainBiz = torch.load('trainBiz.dat')
   testBiz = torch.load('testBiz.dat')

   for t = 1,numTrain do
      print('Iteration number: '..t)

      -- create mini batch
      -- local inputs = torch.Tensor(batchSize,3,imgSize,imgSize)
      local numImgs = trainData[trainBiz[t]]:size(1)
      for i = 1,numImgs do
         -- load new sample
         local time = sys.clock()
         local img = crop_or_resize(photoDir..tostring(trainData[trainBiz[t]][i])..'.jpg')
         img[{{1,1},{1,imgSize},{1,imgSize}}]:add(-123.68)
         img[{{2,2},{1,imgSize},{1,imgSize}}]:add(-116.779)
         img[{{3,3},{1,imgSize},{1,imgSize}}]:add(-103.939)
         if type == 'double' then img = img:double()
         elseif type == 'cuda' then img = img:cuda() end
         local features = feature_net:forward(img)
         -- time taken
         time = sys.clock() - time
         print("\n==> time to extract 1 img = " .. (time*1000) .. 'ms')
         torch.save(writeDirTrain..tostring(trainData[trainBiz[t]][i])..'.dat',features)
         --inputs[{{i,i},{1,3},{1,imgSize},{1,imgSize}}] = img:clone()
      end

      --[[ local vars
      local time = sys.clock()

      local features = feature_net:forward(inputs)

      -- time taken
      time = sys.clock() - time
      print("\n==> time to extract 1 business = " .. (time*1000) .. 'ms')

      torch.save(writeDirTrain..trainBiz[t]..'.dat',features)
      ]]--

    end


    for t = 1,numTest do
      print('Iteration number: '..t)

      -- create mini batch
      local inputs = torch.Tensor(batchSize,3,imgSize,imgSize)
      local numImgs = testData[testBiz[t]]:size(1)
      for i = 1,numImgs do
         -- load new sample
         local time = sys.clock()
         local img = crop_or_resize(photoDir..tostring(testData[testBiz[t]][i])..'.jpg')
         img[{{1,1},{1,imgSize},{1,imgSize}}]:add(-123.68)
         img[{{2,2},{1,imgSize},{1,imgSize}}]:add(-116.779)
         img[{{3,3},{1,imgSize},{1,imgSize}}]:add(-103.939)
         if type == 'double' then img = img:double()
         elseif type == 'cuda' then img = img:cuda() end
         local time = sys.clock()
         local features = feature_net:forward(img)
         -- time taken
         time = sys.clock() - time
         print("\n==> time to extract 1 img = " .. (time*1000) .. 'ms')
         torch.save(writeDirTest..tostring(testData[testBiz[t]][i])..'.dat',features)
         --inputs[{{i,i},{1,3},{1,imgSize},{1,imgSize}}] = img:clone()
      end

      --[[ local vars
      local time = sys.clock()

      local features = feature_net:forward(inputs)

      -- time taken
      time = sys.clock() - time
      print("\n==> time to extract 1 business = " .. (time*1000) .. 'ms')

      torch.save(writeDirTest..testBiz[t]..'.dat',features)
      ]]--

    end

