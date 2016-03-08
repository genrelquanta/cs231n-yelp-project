----------------------------------------------------------------------
-- This script extracts features from the images using VGG Net

-- Pulkit Agrawal
----------------------------------------------------------------------

require 'torch'
require 'image'

----------------------------------------------------------------------
-- parameters

local type = 'double' -- can change to CUDA
local dataSize = 800 -- Number of datapoints
local batchSize = 16 -- Bag or batch size
local photoDir = '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_photos/'
local writeDirFC7= '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_features_fc7/'
local writeDirCV7= '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_features_cv7/'
local writeDirFC6= '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_features_fc6/'
local writeDirCV14= '/Users/Pulkit/Stanford/Academics/Courses/Winter2016/CS231N/Project/dataset/train_features_cv14/'
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
  return scaled_img
end

----------------------------------------------------------------------
print '==> extracting features'

   -- load train and test data indices
   trainData = torch.load('trainDataIds.dat')
   --testData = torch.load('testDataIds.dat')
   --valDataBiz = torch.load('valDataIds.dat')
   trainBiz = torch.load('trainBiz.dat')
   --testBiz = torch.load('testBiz.dat')
   --valBiz = torch.load('trainBiz.dat')

   for t = 1,dataSize do
      print('Iteration number: '..t)

      -- create mini batch
      local numImgs = trainData[trainBiz[t]]:size(1)
      for i = 1,numImgs do
         -- load new sample
         local time = sys.clock()
         img = crop_or_resize(photoDir..tostring(trainData[trainBiz[t]][i])..'.jpg')
         img = img*255
         local red = img[{{1,1},{1,imgSize},{1,imgSize}}]:add(-123.68)
         img[{{2,2},{1,imgSize},{1,imgSize}}]:add(-116.779)
         local blue = img[{{3,3},{1,imgSize},{1,imgSize}}]:add(-103.939):clone()
         img[{{3,3},{1,imgSize},{1,imgSize}}] = red
         img[{{1,1},{1,imgSize},{1,imgSize}}] = blue
         if type == 'double' then img = img:double()
         elseif type == 'cuda' then img = img:cuda() end
         features = feature_net:forward(img)
         -- save features
         torch.save(writeDirFC7..tostring(trainData[trainBiz[t]][i])..'.dat',features)
         torch.save(writeDirCV7..tostring(trainData[trainBiz[t]][i])..'.dat',feature_net.modules[31].output)
         torch.save(writeDirFC6..tostring(trainData[trainBiz[t]][i])..'.dat',feature_net.modules[34].output)
         torch.save(writeDirCV14..tostring(trainData[trainBiz[t]][i])..'.dat',feature_net.modules[24].output)
         -- time taken
         time = sys.clock() - time
         print("\n==> time to extract 1 img = " .. (time*1000) .. 'ms')
      end
    end


