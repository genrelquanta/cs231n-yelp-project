----------------------------------------------------------------------
-- This script converts target labels from sets to tensors

-- Pulkit Agrawal
----------------------------------------------------------------------

require 'csvigo'
require 'torch'

----------------------------------------------------------------------

num_classes = 9

file = csvigo.load({path="biz_label_exp.csv",mode="large"})
biz_to_target = {}
for i=1,2000 do
    target = torch.zeros(num_classes)
    for j=2,#file[i] do
        target[file[i][j]+1] = 1
    end
    biz_to_target[tostring(file[i][1])] = target:clone()
end
torch.save('biz_to_targets.dat',biz_to_target)

file = csvigo.load({path="biz_to_photos.csv",mode="large"})
biz_to_photos = {}
for i=1,2000 do
    photos = torch.zeros(#file[i]-1)
    for j=2,#file[i] do
        photos[j-1] = file[i][j]
    end
    biz_to_photos[tostring(file[i][1])] = photos:clone()
end
torch.save('biz_to_photo_ids.dat',biz_to_photos)


