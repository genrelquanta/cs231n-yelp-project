----------------------------------------------------------------------
-- This file trains the linear model on the data
--
-- Clement Farabet
----------------------------------------------------------------------
require 'torch'

----------------------------------------------------------------------
print '==> executing all'

dofile '4_loss.lua'
dofile '5_modelLinear.lua'
dofile '6_trainLinearMI.lua'

----------------------------------------------------------------------
print '==> training!'

while true do
   train()
   --test()
end
