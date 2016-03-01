----------------------------------------------------------------------
-- This script defines the loss functions which can be used with the
-- multi-instance multi-label classifier. The following loss functions
-- can be defined:
--   + multi-label softmax loss
--   + multi-label margin loss (SVM-like)
-- (Adapted from script by Clement Farabet)
-- Pulkit Agrawal
----------------------------------------------------------------------

require 'torch'
require 'nn'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
print '==> processing options'
cmd = torch.CmdLine()
cmd:text()
cmd:text('MIML Loss Function')
cmd:text()
cmd:text('Options:')
cmd:option('-loss', 'softmax', 'type of loss function to minimize: softmax | margin')
cmd:text()
opt = cmd:parse(arg or {})

-- Number of classes
local num_classes = 10

----------------------------------------------------------------------
print '==> define loss'

if opt.loss == 'margin' then

    -- This loss optimizes a multi-label multi-classification hinge loss
    -- (margin-based loss) between input x (a 1D Tensor) and output y (a 1D Tensor)

    criterion = nn.MultiLabelMarginCriterion()

elseif opt.loss == 'softmax' then

    -- This loss optimizes a multi-label one-versus-all loss based on max-entropy,
    -- between input x (a 1D Tensor) and target y (a binary 1D Tensor)

    criterion = nn.MultiLabelSoftMarginCriterion()


else

    error('unknown -loss')

end

