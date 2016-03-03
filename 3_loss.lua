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

loss_type = 'softmax' -- can change to 'margin'

----------------------------------------------------------------------
print '==> define loss'

if loss_type == 'margin' then

    -- This loss optimizes a multi-label multi-classification hinge loss
    -- (margin-based loss) between input x (a 1D Tensor) and output y (a 1D Tensor)

    criterion = nn.MultiLabelMarginCriterion()

elseif loss_type == 'softmax' then

    -- This loss optimizes a multi-label one-versus-all loss based on max-entropy,
    -- between input x (a 1D Tensor) and target y (a binary 1D Tensor)

    criterion = nn.MultiLabelSoftMarginCriterion()
end