In logs directory, experimental results of ensemble-based systems are listed as follows:

* freeze: This version applys the weight freezing except for linear layer
* freeze_linear: This version applys the weight freezing including linear layer
* nonfreeze: This version does not apply the weight freezing
* nonfreezeconv1: This version applys the weight freezing except for the first convolutional layer
* nonfreezeconvs: This version applys the weight freezing except for last convolution layer of each basic block and linear layer
* nonfreezeshortcut: This version applys the weight freezing except for the fp32 shortcut and linear layer
