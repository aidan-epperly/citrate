--!strict

local h = require("./harness")

require("./complex_scalar")
require("./real")
require("./complex")
require("./space")
require("./qr")
require("./inplace")
require("./solve")
require("./qrp")
require("./chol")
require("./eigen")
require("./eig")
require("./svd")
require("./fft")
require("./structured")
require("./sparse")
require("./precond")
require("./cheb")

h.finish()
