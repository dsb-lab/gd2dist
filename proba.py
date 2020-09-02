import gaussiandeconvolution as gd

model = gd.nestedsampler()
model.fit([1,2,3],[1,2,3])
model.sample_deconvolution(10)