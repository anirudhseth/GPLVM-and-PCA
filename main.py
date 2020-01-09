import pcaAlgo

#Dataset 1#
Y = 'Dataset1/train_data.txt'
labels='Dataset1//labels.txt'
lowerDim=2
title='PCA on Oil Flow Data'
pca1=pcaAlgo.pcaAlgo(lowerDim,Y,labels,title)
pca1.fitandPLot()

#Dataset 2#
