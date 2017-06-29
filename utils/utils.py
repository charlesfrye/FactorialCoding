import os

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy

import copy

import itertools

from PIL import Image
import scipy.signal

#####
# Single Image Code
#

def getSingleImageEntropies(image=None, filt=None, filterSize=16, radius=8,
                  doWhiten=True, doNormalize=True, doZScore=False, plotMLE=False):
    
    preprocessSteps = [doZScore*'z-score',
                       doWhiten*'whiten',
                       doNormalize*'normalize local contrast']
    
    preprocessString = ', '.join([preprocessStep for preprocessStep in preprocessSteps
                                  if preprocessStep != '']
                                )
    
    if image is None:
        image = getRandomImage()
        
    originalImage = copy.deepcopy(image)
        
    if doZScore:
        image = zScore(image)
        
    if doWhiten:
        image = whiten(image)
    
    if doNormalize:
        image = localContrastNormalize(image,radius)
        
    finalImage = image
    
    if filt is None:
        filt = generateRandomFilter(filterSize)
    
    filteredImage = scipy.signal.convolve2d(finalImage,filt,mode='full')
    
    viewSingleResults(originalImage,finalImage,filteredImage,filt,preprocessString,plotMLE)

#####
# Loading Filters, Images, and Patches
#

def getRandomImageBatch(batchSize,normalize=True):
    return [getRandomImage(normalize=normalize) for _ in range(batchSize)]

def getRandomImage(normalize=True):
    imageNetPath = "/media/tbell/datasets/imagenet/ILSVRC/Data/DET/test"
    files = [file for file in os.listdir(imageNetPath) if file.split('.')[-1] == "JPEG"]
    randomFilePath = os.path.join(imageNetPath,np.random.choice(files))
    image = np.asarray(Image.open(randomFilePath).convert('L'))
    
    if normalize:
        image = image/256
        
    return image

def loadFilters(filename):
    fileType = filename.split('.')[-1]
    if fileType == "csv":
        return loadFilters_csv(filename)
    elif fileType == "npz":
        return loadFilters_npz(filename)
    else:
        print("error loading filters")

def loadFilters_csv(filename):
    rawFilts = np.loadtxt(filename,delimiter=',')
    reshaped3d = np.reshape(rawFilts,(12,12,288))
    split3d = np.dsplit(reshaped3d,288)
    filterBank = list(split3d)
    filterBank = [np.squeeze(filter) for filter in filterBank]
    return filterBank

def loadFilters_npz(filename):
    weights = np.load(filename)['data']
    
    filterBank = list(np.transpose(np.reshape(weights,(16,16,500)),axes=(2,0,1)))
    
    return filterBank

def generateRandomFilter(filterSize):
    filt = np.random.standard_normal(size=(filterSize,filterSize))
    filt = filt/np.sqrt(np.sum(np.square(filt)))
    
    return filt

def displayFromBank(filterBank,N):
    numFilters = len(filterBank)
    for filterIdx in np.random.choice(numFilters,N):
        plt.figure()
        imagePlot(filterBank[filterIdx], plt.gca())
        
def getPatchesBatch(imageBatch,size):
    patchesBatch = [getPatches(image,size) for image in imageBatch]
    return list(itertools.chain.from_iterable(patchesBatch))      

def getPatches(img,size):
    smallestDimension = min(img.shape)
    patchesPerRow = patchesPerColumn = smallestDimension // 12
    rowIndices = range(patchesPerRow); colIndices = range(patchesPerColumn)
    patches = [img[r*size:(r+1)*size,
                   c*size:(c+1)*size] for r,c in zip(rowIndices,colIndices)]
    return [patch for patch in patches if np.std(patch) > 0]

#####
# Image Processing
#

#####
### Normalization
###

def zScore(img):
    return (img-np.mean(img))/np.std(img)

def normalizeBatch(patchBatch):
    minim = np.min(patchBatch); maxim = np.max(patchBatch)
    normalizedBatch = [normalizePatch(patch,minim,maxim)
                            for patch in patchBatch]
    return normalizedBatch
    
def normalizePatch(patch,minim,maxim):
    return (patch-minim)/(maxim-minim)-1/2

##### 
### Whitening
###

def whitenBatch(imageBatch):
    return [whiten(image) for image in imageBatch]

def whiten(img):
    
    lengths = list(img.shape)

    for dimension in range(2):
        if (lengths[dimension]%2) != 0:
            lengths[dimension] = lengths[dimension]-1
            
    img = img[:lengths[0],:lengths[1]]
    
    filtf = generateWhiteningFilter(img)
    
    imgF = fft.fftshift(fft.fft2(img))
    
    imgWhiteF = np.multiply(imgF,filtf)
    
    imgWhite = np.real(fft.ifft2(fft.ifftshift(imgWhiteF)))
    
    return imgWhite

def generateWhiteningFilter(img):
    
    lengths = list(img.shape)
    
    nyquists = [length/2 for length in lengths]
    frequencies = [np.linspace(-nyquist,nyquist,num=length) 
                   for length,nyquist in zip(lengths,nyquists)]
    
    frequencyGrid = np.meshgrid(*frequencies,indexing='ij')
    
    ramp = np.sqrt(np.square(frequencyGrid[0])+np.square(frequencyGrid[1]))
    lowpass = np.exp(-0.5*np.square(ramp/(0.7*np.mean(nyquists))))
    
    filt = np.multiply(ramp,lowpass)
    
    return filt

#####
### Contrast Normalization
###

def contrastNormalizeBatch(imageBatch):
    return [localContrastNormalize(image,12) for image in imageBatch]

def localContrastNormalize(img,r):
    pooler = generateLocalContrastNormalizer(r)
    localIntensityEstimate = scipy.signal.convolve2d(np.square(img),pooler,mode='same')
    normalizedImage = np.divide(img,np.sqrt(localIntensityEstimate))
    return normalizedImage

def generateLocalContrastNormalizer(radius):
    
    xs = np.linspace(-radius,radius-1,num=2*radius)
    xs,ys = np.meshgrid(xs,xs)
    gauss = np.exp(-0.5*((np.square(xs)+np.square(ys))/radius**2))
    gauss = gauss/np.sum(gauss)
    
    return gauss

###
# Entropy Calculations
#

def estimatePixelwiseEntropies(patches):
    patchDims = patches[0].shape
    patchArea = patchDims[0]*patchDims[1]
    
    flattenedPatches = [np.reshape(patch,(patchArea,)) for patch in patches]
    flatArray = np.asarray(flattenedPatches)
    
    pixelHistograms = []
    pixelEntropies = []
    
    for pixelIdx in range(patchArea):
        pixelHistogram = np.histogram(flatArray[:,pixelIdx],
                                      bins=100,normed=True)
        ps = pixelHistogram[0]; edges = pixelHistogram[1]
        pixelEntropy = calcuateDifferentialEntropy(ps,edges)
        pixelHistograms.append(pixelHistogram)
        pixelEntropies.append(pixelEntropy)
    
    return pixelEntropies, pixelHistograms

def estimateMarginalEntropy(patches):
    
    marginalHistogram = np.histogram(patches,
                                    bins = 100,
                                    normed = True)
    ps = marginalHistogram[0]; edges = marginalHistogram[1]
    marginalEntropy = calcuateDifferentialEntropy(ps,edges)
    
    return {'entropy':marginalEntropy,
            'histogram':marginalHistogram}

def calcuateDifferentialEntropy(ps,edges):
    ws = np.diff(edges)
    entropy = np.sum([-p*np.log(p/w) for p,w in zip(ps,ws)
                                     if p != 0])
    return entropy

def fitGauss(x):
    return np.mean(x), np.var(x)
    
def gaussDist(xs,mu,var):
    Z = np.sqrt(2*np.pi*var)
    return (1/Z)*np.exp(-np.square(xs-mu)/(2*var))

###
# Plotting
#

def viewBatchResults(resultsDict,plotGauss=False):
    numResults = len(resultsDict)
    plt.figure(figsize=(12,4*numResults))
    batches = resultsDict.keys()
    
    yMin = np.inf; yMax = -np.inf;
    
    for idx,batch in enumerate(batches):
        result = resultsDict[batch]
        density = result['histogram'][0]
        nonZeroDensity = density[np.where(np.greater(density,0))]
        mn = np.min(nonZeroDensity)
        mx = np.max(nonZeroDensity)
        if mn*0.9 < yMin:
            yMin = mn*0.9
        if mx*1.1 > yMax:
            yMax = mx*1.1
    
    for idx,batch in enumerate(batches):
        result = resultsDict[batch]
        histogram = result['histogram']
        if idx == 0:
            histAxs = []
            histAxs.append(plt.subplot(numResults,1,idx+1))
        else:
            #histAxs.append(plt.subplot(numResults,1,idx+1,sharey=histAxs[0]))
            histAxs.append(plt.subplot(numResults,1,idx+1))
            
        plotHistogram(histogram)
        
        if (plotGauss == True) & ('gaussMu' in result.keys()):
            
            mu = result['gaussMu']; var = result['gaussVar']
            leftEdges = histogram[1]
            
            gaussMLEpdf = gaussDist(leftEdges,mu,var)
            h = plt.plot(leftEdges,gaussMLEpdf,
                            color='r',linewidth=4,label='MLE gauss')
            plt.legend()

        plt.gca().set_xlim([-0.55,0.55])
        plt.gca().set_ylim([yMin,yMax])

        plt.title(batch,fontweight='bold',fontsize='xx-large')
        
    for idx,batch in enumerate(batches):
        result = resultsDict[batch]
        entropy = result['entropy']
        
        addEntropyLabel(entropy,histAxs[idx])
       
    plt.tight_layout()

def viewSingleResults(original,final,filtered,filt,preprocessString,plotMLE):
    
    plt.figure(figsize=(16,16))
    subplotsWide = 5
    subplotsHigh = 3
    subplotGridShape = (subplotsWide,subplotsHigh)
    
    yMax = -np.inf
    yMin = np.inf
    
    imageAxs = [plt.subplot2grid(subplotGridShape,(idx//2,idx%2)) 
                               for idx in range(3)]
    
    images = [original,final,filtered,]
    titles = ["raw", "pre-processed", "filtered"]
    
    imagesInfo = zip(images,
                titles,
                imageAxs)
    
    for imageInfo in imagesInfo:
        image, title, ax = imageInfo
        imagePlot(image,ax)
        titleAxis(title + " image",ax)
    
    filterAx = plt.subplot2grid(subplotGridShape,(1,1))
    imagePlot(filt,filterAx)
    titleAxis("filter",filterAx)
    
    histAxs = [plt.subplot2grid(subplotGridShape,(idx,2),colspan=3) for idx in range(3)]
    
    for idx,(image,title,ax) in enumerate(zip(images,titles,histAxs)):
        
        ps,locs,_ = ax.hist(np.ravel(image), bins=100,log=True,normed=True);
        
        if 1.1*max(ps) > 1.1*yMax:
            yMax = 1.1*max(ps)
        if 0.9*min(ps) < 0.9*yMin:
            yMin = 0.9*min(ps[ps>0])
            
        if plotMLE:
            plotGaussMLE(locs,image,ax=ax)
            
        if idx == 1:
            plt.ylabel("log probability",
                fontsize="x-large",
                fontweight='bold')
            
        if title == "pre-processed":
            titleAxis("pre-processed pixel histogram:\n"+ preprocessString,ax)
        else:
            titleAxis(title + " histogram",ax)

    for ax in histAxs:
        ax.set_ylim(bottom=yMin,top=yMax)
        
    plt.tight_layout()
    return

def imagePlot(image,ax):
    ax.imshow(image, cmap="Greys_r")
    removeAxes(ax)
        
def titleCurrentAxis(title):
    plt.title(title, fontsize="large",fontweight="bold")
    
def titleAxis(title,ax):
    ax.set_title(title, fontsize="large",fontweight="bold")

def removeAxes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plotGaussMLE(xs,values,ax=None):
    mu,var = fitGauss(values)
    gaussMLEpdf = gaussDist(xs,mu,var)
    if ax is None:
        ax = plt.gca()
    h = ax.plot(xs,gaussMLEpdf,
            color='r',linewidth=4,label='MLE gauss')
    ax.legend()
    
def randomHistogramResult(entropies,histograms):
    randomIdx = np.random.choice(len(histograms))
    entropy = entropies[randomIdx]
    histogram = histograms[randomIdx]
    leftEdges = histogram[1][:-1]
    
    plt.figure(figsize=(12,4))
    plt.bar(histogram[1][:-1],histogram[0],width=np.diff(histogram[1]),log=True);
    print(entropy)
    
def plotHistogram(histogram):
    ps = histogram[0]
    leftEdges = histogram[1][:-1]
    width = np.diff(histogram[1])
    plt.bar(leftEdges,ps,width=width,log=True)
    
def addEntropyLabel(entropy,ax):
    x = np.dot(ax.get_xlim(),[2/3,1/3])
    y = np.float_power((ax.get_ylim()[1]/ax.get_ylim()[0]),1/6)*ax.get_ylim()[0]
    ax.text(x,y,'$H = $'+str(round(entropy,2)), fontsize="x-large",
               fontweight='bold')