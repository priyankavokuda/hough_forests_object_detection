import numpy as np

# We start with super simple features
# and extend this later on to deep features (I have already code for this).

class FeatureExtractor(object):
    """ FeatureExtractor abstract class """

    def __init__(self, **kwargs):
        pass

    #@abstractmethod
    def get_features(self, **kwargs):
        # Should return a feature bank with shape
        # (X,Y,N) where (X,Y) is the shape of the input image, feat_dim
        # should also be an atribute of the class with value N , the length of the
        # feature vector
        pass

class FeatureExtractorRaw(FeatureExtractor):
    """ Just use RGB values as features """

#    scale = 255.0
    scale = 1.0
    feat_dim = 3

    def __init__(self, **kwargs):
        super(FeatureExtractorRaw, self).__init__(**kwargs)
        if 'scale' in kwargs:
            self.scale = kwargs['scale']

    def get_features(self, img):

#        feat = np.copy(img)/self.scale
#        if len(feat.shape)==2:
#            # grayscale image
#            feat = np.reshape(feat,
#                    (feat.shape[0], feat.shape[1], 1))
#            self.feat_dim = 1
#        else:
#            self.feat_dim = img.shape[2]

        return img


class FeatureExtractorStack(FeatureExtractor):
    """ Concatenated features """

    extractors = []
    feat_dim = 0

    def __init__(self, **kwargs):
        super(FeatureExtractorStack, self).__init__(**kwargs)
        if 'extractors' in kwargs:
            self.extractors = kwargs['extractors']
            self.feat_dim = sum( [e.feat_dim for e in self.extractors ] )

    def isempty(self):
        return (len(self.extractors)==0)

    def append(self, extractor):
        self.extractors.append(extractor)
        self.feat_dim += extractor.feat_dim

    def get_features(self, img):
        feature_layers = np.dstack( [ e.get_features(img) for e in self.extractors ] )
        return feature_layers

