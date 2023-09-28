import pandas as pd
import random
from math import log


_path : str = ".WorkAll/InptData/test_this.csv"
_labels : list = ["pclass", "sex", "embarked", "survived"]

class Do_Decision_Tree():
    def __init__(self,
                 classCount : dict = {},
                 labelCounts : dict = {},
                 shannonEnt : float = 0.0,
                 retDataSet : list = []) -> None:
        self.classCount, self.labelCounts, self.shannonEnt, \
            self.retDataSet = classCount, labelCounts, \
                shannonEnt, retDataSet

    def majorityCnt(self, 
                    classList : list):
        for vote in classList:
            if vote not in self.classCount.keys():self.classCount[vote] = 0
            self.classCount[vote] += 1
        sortedClassCount = sorted(self.classCount.items(), 
                                  reverse=True)
        return sortedClassCount[0][0]


    # for calculting entropy
    def calcShannonEnt(self, 
                       dataSet : list) -> float:

        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in self.labelCounts.keys():
                self.labelCounts[currentLabel] = 0
                self.labelCounts[currentLabel] += 1

        for key in self.labelCounts:
            prob = float(self.labelCounts[key])/len(dataSet)
            self.shannonEnt -= float(self.labelCounts[key])/len(dataSet) * log(prob, 2)
        return self.shannonEnt

    def splitDataSet(self, 
                     dataSet : list, 
                     axis : int, 
                     value : int) -> list:

        for dat in dataSet:
            if dat[axis] == value:
                reducedFeatVec = dat[:axis]
                reducedFeatVec.extend(dat[axis+1:])
                self.retDataSet.append(reducedFeatVec)
        return self.retDataSet

    # choosing the best feature to split
    def chooseBestFeatureToSplit(self, 
                                 dataSet : list, 
                                 labels : list) -> int:
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain = -1
        bestFeature = 0
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy

            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i

        return bestFeature


    def createTree(self, 
                   dataSet : list, 
                   labels : list) -> dict:
        classList = [ex[-1] for ex in dataSet]
        if not len(classList):return

        if classList.count(classList[0]) == len(classList):return classList[0]
        if len(dataSet[0]) == 1:return self.majorityCnt(classList)

        featureVectorList = [r[:len(r)-1] for r in dataSet]
        bestFeat = self.chooseBestFeatureToSplit(featureVectorList, 
                                                 labels)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        del(labels[bestFeat])
        featValues = [ex[bestFeat] for ex in dataSet]
        for val in set(featValues):
            subLabels = labels[:]
            myTree[bestFeatLabel][val] = self.createTree(
                self.splitDataSet(dataSet, bestFeat, val), 
                                  subLabels)
        return myTree


def get_main():
    global _path, _labels
    df = pd.read_csv(_path)  # Reading from the data file
    df.replace("male", 0, inplace=True)
    df.replace("female", 1, inplace=True)
    df.replace('S', 0, inplace=True)
    df.replace('C', 1, inplace=True)
    df.replace('Q', 2, inplace=True)
    df['embarked'] = df['embarked'].fillna(1)
    data = df.astype(float).values.tolist()

    random.shuffle(data)

    custom_DTree = Do_Decision_Tree()
    print(custom_DTree.createTree(data, _labels))


if __name__ == "__main__":
    get_main()