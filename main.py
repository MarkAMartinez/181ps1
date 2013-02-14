# main.py
# -------
# Jimmy Sun

from dtree import *
import sys

class Globals:
	noisyFlag = False
	pruneFlag = False
	valSetSize = 0
	dataset = None

# pretty float formatting
class prettyfloat(float):
  def __repr__(self):
    return "%0.2f" % self

##Classify
#---------
def classify(decisionTree, example):
	return decisionTree.predict(example)

##Learn
#-------
def learn(dataset):
	learner = DecisionTreeLearner()
	learner.train(dataset)
	return learner.dt

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
	parseArgs([ 'main.py', '-n', '-p', 5 ]) = { '-n':True, '-p':5 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
	if args[i][0] == '-':
	  args_map[args[i]] = True
	  curkey = args[i]
	else:
	  assert curkey
	  args_map[curkey] = args[i]
	  curkey = None
  return args_map

def validateInput(args):
	args_map = parseArgs(args)
	valSetSize = 0
	noisyFlag = False
	pruneFlag = False
	boostRounds = -1
	maxDepth = -1
	if '-n' in args_map:
	  noisyFlag = True
	if '-p' in args_map:
	  pruneFlag = True
	  valSetSize = int(args_map['-p'])
	if '-d' in args_map:
	  maxDepth = int(args_map['-d'])
	if '-b' in args_map:
	  boostRounds = int(args_map['-b'])
	return [noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds]


# Part 2a
def xvalidate(dataset, nfolds):
  datasz = len(dataset.examples)/2
  testsz = datasz/nfolds
  trainsz = datasz - testsz
  testscore = trainscore = 0

  # n-folds
  for i in range(nfolds):
    # partitions
    trainset = DataSet(examples = dataset.examples[i*testsz : i*testsz + trainsz], 
                       values = dataset.values)
    testset = DataSet(examples = dataset.examples[i*testsz + trainsz : i*testsz + datasz], 
                      values = dataset.values)

    # learn, prune trees 
    dtree = learn(trainset)
    testscore += scoretree(dtree, testset)
    trainscore += scoretree(dtree, trainset)

  print "Training score : ", trainscore / float(nfolds)
  print "Test score     : ", testscore / float(nfolds)

# tests a decision tree on testset (assume one copy of dataset examples)
def scoretree(dtree, dataset):
  score = 0
  for i in range(len(dataset.examples)):
    if dtree.predict(dataset.examples[i]) == dataset.examples[i].attrs[dataset.target]:
      score += 1
  return score / float(len(dataset.examples))

# returns a dataset with only elements with the specified attribute = val
def subset(dataset, attr, val): 
  newex = []
  for i in dataset.examples:
    if i.attrs[attr] == val:
      newex.append(i)
  try:
    newdataset = DataSet(examples = newex, values = dataset.values)
    return newdataset
  except Exception:
    pass
  return None

# bottom-up post pruning. NOTE: DESTROYS 'tree' (actually prunes it)
def bup_prune(tree, testnode, curdata, validset):
  # at leaf just returns the leaf
  if testnode.nodetype == DecisionTree.LEAF:
    return testnode
  else:
    # recursively prune children
    for key in testnode.branches:
      subvalset = subset(curdata, testnode.attr, key)
      testnode.branches[key] = bup_prune(tree, testnode.branches[key], 
                                         subvalset, validset)

    # score this tree
    oldscore = scoretree(tree, validset)

    # change testnode into a leaf (but keeps subtrees there)
    testnode.nodetype = DecisionTree.LEAF
    testnode.classification = find_top_class(curdata, testnode.attr)

    # score this pruning
    newscore = scoretree(tree, validset)

    # reset node to be a branch if pruned score is worse
    if newscore < oldscore:
      testnode.nodetype = DecisionTree.NODE
    return testnode

# find majority class of validation set with attribute atrr
def find_top_class(dataset, attr):
  attrlist = []
  for i in dataset.examples:
    attrlist.append(i.attrs[attr])
  return max(set(attrlist), key = attrlist.count)


# 2b
def prune_xvalidate(dataset, nfolds):
  testscores = []
  trainscores = []

  # keep test size constant at 10, so trainsz goes from 10 to 89
  for i in range(1,81):
    datasz = len(dataset.examples)/2
    validsz = i
    testsz = datasz/nfolds
    trainsz = datasz - validsz - testsz
    trainscore = testscore = 0

    # partition into TRAIN | TEST | VALID
    for j in range(nfolds):
      trainset = DataSet(examples = dataset.examples[j*testsz : j*testsz + trainsz], 
                         values = dataset.values)
      testset = DataSet(examples = dataset.examples[j*testsz + trainsz : j*testsz + trainsz + testsz],
                        values = dataset.values)
      validset = DataSet(examples = dataset.examples[j*testsz + datasz - validsz : j*testsz + datasz],
                         values = dataset.values)
      dtree = learn(trainset)
      bup_prune(dtree, dtree, trainset, validset)
      trainscore += scoretree(dtree, trainset)
      testscore += scoretree(dtree, testset)

    trainscores.append(trainscore / float(nfolds))
    testscores.append(testscore / float(nfolds))

  print "Training : ", map(prettyfloat, trainscores)
  print "Testing  : ", map(prettyfloat, testscores)

def main():
    arguments = validateInput(sys.argv)
    noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds = arguments
    print noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds

    # Read in the data file
    if noisyFlag:
		  f = open("noisy.csv")
    else:
		  f = open("data.csv")

    data = parse_csv(f.read(), " ")
    dataset = DataSet(data)
	
    # Copy the dataset so we have two copies of it
    examples = dataset.examples[:]

    dataset.examples.extend(examples)
    dataset.max_depth = maxDepth
    if boostRounds != -1:
      dataset.use_boosting = True
      dataset.num_rounds = boostRounds

    # yay function
    xvalidate(dataset, 10)
    prune_xvalidate(dataset, 10)
    

main()


	
