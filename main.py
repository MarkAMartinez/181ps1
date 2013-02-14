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


# Part 2
def xvalidate(dataset, nfolds):
  # training size: 80 | validation size: 10 | test size: 10
  datasz = len(dataset.examples)/2
  testsz = datasz/nfolds
  trainsz = datasz - testsz - testsz
  rawscore = prunedscore = 0

  # n-folds
  for i in range(nfolds):
    # partitions
    trainset = DataSet(examples = dataset.examples[i*testsz : i*testsz + trainsz], 
                       values = dataset.values)
    validationset = DataSet(examples = dataset.examples[i*testsz + trainsz: i*testsz + trainsz + testsz], 
                            values = dataset.values)
    testset = DataSet(examples = dataset.examples[i*testsz + trainsz + testsz : i*testsz + datasz], 
                      values = dataset.values)

    # learn, prune trees 
    dtree = learn(trainset)
    rawscore += scoretree(dtree, testset)

    pruned = bup_prune(dtree, dtree, validationset, validationset)
    prunedscore += scoretree(pruned, testset)

  print rawscore / float(nfolds), prunedscore / float(nfolds)

# tests a decision tree on testset (assume one copy of dataset examples)
def scoretree(dtree, dataset):
  score = 0
  for i in range(len(dataset.examples)):
    if dtree.predict(dataset.examples[i]) == dataset.examples[i].attrs[dataset.target]:
      score += 1
  return score / float(len(dataset.examples))

# returns a dataset with only elements with the specified attribute = val
def subset(dataset, attr, val): 
  newdata = copy.deepcopy(dataset)
  for i in newdata.examples:
    if i.attrs[attr] != val:
      newdata.examples.remove(i)
  return newdata

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
                                         curdata, validset)

    # score this tree
    oldscore = scoretree(tree, validset)

    # change testnode into a branch
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

    # yay
    xvalidate(dataset, 10)

main()


	
