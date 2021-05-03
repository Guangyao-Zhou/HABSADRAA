# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC
import tensorflow as tf
import lcrModel
import lcrModelInverse
import lcrModelAlt
import cabascModel
import svmModel
from OntologyReasoner import OntReasoner
from loadData import *
import lcrDoubleRAA
import lcrinvmodel2
import lcrnewmodel2
#import parameter configuration and data paths
from config import *
#import modules
import numpy as np
import sys

# main function
def main(_):
    loadData = False
    useOntology = False
    runCABASC = False
    runLCRROT = False
    runLCRROTINVERSE = False
    runLCRROTALT = False
    runSVM = False
    runlcrDoubleRAA = True
    runINVMULTIHOP1 = False
    runLCRNEWMODEL2 = False
    weightanalysis = False
    
    #determine if backupmethod is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM or runlcrDoubleRAA or runINVMULTIHOP1:
        backup = True
    else:
        backup = False
    
    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)
    print(test_size)
    remaining_size = 250
    accuracyOnt = 0.87

    if useOntology == True:
        print('Starting Ontology Reasoner')
        Ontology = OntReasoner()
        #out of sample accuracy
        accuracyOnt, remaining_size = Ontology.run(backup,FLAGS.test_path, runSVM)
        #in sample accuracy
        Ontology = OntReasoner()
        accuracyInSampleOnt, remaining_size = Ontology.run(backup,FLAGS.train_path, runSVM)
        if runSVM == True:
            test = FLAGS.remaining_svm_test_path
        else:
            test = FLAGS.remaining_test_path
        print('train acc = {:.4f}, test acc={:.4f}, remaining size={}'.format(accuracyOnt, accuracyOnt, remaining_size))
    else:
        if runSVM == True:
            test = FLAGS.test_svm_path
        else:
            test = FLAGS.test_path

    # LCR-Rot model
    if runLCRROT == True:
        _, pred1, fw1, bw1, tl1, tr1, sent, target, true = lcrModel.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
        tf.reset_default_graph()

    # LCR-Rot-inv model
    if runLCRROTINVERSE == True:
        lcrModelInverse.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
        tf.reset_default_graph()

    # LCR-Rot-hop model
    if runLCRROTALT == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
        tf.reset_default_graph()

    if runlcrDoubleRAA == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrDoubleRAA.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
       tf.reset_default_graph()
    
    if runINVMULTIHOP1 == True:   
       _, pred2, fw2, bw2, tl2, tr2 = lcrinvmodel2.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
       tf.reset_default_graph()
    
    if runLCRNEWMODEL2 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrnewmodel2.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
        tf.reset_default_graph()
    
    # BoW model
    if runSVM == True:
        svmModel.main(FLAGS.train_svm_path,test, accuracyOnt, test_size, remaining_size)
   
    print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
