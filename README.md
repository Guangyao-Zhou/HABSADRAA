# HABSADRA
We introduce a new model named "A Hybrid Solution for Aspect-Based Sentiment Analysis using Double Rotatory Attention model, which is capable of extracting these opinions and predicting the sentiment scores in aspect-level sentiment mining. In our two-step approach, a lexicalised domain ontology is firstly applied for sentiment classification. If the result is inconclusive from the first step, the backup model double rotatory attention mechanism is applied, which utilises deep contextual word embeddings to better capture the (multi-)word semantics in the given text. This study contributes to the current research by introducing novel repetition and rotatory structures to refine the attention mechanism. It is shown that our model outperforms state-of-the-art methods on the datasets of SemEval 2015 and SemEval 2016.
## Software
The HABSADRAA source code: https://github.com/ofwallaart/HAABSA and https://github.com/mtrusca/HAABSA_PLUS_PLUS need to be installed. Then, 
-Update the config.py, main.py.

-Add files:
 - Double rotatory attention Adjustment model with Type 1:
   - lcrDoubleRAA
 - Double rotatory attention Adjustment model with Type 2:
   - lcrDoubleRAAtype2

The training and testing databases are SemEval 2015 and SemEval 2016. The files are available for BERT word emebddings.
## Word embeddings
 - BERT word embeddings (SemEval 2015): https://drive.google.com/file/d/1-P1LjDfwPhlt3UZhFIcdLQyEHFuorokx/view?usp=sharing
 - BERT word embeddings (SemEval 2016): https://drive.google.com/file/d/1eOc0pgbjGA-JVIx4jdA3m1xeYaf0xsx2/view?usp=sharing
## Reference ##
- Zheng, S. and Xia, R. (2018). Left-center-right separated neural network for aspect-based sentiment analysis with rotatory attention. arXiv preprint arXiv:1802.00892.
- Schouten, K. and Frasincar, F. (2018). Ontology-driven sentiment analysis of product and service aspects. In Proceedings of the 15th Extended Semantic Web Conference (ESWC 2018). Springer. To appear
- M. M. Tru¸scˇa, D. Wassenberg, F. Frasincar, and R. Dekker, “A hybrid approach for aspectbased sentiment analysis using deep contextual word embeddings and hierarchical attention,” in International Conference on Web Engineering, Springer, 2020, pp. 365–380.
