"""
    This is an example model to run on a DL AMI - for the original implementation,
    comments and insights, please check the appropriate sections in the evalRS repository:
    https://github.com/RecList/evalRS-CIKM-2022/tree/main/notebooks/merlin_tutorial.

    This model is built using Merlin. For more information about the open source framework,
    please check the project page: https://github.com/NVIDIA-Merlin/Merlin.
"""
import os
import pandas as pd
from gensim.models import KeyedVectors
from joblib import Parallel, delayed
import numpy as np
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora
import gensim
import time
import random
import nltk
import itertools
from collections import Counter
from tqdm import tqdm

from evaluation.EvalRSRunner import EvalRSRunner
from evaluation.EvalRSRunner import ChallengeDataset
from evaluation.EvalRSRecList import EvalRSRecList
from reclist.abstractions import RecModel

class MyModel(RecModel):

    def __init__(self, top_k: int=100,sample_num=40,cold_num=10000,seed=None,drop_num=2,
                 diversity_keeptop=12,diversity_history=10,worker=8,diversity_flag=False):
        super(MyModel, self).__init__()
        """
        :param top_k: numbers of recommendation to return for each user. Defaults to 20.
        """
        self.top_k = top_k
        self.n_sim_movie = 10
        self.ngram=2
        self.user_track_sample=sample_num
        self.cold_num=cold_num
        self.drop_num=drop_num
        self._dense_repr = KeyedVectors.load(os.path.join('/home/ubuntu/.cache/evalrs/evalrs_dataset','song2vec.wv'))
        self.dense_repr_dict={k:self._dense_repr[k] for k in self._dense_repr.index_to_key}
        self.diversity_keeptop=diversity_keeptop
        self.diversity_history=diversity_history
        self.worker=worker
        self.diversity_flag=diversity_flag
        self._random_state = int(time.time()) if not seed else seed
        print(f'ngram_random_state:{self._random_state}')
        random.seed(self._random_state)

    def train(self, train_df: pd.DataFrame, **kwargs):
        
        # let's put tracks in order so we can build those sentences
        df = train_df.sort_values('timestamp')
        self.hot_item=list(df.track_id.value_counts().index[:100])
        
        # we group by user and create sequences of tracks. 
        # each row in "track_id" will be a sequence of tracks
        p = df.groupby('user_id', sort=False)['track_id'].agg(list)
        
        # we now build "sentences" : sequences of tracks
        seq_data = p.values.tolist()

        sentences = [nltk.ngrams(seq, self.ngram) for seq in seq_data]
        phrase_freq = nltk.FreqDist(itertools.chain(*sentences))
        
        count = 0
        rec_rs = {}
        for phrase in tqdm(phrase_freq.most_common()):
            if phrase[1] > 0:  
                count += 1
                for i in range(1, self.ngram):
                    if rec_rs.get(phrase[0][0]) is None:
                        rec_rs.setdefault(phrase[0][0], [phrase[0][i]])
                    elif len(rec_rs.get(phrase[0][0]))<self.n_sim_movie:
                        rec_rs[phrase[0][0]].append(phrase[0][i])
        
        self.i2i_dict=rec_rs
        
        df = train_df.sort_values('user_track_count',ascending=False)
        p = df.groupby('user_id', sort=False)['track_id'].agg(list)
        seq_data = p.values.tolist()
        corpus_value_str=[]
        for line in seq_data:
            corpus_value_str.append([str(l) for l in line])
        dictionary = gensim.corpora.Dictionary(corpus_value_str)
        train_gb=train_df.groupby('user_id', sort=False)
        corpus=[]
        user_corpus_dict={}
        for uid in tqdm(train_df.user_id.unique()):
            tmp=[]
            d=train_gb.get_group(uid)
            for item,count in zip(d['track_id'],d['user_track_count']):
                tmp.append((dictionary.token2id[str(item)],count))
            corpus.append(tmp)
            user_corpus_dict[uid]=tmp
        user_tracks = pd.DataFrame(p)
        tf_idf_model = TfidfModel(corpus, normalize=True)
        
        user_tracks["track_id_sampled"] = pd.Series(user_tracks.index,index=user_tracks.index).apply(lambda x : [int(dictionary[a]) for a,b in sorted(tf_idf_model[user_corpus_dict[x]],key=lambda y:y[1],reverse=True)]) 

        self.mappings = user_tracks.T.to_dict()
        self.all_track_id=list(train_df.track_id.unique())
        
        #self.cold_artist_top5k=list(train_df.track_id.value_counts().index[-5000:])
        #self.cold_track_top1w=list(train_df[train_df.artist_id.isin(self.cold_artist_top5k)].track_id.value_counts().index[-5000:])
       # self.cold_track_top1w=list(train_df.track_id.value_counts().index[-self.cold_num:])
        self.cold_track_top1w=list(train_df.groupby('track_id')[['user_track_count']].sum().reset_index().sort_values('user_track_count')[:self.cold_num].track_id.values)
    def cosine_sim(self,u: np.array, v: np.array) -> np.array:
        return np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    def diversity_rerank(self,uid_list):
        rerank_list=[]
        for uid in (uid_list):
            rec_list=self.uid_predictions[uid]
            keep_top=self.diversity_keeptop
            rerank_items=list(rec_list[:keep_top])
            user_hist=self.mappings[uid]["track_id_sampled"][:self.diversity_history]
            
            gt_vectors = np.sum(np.array([self.dense_repr_dict[i] for i in user_hist]) , axis=0) / len(user_hist)
            less_wrong_gt_vectors = np.sum(np.array([self.dense_repr_dict[i] for i in user_hist]) , axis=0) / len(user_hist)
            distance_to_gt = 1-self.cosine_sim(less_wrong_gt_vectors,np.array([self.dense_repr_dict[i] for i in rec_list] ))
            best=sorted(zip(distance_to_gt,rec_list),reverse=False)[0][1]
            rerank_items=[best]+[i for i in rec_list[:keep_top] if i not in [best]]
    
            for i in range(20-keep_top):
                tmp_item_list=[]
                tmp_score=[]
                for item in (set(rec_list)-set(rerank_items)):
                    tmp_rerank_items=rerank_items.copy()
                    tmp_rerank_items.append(item)
                    pred_vectors = np.array([self.dense_repr_dict[i] for i in tmp_rerank_items])
                    mean_pred_vector = np.sum(pred_vectors, axis=0) / len(pred_vectors)
                    distance_to_mean = 1-self.cosine_sim(mean_pred_vector, pred_vectors)   
                    mean_distance = np.sum(distance_to_mean, axis=0) /len(pred_vectors)
                    bias_distance = 1-self.cosine_sim(mean_pred_vector, gt_vectors)
                    score=float((0.3*mean_distance-0.7*bias_distance))
                    tmp_item_list.append(item)
                    tmp_score.append(score)
                best=sorted(zip(tmp_score,tmp_item_list),reverse=True)[0][1]
                #print(sorted(zip(tmp_score,tmp_item_list),reverse=True))
                rerank_items.append(best)
            rerank_items=rerank_items+[i for i in rec_list if i not in set(rerank_items)]
            rerank_list.append(rerank_items)
        return rerank_list
    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:

        user_ids = user_ids.copy()
        predictions = []
        
        # probably not the fastest way to do this
        for user in (user_ids["user_id"]):
          
          	# for each user we get their sample tracks
            user_tracks = self.mappings[user]["track_id_sampled"][:self.user_track_sample]
            tmp_rec=[]
            user_predictions=[]
            for item in user_tracks:
                r_item = self.i2i_dict.get(item, [])
                tmp_rec.extend(r_item)
            for item,count in Counter(tmp_rec).most_common(len(user_tracks) + self.top_k):     
                user_predictions.append(item)

            user_predictions = list(filter(lambda x: x not in 
                                           user_tracks[:self.drop_num], user_predictions))[0:self.top_k]
            user_predictions=user_predictions[:2] +user_predictions[10:12] +user_predictions[20:22]+user_predictions[30:32]+user_predictions[40:42]+user_predictions[50:52]

            if len(user_predictions)<100:
                random_track=random.choices(self.cold_track_top1w, k=110)
                for i in random_track:
                    if i not in set(user_predictions):
                        user_predictions.append(i)
                    if len(user_predictions)==100:
                        break
            
            # append to the return list
            predictions.append(user_predictions)
        all_sorted_predictions=[]    
        if self.diversity_flag: 
            self.uid_predictions=dict(zip(list(user_ids["user_id"].values),predictions))
            print('diversity_rerank')
            results = Parallel(n_jobs=self.worker, verbose=1,backend='multiprocessing' )(
                    delayed(self.diversity_rerank)(num) for num in
                    np.array_split(np.array(list(user_ids["user_id"].values)),2*self.worker))
            for s in results:
                all_sorted_predictions=all_sorted_predictions+s
        else:
            all_sorted_predictions=predictions
            
        users = user_ids["user_id"].values.reshape(-1, 1)   
        predictions = np.concatenate([users, np.array(all_sorted_predictions)], axis=1)
        return pd.DataFrame(predictions, columns=['user_id', *[str(i) for i in range(self.top_k)]]).set_index('user_id') 
