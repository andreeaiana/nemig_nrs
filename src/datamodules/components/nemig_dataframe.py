from ast import literal_eval
from typing import List, Dict, Tuple, Any, Union, Optional

import re
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset

from src import utils

tqdm.pandas()

log = utils.get_pylogger(__name__)


class NemigDataFrame(Dataset):
    def __init__(
            self,
            seed: int,
            lang: str,
            kg_type: Optional[str],
            data_dir: str,
            word_embeddings_dirname: str,
            word_embeddings_fpath: str,
            entity_embeddings_filename: str,
            id2index_filenames: Dict[str, str],
            word_embedding_dim: int,
            entity_embedding_dim: int,
            train: bool,
            validation: bool,
            ) -> None:

        super().__init__()
       
        self.seed = seed
        self.lang = lang
        self.kg_type = kg_type

        self.data_dir = data_dir

        self.word_embeddings_dirname = word_embeddings_dirname
        self.entity_embeddings_filename = entity_embeddings_filename
        self.id2index_filenames = id2index_filenames
      
        self.word_embedding_dim = word_embedding_dim
        self.entity_embedding_dim = entity_embedding_dim

        self.validation = validation
        
        if train:
            self.data_split = 'train'
        else:
            self.data_split = 'dev' # test

        self.word_embeddings_fpath = os.path.join(self.data_dir, self.word_embeddings_dirname, self.lang, word_embeddings_fpath)
        self.dst_dir = os.path.join(self.data_dir, self.lang)
        self.categ2index_fpath = os.path.join(self.dst_dir, self.id2index_filenames['categ2index'])
        self.politic2index_fpath = os.path.join(self.dst_dir, self.id2index_filenames['politic2index'])
        self.sentiment2index_fpath = os.path.join(self.dst_dir, self.id2index_filenames['sentiment2index'])
        self.entity_embeddings_fpath = os.path.join(self.dst_dir, 'kg', self.kg_type, self.entity_embeddings_filename)
        self.word2index_fpath = os.path.join(self.dst_dir, self.id2index_filenames['word2index'])
        self.entity2index_fpath = os.path.join(self.dst_dir, self.id2index_filenames['entity2index'])
        self.uid2index_fpath = os.path.join(self.dst_dir, self.id2index_filenames['uid2index'])

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.news, self.behaviors = self.load_data()
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        user_bhv = self.behaviors.iloc[idx]

        history = user_bhv['history']
        candidates = user_bhv['candidates']
        labels = user_bhv['labels']

        history = self.news.loc[history]
        candidates = self.news.loc[candidates]
        labels = np.array(labels)

        return history, candidates, labels

    def __len__(self) -> int:
        return len(self.behaviors)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Loads the parsed news and user behaviors.   

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                Tuple of news and behaviors datasets.
        """
        news = self._load_news()
        log.info(f'News data size: {len(news)}')

        behaviors = self._load_behaviors()
        log.info(f'Behaviors data size for data split {self.data_split}, validation={self.validation}: {len(behaviors)}')

        return news, behaviors

    def _load_news(self):
        """ Loads the parsed news. If not already parsed, loads and preprocesses the raw news data.  

        Args:
            news (pd.DataFrame): Dataframe of news articles.

        Returns:
            pd.DataFrame: Parsed news data. 
        """
        parsed_news_file = os.path.join(self.dst_dir, 'parsed_news_' + str(self.kg_type) + '.tsv')
        transformed_entity_embeddings_filename = 'pretrained_entity_embeddings_' + str(self.kg_type) 

        if self._check_integrity(parsed_news_file):
            # news data already parsed
            log.info(f'News data already parsed. Loading from {parsed_news_file}.')
            news = pd.read_table(
                    filepath_or_buffer=parsed_news_file,
                    converters={
                        attribute: literal_eval 
                        for attribute in ['title', 'abstract', 'title_entities', 'abstract_entities']
                        }
                    )

        else:
            log.info(f'News data not parsed. Loading and parsing raw data.')
            columns_names=['nid', 'title', 'abstract', 'title_entities', 'abstract_entities', 'category', 'sentiment_label', 'political_class']
            news = pd.read_table(
                    filepath_or_buffer=os.path.join(self.dst_dir, 'news.tsv'),
                    header=None,
                    names=columns_names,
                    usecols=range(len(columns_names))
                    )
            
            # replace missing values
            news['abstract'].fillna('', inplace=True)
            news['title_entities'].fillna('[]', inplace=True)
            news['abstract_entities'].fillna('[]', inplace=True)
            
            news['title_entities'] = news['title_entities'].apply(lambda x: literal_eval(x))
            news['abstract_entities'] = news['abstract_entities'].apply(lambda x: literal_eval(x))
            
            # tokenize text
            news['title'] = news['title'].progress_apply(self.word_tokenize)
            news['abstract'] = news['abstract'].progress_apply(self.word_tokenize)

            transformed_word_embeddings_filename = 'pretrained_word_embeddings'

            if self.data_split == 'train':
                # categ2index map
                log.info('Constructing categ2index map.')
                news_category = news['category'].drop_duplicates().reset_index(drop=True)
                categ2index = {v: k+1 for k, v in news_category.to_dict().items()}
                log.info(f'Saving categ2index map of size {len(categ2index)} in {self.categ2index_fpath}')
                self._to_tsv(df=pd.DataFrame(categ2index.items(), columns=['category', 'index']),
                             fpath=self.categ2index_fpath)
                
                # politic2index map
                log.info('Constructing politic2index map.')
                news_pol_class = news['political_class'].drop_duplicates().reset_index(drop=True)
                politic2index = {v: k+1 for k, v in news_pol_class.to_dict().items()}
                log.info(f'Saving politic2index map of size {len(politic2index)} in {self.politic2index_fpath}')
                self._to_tsv(df=pd.DataFrame(politic2index.items(), columns=['political_class', 'index']),
                             fpath=self.politic2index_fpath)
                
                # sentiment2index map
                log.info('Constructing sentiment2index map.')
                news_sentiment = news['sentiment_label'].drop_duplicates().reset_index(drop=True)
                sentiment2index = {v: k+1 for k, v in news_sentiment.to_dict().items()}
                log.info(f'Saving sentiment2index map of size {len(sentiment2index)} in {self.sentiment2index_fpath}')
                self._to_tsv(df=pd.DataFrame(sentiment2index.items(), columns=['sentiment_label', 'index']),
                             fpath=self.sentiment2index_fpath)

                # construct word2index map
                log.info('Constructing word2index map.')
                word_cnt = Counter() 
                for idx in tqdm(news.index.tolist()):
                    word_cnt.update(news.loc[idx]['title'])
                    word_cnt.update(news.loc[idx]['abstract'])
                word2index = {k: v+1 for k, v in zip(word_cnt, range(len(word_cnt)))}
                log.info(f'Saving word2index map of size {len(word2index)} in {self.word2index_fpath}')
                self._to_tsv(df=pd.DataFrame(word2index.items(), columns=['word', 'index']),
                             fpath=self.word2index_fpath)
                    
                # construct word embedding matrix
                log.info('Constructing word embedding matrix.')
                self._generate_word_embeddings(
                        word2index = word2index,
                        embeddings_fpath = self.word_embeddings_fpath,
                        embedding_dim=self.word_embedding_dim,
                        transformed_embeddings_filename = transformed_word_embeddings_filename)

                log.info('Constructing entity2index map.')
                self.entity2freq = {}
                self._count_entity_freq(news['title_entities'])
                self._count_entity_freq(news['abstract_entities'])

                self.entity2index = {}
                for entity, _ in self.entity2freq.items():
                    self.entity2index[entity] = len(self.entity2index) + 1
                
                log.info(f'Saving entity2index map of size {len(self.entity2index)} in {self.entity2index_fpath}')
                self._to_tsv(
                        df=pd.DataFrame(self.entity2index.items(), columns=['entity', 'index']), 
                        fpath=self.entity2index_fpath
                        )
                
                # construct entity embedding matrix
                log.info('Constructing embedding embedding matrix.')
                self._generate_word_embeddings(
                        word2index = self.entity2index,
                        embeddings_fpath = self.entity_embeddings_fpath,
                        embedding_dim=self.entity_embedding_dim,
                        transformed_embeddings_filename = transformed_entity_embeddings_filename)

            else:
                log.info('Loading indices maps.')
                # load categ2index map
                categ2index = self._load_idx_map_as_dict(self.categ2index_fpath)
                
                # load politic2index map
                politic2index = self._load_idx_map_as_dict(self.politic2index_fpath)

                # load sentiment2index map
                sentiment2index = self._load_idx_map_as_dict(self.sentiment2index_fpath)
               
                # load word2index map
                word2index = self._load_idx_map_as_dict(self.word2index_fpath)
               
                # load entity2index map
                self.entity2index = self._load_idx_map_as_dict(self.entity2index_fpath)

                # construct entity embedding matrix
                log.info('Constructing word embedding matrix.')
                self._generate_word_embeddings(
                        word2index = word2index,
                        embeddings_fpath = self.word_embeddings_fpath,
                        embedding_dim=self.word_embedding_dim,
                        transformed_embeddings_filename = transformed_word_embeddings_filename)

                # construct entity embedding matrix
                log.info('Constructing embedding embedding matrix.')
                self._generate_word_embeddings(
                        word2index = self.entity2index,
                        embeddings_fpath = self.entity_embeddings_fpath,
                        embedding_dim=self.entity_embedding_dim,
                        transformed_embeddings_filename = transformed_entity_embeddings_filename)

            # parse news
            log.info('Parsing news.')
            news['category'] = news['category'].progress_apply(lambda x: categ2index.get(x, 0))
            news['political_class'] = news['political_class'].progress_apply(lambda x: politic2index.get(x, 0))
            news['sentiment_label'] = news['sentiment_label'].progress_apply(lambda x: sentiment2index.get(x, 0))

            news['title'] = news['title'].progress_apply(lambda tokenized_title: [word2index.get(x, 0) for x in tokenized_title])
            news['abstract'] = news['abstract'].progress_apply(lambda tokenized_abstract: [word2index.get(x, 0) for x in tokenized_abstract])
            
            news['title_entities'] = news['title_entities'].progress_apply(lambda row: self._filter_entities(row))
            news['abstract_entities'] = news['abstract_entities'].progress_apply(lambda row: self._filter_entities(row))

            # cache parsed data
            log.info(f'Caching parsed news of size {len(news)} to {parsed_news_file}.')
            self._to_tsv(news, parsed_news_file)

        news = news.set_index('nid', drop=True)

        return news

    def _load_behaviors(self) -> pd.DataFrame:
        """ Loads the parsed user behaviors. If not already parsed, loads and preprocesses the raw behavior data. 

        Returns:
            pd.DataFrame: Parsed user behavior data. 
        """
        file_prefix = ''
        if self.data_split == 'train':
            file_prefix = 'train_' if not self.validation else 'val_'
        else:
            file_prefix = "test_"
        parsed_behaviors_file = os.path.join(self.dst_dir, file_prefix + 'parsed_behaviors_' + str(self.seed) + '.tsv')

        if self._check_integrity(parsed_behaviors_file):
            # behaviors data already parsed
            log.info(f'User behaviors data already parsed. Loading from {parsed_behaviors_file}.')
            behaviors = pd.read_table(
                    filepath_or_buffer=parsed_behaviors_file,
                    converters={
                        'history': lambda x: x.strip("[]").replace("'","").split(", "),
                        'candidates': lambda x: x.strip("[]").replace("'","").split(", "),
                        'labels': lambda x: list(map(int, x.strip("[]").split(", "))),
                        }
                    )
        else:
            log.info(f'User behaviors data not parsed. Loading and parsing raw data.')
            columns_names=['impid', 'uid', 'history', 'impressions']
            behaviors = pd.read_table(
                    filepath_or_buffer=os.path.join(self.dst_dir, 'behaviors.tsv'),
                    header=None,
                    names=columns_names,
                    usecols=range(len(columns_names))
                    )

            # parse behaviors
            log.info('Parsing behaviors.')
            behaviors['history'] = behaviors['history'].fillna('').str.split()
            behaviors = behaviors[behaviors['impressions'].isna()==False]
            behaviors['impressions'] = behaviors['impressions'].str.split()
            behaviors['candidates'] = behaviors['impressions'].apply(
                    lambda x: [impression.split("-")[0] for impression in x ])
            behaviors['labels'] = behaviors['impressions'].apply(
                    lambda x: [int(impression.split("-")[1]) for impression in x ])
            behaviors = behaviors.drop(columns=['impressions'])

            # drop interactions of users without history 
            count_interactions = len(behaviors)
            behaviors = behaviors[behaviors['history'].apply(len) > 0]
            dropped_interactions = count_interactions - len(behaviors)
            log.info(f'Removed {dropped_interactions} ({dropped_interactions/count_interactions}%) interactions without user history.')
            
            behaviors = behaviors.reset_index(drop=True)

            log.info('Splitting behavior data into train and validation sets.')
            train_behaviors = behaviors.sample(frac=0.7, random_state=self.seed)
            remaining = behaviors.drop(train_behaviors.index)
            val_behaviors = remaining.sample(frac=0.1, random_state=self.seed)
            test_behaviors = remaining.drop(val_behaviors.index)
            log.info(f'User behaviors: {len(train_behaviors)} train, {len(val_behaviors)} val, {len(test_behaviors)} test.')

            if self.data_split == 'train':
                if not self.validation:
                    behaviors = train_behaviors

                    # compute uid2index map
                    log.info('Constructing uid2index map.')
                    uid2index = {}
                    for idx in tqdm(behaviors.index.tolist()):
                        uid = behaviors.loc[idx]['uid']
                        if uid not in uid2index:
                            uid2index[uid] = len(uid2index) + 1

                    log.info(f'Saving uid2index map of size {len(uid2index)} in {self.uid2index_fpath}')
                    self._to_tsv(df = pd.DataFrame(uid2index.items(), columns=['uid', 'index']),
                                 fpath = self.uid2index_fpath)

                else:     
                    behaviors = val_behaviors

                    # load uid2index map
                    log.info('Loading uid2index map.')
                    uid2index = self._load_idx_map_as_dict(self.uid2index_fpath)
           
            else:
                behaviors = test_behaviors

                # load uid2index map
                log.info('Loading uid2index map.')
                uid2index = self._load_idx_map_as_dict(self.uid2index_fpath)
           
            # map uid to index
            log.info('Mapping uid to index.')
            behaviors['user'] = behaviors['uid'].apply(lambda x: uid2index.get(x, 0))
            behaviors = behaviors[['user', 'history', 'candidates', 'labels']]

            # cache processed data
            log.info(f'Caching parsed behaviors of size {len(behaviors)} to {parsed_behaviors_file}.')
            self._to_tsv(behaviors, parsed_behaviors_file)

        return behaviors

    def word_tokenize(self, sentence: str) -> List[str]:
        """Splits a sentence into word list using regex.

        Args:
            sentence (str): input sentence

        Returns:
            list: word list
        """
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sentence, str):
            return pat.findall(sentence.lower())
        else:
            return []
    
    def _generate_word_embeddings(self, word2index: Dict[str, int], embeddings_fpath: str, embedding_dim: int, transformed_embeddings_filename: Union[str, None]) -> None:
        """ Loads pretrained embeddings for the words (or entities) in word_dict.

        Args:
            word2index (Dict[str, int]): word dictionary
            embeddings_fpath (str): the filepath of the embeddings to be loaded
            ebedding_dim (int): dimensionality of embeddings
            transformed_embeddings_filename (str): the name of the transformed embeddings file
        """

        embedding_matrix = np.random.normal(size=(len(word2index) + 1, embedding_dim))
        exist_word = set()

        with open(embeddings_fpath, "r") as f:
            for line in tqdm(f):
                linesplit = line.split(" ")
                word = line[0]
                if len(word) != 0:
                    if word in word2index:
                        embedding_matrix[word2index[word]] = np.asarray(list(map(float, linesplit[1:])))
                        exist_word.add(word)
        
        log.info(f'Rate of word missed in pretrained embedding: {(len(exist_word)/len(word2index))}.')

        fpath = os.path.join(self.dst_dir, transformed_embeddings_filename)
        if not self._check_integrity(fpath):
            log.info(f'Saving word embeddings in {fpath}')
            np.save(fpath, embedding_matrix, allow_pickle=True)
    
    def _count_entity_freq(self, data: pd.Series) -> None:
        for row in tqdm(data):
            for entity in row:
                if entity['WikidataId'] not in self.entity2freq:
                    self.entity2freq[entity['WikidataId']] = 1
                else:
                    self.entity2freq[entity['WikidataId']] += 1

    def _filter_entities(self, data: pd.Series) -> List[int]:
        filtered_entities = []
        for entity in data:
            if entity['WikidataId'] in self.entity2index:
                filtered_entities.append(self.entity2index[entity['WikidataId']])
        return filtered_entities

    def _to_tsv(self, df: pd.DataFrame, fpath: str) -> None:
        df.to_csv(fpath, sep='\t', index=False)

    def _load_idx_map_as_dict(self, fpath: str) -> Dict[str, int]:
        idx_map_dict = dict(pd.read_table(fpath).values.tolist())
        return idx_map_dict

    def _check_exists(self) -> bool:
        return os.path.isdir(self.dst_dir) and os.listdir(self.dst_dir)

    def _check_integrity(self, fpath: str) -> bool:
        if not os.path.isfile(fpath):
            return False
        return True

