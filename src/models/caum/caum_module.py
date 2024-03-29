from typing import Any, List, Tuple

import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch_geometric.utils import to_dense_batch
from pytorch_lightning import LightningModule
from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG
from torchmetrics import MeanMetric, MinMetric, MetricCollection

from src.datamodules.components.nemig_batch import NemigBatch
from src.models.caum.news_encoder import NewsEncoder
from src.models.caum.user_encoder import UserEncoder
from src.models.components.click_predictors import DotProduct
from src.metrics.diversity import Diversity
from src.metrics.personalization import Personalization


class CAUMModule(LightningModule):
    def __init__(
            self,
            pretrained_word_embeddings_path: str,
            pretrained_entity_embeddings_path: str,
            word_embedding_dim: int,
            entity_embedding_dim: int,
            num_categories: int,
            category_embedding_dim: int,
            news_vector_dim: int,
            user_vector_dim: int,
            num_attention_heads: int,
            num_filters: int,
            dense_att_hidden_dim1: int,
            dense_att_hidden_dim2: int,
            query_vector_dim: int,
            dropout_probability: float,
            num_polit_classes: int,
            num_sent_classes: int,
            optimizer: torch.optim.Optimizer
            ) -> None:
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # load pretrained word embeddings
        pretrained_word_embeddings = torch.from_numpy(np.load(self.hparams.pretrained_word_embeddings_path)).float()
        
        # load pretrained entity embeddings
        pretrained_entity_embeddings = torch.from_numpy(np.load(self.hparams.pretrained_entity_embeddings_path)).float()

        # model components
        self.news_encoder = NewsEncoder(
                pretrained_word_embeddings=pretrained_word_embeddings,
                pretrained_entity_embeddings=pretrained_entity_embeddings,
                word_embedding_dim=self.hparams.word_embedding_dim,
                entity_embedding_dim=self.hparams.entity_embedding_dim,
                num_attention_heads=self.hparams.num_attention_heads,
                query_vector_dim=self.hparams.query_vector_dim,
                num_categories=self.hparams.num_categories,
                category_embedding_dim=self.hparams.category_embedding_dim,
                news_vector_dim=self.hparams.news_vector_dim,
                dropout_probability=self.hparams.dropout_probability
                )

        # user encoder 
        self.user_encoder = UserEncoder(
               news_vector_dim=self.hparams.news_vector_dim,
               num_filters=self.hparams.num_filters,
               dense_att_hidden_dim1=self.hparams.dense_att_hidden_dim1,
               dense_att_hidden_dim2=self.hparams.dense_att_hidden_dim2,
               user_vector_dim=self.hparams.user_vector_dim,
               num_attention_heads=self.hparams.num_attention_heads,
               dropout_probability=self.hparams.dropout_probability
               )

        # click predictor
        self.click_predictor = DotProduct()
      
        self.criterion = CrossEntropyLoss()
        
        # metric objects for calculating and averaging performance across batches        
        metrics = MetricCollection({
            'auc': AUROC(task='binary', num_classes=2),
            'mrr': RetrievalMRR(),
            'ndcg@3': RetrievalNormalizedDCG(k=3),
            'ndcg@6': RetrievalNormalizedDCG(k=6)
            })
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        
        polit_div_metrics = MetricCollection({
            'polit_div@3': Diversity(num_classes=self.hparams.num_polit_classes, k=3),
            'polit_div@6': Diversity(num_classes=self.hparams.num_polit_classes, k=6) 
            })
        categ_div_metrics = MetricCollection({
            'categ_div@3': Diversity(num_classes=self.hparams.num_categories, k=3),
            'categ_div@6': Diversity(num_classes=self.hparams.num_categories, k=6) 
            })
        sent_div_metrics = MetricCollection({
            'sent_div@3': Diversity(num_classes=self.hparams.num_sent_classes, k=3),
            'sent_div@6': Diversity(num_classes=self.hparams.num_sent_classes, k=6) 
            })
        polit_pers_metrics = MetricCollection({
            'pol_pers@3': Personalization(num_classes=self.hparams.num_polit_classes, k=3),
            'pol_pers@6': Personalization(num_classes=self.hparams.num_polit_classes, k=6) 
            })
        categ_pers_metrics = MetricCollection({
            'categ_pers@3': Personalization(num_classes=self.hparams.num_categories, k=3),
            'categ_pers@6': Personalization(num_classes=self.hparams.num_categories, k=6) 
            })
        sent_pers_metrics = MetricCollection({
            'sent_pers@3': Personalization(num_classes=self.hparams.num_sent_classes, k=3),
            'sent_pers@6': Personalization(num_classes=self.hparams.num_sent_classes, k=6) 
            })
        self.test_polit_div_metrics = polit_div_metrics.clone(prefix='test/')
        self.test_categ_div_metrics = categ_div_metrics.clone(prefix='test/')
        self.test_sent_div_metrics = sent_div_metrics.clone(prefix='test/')
        self.test_polit_pers_metrics = polit_pers_metrics.clone(prefix='test/')
        self.test_categ_pers_metrics = categ_pers_metrics.clone(prefix='test/')
        self.test_sent_pers_metrics = sent_pers_metrics.clone(prefix='test/')

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()
    
    def forward(self, batch: NemigBatch) -> torch.Tensor:
        # encode user history
        clicked_news_vector = self.news_encoder(batch['x_hist'])
        clicked_news_vector_agg, mask_hist = to_dense_batch(clicked_news_vector, batch['batch_hist'])

        # encode candidate news
        candidate_news_vector = self.news_encoder(batch['x_cand'])
        candidate_news_vector_agg, _ = to_dense_batch(candidate_news_vector, batch['batch_cand'])
        
        # encode user
        scores = torch.zeros(candidate_news_vector_agg.shape[0], candidate_news_vector_agg.shape[1], device=self.device)
        scores = scores.transpose(1, 0)

        for i in range(candidate_news_vector_agg.shape[1]):
            cand_score = self.user_encoder(clicked_news_vector_agg, candidate_news_vector_agg[:, i, :])
            scores[i, :] = cand_score

        scores = scores.transpose(1, 0)
    
        return scores

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_loss_best.reset()

    def model_step(self, batch: NemigBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self(batch)
        y_true, mask_cand = to_dense_batch(batch['labels'], batch['batch_cand'])

        candidate_politic, _ = to_dense_batch(batch['x_cand']['politic'], batch['batch_cand'])
        candidate_categories, _ = to_dense_batch(batch['x_cand']['category'], batch['batch_cand'])
        candidate_sentiments, _ = to_dense_batch(batch['x_cand']['sentiment'], batch['batch_cand'])
        
        clicked_politic, mask_hist = to_dense_batch(batch['x_hist']['politic'], batch['batch_hist'])
        clicked_categories, _ = to_dense_batch(batch['x_hist']['category'], batch['batch_hist'])
        clicked_sentiments, _ = to_dense_batch(batch['x_hist']['sentiment'], batch['batch_hist'])
        
        loss = self.criterion(scores, y_true)
        
        # predictions, targets, indexes for metric computation
        preds = torch.cat([scores[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0).detach()
        targets = torch.cat([y_true[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0).long() 
        
        cand_news_size = torch.tensor([torch.where(mask_cand[n])[0].shape[0] for n in range(mask_cand.shape[0])])
        hist_news_size = torch.tensor([torch.where(mask_hist[n])[0].shape[0] for n in range(mask_hist.shape[0])])

        target_politic = torch.cat([candidate_politic[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0).long() 
        target_categories = torch.cat([candidate_categories[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0).long() 
        target_sentiments = torch.cat([candidate_sentiments[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0).long() 
        
        hist_politic = torch.cat([clicked_politic[n][mask_hist[n]] for n in range(mask_hist.shape[0])], dim=0).long() 
        hist_categories = torch.cat([clicked_categories[n][mask_hist[n]] for n in range(mask_hist.shape[0])], dim=0).long() 
        hist_sentiments = torch.cat([clicked_sentiments[n][mask_hist[n]] for n in range(mask_hist.shape[0])], dim=0).long() 
        
        return loss, preds, targets, cand_news_size, hist_news_size, target_politic, target_categories, target_sentiments, hist_politic, hist_categories, hist_sentiments 

    def training_step(self, batch: NemigBatch, batch_idx: int):
        loss, preds, targets, cand_news_size, _, _, _, _, _, _, _ = self.model_step(batch)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {
                "loss": loss, 
                "preds": preds, 
                "targets": targets, 
                "cand_news_size": cand_news_size
                }
    
    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        preds = torch.cat([o['preds'] for o in outputs])
        targets = torch.cat([o['targets'] for o in outputs])
        cand_news_size = torch.cat([o['cand_news_size'] for o in outputs])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        
        self.train_metrics(preds, targets, **{'indexes': indexes})
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch: NemigBatch, batch_idx: int):
        loss, preds, targets, cand_news_size, _, _, _, _, _, _, _  = self.model_step(batch)

        # update and log loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {
                "loss": loss, 
                "preds": preds, 
                "targets": targets, 
                "cand_news_size": cand_news_size
                }
    
    def validation_epoch_end(self, outputs: List[Any]):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True, logger=True, sync_dist=True)

        preds = torch.cat([o['preds'] for o in outputs])
        targets = torch.cat([o['targets'] for o in outputs])
        cand_news_size = torch.cat([o['cand_news_size'] for o in outputs])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        
        self.val_metrics(preds, targets, **{'indexes': indexes})
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: NemigBatch, batch_idx: int):
        loss, preds, targets, cand_news_size, hist_news_size, target_politic, target_categories, target_sentiments, hist_politic, hist_categories, hist_sentiments = self.model_step(batch)

        # update and log loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {
            "loss": loss, 
            "preds": preds, 
            "targets": targets, 
            "cand_news_size": cand_news_size,
            "hist_news_size": hist_news_size,
            "target_politic": target_politic,
            "target_categories": target_categories, 
            "target_sentiments": target_sentiments,
            "hist_politic": hist_politic,
            "hist_categories": hist_categories, 
            "hist_sentiments": hist_sentiments 
            }  
    
    def test_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([o['preds'] for o in outputs])
        targets = torch.cat([o['targets'] for o in outputs])
       
        cand_news_size = torch.cat([o['cand_news_size'] for o in outputs])
        hist_news_size = torch.cat([o['hist_news_size'] for o in outputs])
        
        target_politic = torch.cat([o['target_politic'] for o in outputs])
        target_categories = torch.cat([o['target_categories'] for o in outputs])
        target_sentiments = torch.cat([o['target_sentiments'] for o in outputs])
        
        hist_politic = torch.cat([o['hist_politic'] for o in outputs])
        hist_categories = torch.cat([o['hist_categories'] for o in outputs])
        hist_sentiments = torch.cat([o['hist_sentiments'] for o in outputs])
        
        cand_indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        hist_indexes = torch.arange(hist_news_size.shape[0]).repeat_interleave(hist_news_size)
        
        self.test_metrics(preds, targets, **{'indexes': cand_indexes})
        self.test_polit_div_metrics(preds, target_politic, cand_indexes)
        self.test_categ_div_metrics(preds, target_categories, cand_indexes)
        self.test_sent_div_metrics(preds, target_sentiments, cand_indexes)
        self.test_polit_pers_metrics(preds, target_politic, hist_politic, cand_indexes, hist_indexes)
        self.test_categ_pers_metrics(preds, target_categories, hist_categories, cand_indexes, hist_indexes)
        self.test_sent_pers_metrics(preds, target_sentiments, hist_sentiments, cand_indexes, hist_indexes)
        
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.test_polit_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.test_categ_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.test_sent_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.test_polit_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.test_categ_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.test_sent_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        
        return {"optimizer": optimizer}
