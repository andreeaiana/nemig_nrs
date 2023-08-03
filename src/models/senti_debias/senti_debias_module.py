# Code adapted from https://github.com/wuch15/Sentiment-debiasing/blob/main/model.py

from typing import List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torch_geometric.utils import to_dense_batch
from torchmetrics import MetricCollection, MaxMetric
from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG

from src.datamodules.components.nemig_batch import NemigBatch
from src.models.senti_debias.user_encoder import UserEncoder
from src.models.senti_debias.news_encoder import NewsEncoder
from src.models.components.click_predictors import DotProduct
from src.metrics.diversity import Diversity
from src.metrics.personalization import Personalization


class SentimentEncoder(nn.Module):
    def __init__(
            self,
            num_sent_classes: int,
            sent_embedding_dim: int,
            sent_output_dim: int
            ) -> None:
        super().__init__()

        self.embedding_layer = nn.Embedding(
                num_embeddings=num_sent_classes,
                embedding_dim=sent_embedding_dim,
                padding_idx=0
                )
        self.linear = nn.Linear(
                in_features=sent_embedding_dim,
                out_features=sent_output_dim
                )

    def forward(self, sentiment):
        return torch.tanh(self.linear(self.embedding_layer(sentiment)))


class Generator(nn.Module):
    def __init__(
            self,
            pretrained_pretrained_word_embeddings_path: str,
            dropout_probability: float,
            word_embbeding_dim: int,
            num_attention_heads: int,
            query_vector_dim: int,
            num_sent_classes: int,
            sent_embedding_dim: int
            ) -> None:
        super().__init__()

        # model components
        pretrained_word_embeddings = torch.from_numpy(np.load(pretrained_pretrained_word_embeddings_path)).float()
        self.news_encoder = NewsEncoder(
                pretrained_word_embeddings=pretrained_word_embeddings,
                word_embedding_dim=word_embbeding_dim,
                num_attention_heads=num_attention_heads,
                query_vector_dim=query_vector_dim,
                dropout_probability=dropout_probability
        )

        self.user_encoder = UserEncoder(
                word_embedding_dim=word_embbeding_dim,
                num_attention_heads=num_attention_heads,
                query_vector_dim=query_vector_dim
                )

        self.sentiment_encoder = SentimentEncoder(
                num_sent_classes=num_sent_classes,
                sent_embedding_dim=sent_embedding_dim,
                sent_output_dim=word_embbeding_dim
                )
        
        self.click_predictor_bias_free = DotProduct()
        self.click_predictor_bias_aware = DotProduct()

    def forward(self, batch: NemigBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # encode clicked news
        clicked_news_vector = self.news_encoder(batch["x_hist"]["title"])
        clicked_news_vector_agg, _ = to_dense_batch(clicked_news_vector, batch["batch_hist"])

        # encode candidate news
        candidate_news_vector = self.news_encoder(batch["x_cand"]["title"])
        candidate_news_vector_agg, _ = to_dense_batch(candidate_news_vector, batch["batch_cand"])

        # encode clicked sentiments
        clicked_senti_vector = self.sentiment_encoder(batch['x_hist']['sentiment'])
        clicked_senti_vector_agg, _ = to_dense_batch(clicked_senti_vector, batch['batch_hist'])

        # encode candidate sentiments
        candidate_senti_vector = self.sentiment_encoder(batch['x_cand']['sentiment'])
        candidate_senti_vector_agg, _ = to_dense_batch(candidate_senti_vector, batch['batch_cand'])

        # bias-free user embedding
        user_bias_free_vector = self.user_encoder(clicked_news_vector_agg)

        # bias-aware user embedding
        user_bias_aware_vector = self.user_encoder(clicked_senti_vector_agg)

        # regularization losses
        loss_orth_clicked_news = torch.mean(
                (torch.div(
                    torch.sum(clicked_news_vector * clicked_senti_vector, dim=-1),
                    1e-8 + torch.linalg.norm(clicked_news_vector, dim=1, ord=2) * torch.linalg.norm(clicked_senti_vector, dim=1, ord=2)
                    )
                 ),
                dim=-1
                )
        loss_orth_candidate_news = torch.mean(
                (torch.div(
                    torch.sum(candidate_news_vector * candidate_senti_vector, dim=-1),
                    1e-8 + torch.linalg.norm(candidate_news_vector, dim=1, ord=2) * torch.linalg.norm(candidate_senti_vector, dim=1, ord=2)
                    )
                 ),
                dim=-1
                )
        loss_orth_user = torch.div(
                torch.bmm(
                    user_bias_free_vector.unsqueeze(dim=1),
                    user_bias_aware_vector.unsqueeze(dim=-1)
                    ).squeeze(dim=1),
                (1e-8 + torch.linalg.norm(user_bias_free_vector, dim=1, ord=2) * (torch.linalg.norm(user_bias_aware_vector, dim=1, ord=2))
                ).unsqueeze(dim=1)
            )
        
        loss_orth = torch.abs(loss_orth_clicked_news) + torch.abs(loss_orth_candidate_news) + torch.abs(loss_orth_user)
        loss_orth = torch.mean(loss_orth)
        
        # scores
        bias_free_scores = self.click_predictor_bias_free(
                user_bias_free_vector.unsqueeze(dim=1),
                candidate_news_vector_agg.permute(0, 2, 1)
                )
        bias_aware_scores = self.click_predictor_bias_aware(
                user_bias_aware_vector.unsqueeze(dim=1),
                candidate_senti_vector_agg.permute(0, 2, 1)
                )
        combined_scores = bias_free_scores + bias_aware_scores

        return combined_scores, bias_free_scores, loss_orth, clicked_news_vector, candidate_news_vector
       

class Discriminator(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int
            ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, clicked_news_vector: torch.Tensor, candidate_news_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_clicked_senti = self.linear2(torch.tanh(self.linear1(clicked_news_vector)))
        predicted_candidate_senti = self.linear2(torch.tanh(self.linear1(candidate_news_vector)))

        return predicted_clicked_senti, predicted_candidate_senti


class SentiDebiasModule(LightningModule):
    def __init__(
        self,
        pretrained_word_embeddings_path: str,
        dropout_probability: float,
        word_embedding_dim: int,
        num_attention_heads: int,
        query_vector_dim: int,
        num_categ_classes: int,
        num_sent_classes: int,
        sent_embedding_dim: int,
        sent_hidden_dim: int,
        alpha_coefficient: float,
        beta_coefficient: float,
        num_polit_classes: int,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        # networks
        self.generator = Generator(
                pretrained_pretrained_word_embeddings_path=self.hparams.pretrained_word_embeddings_path,
                dropout_probability=self.hparams.dropout_probability,
                word_embbeding_dim=self.hparams.word_embedding_dim,
                num_attention_heads=self.hparams.num_attention_heads,
                query_vector_dim=self.hparams.query_vector_dim,
                num_sent_classes=self.hparams.num_sent_classes,
                sent_embedding_dim=self.hparams.sent_embedding_dim
                )
        self.discriminator = Discriminator(
                input_dim=self.hparams.word_embedding_dim,
                hidden_dim=self.hparams.sent_hidden_dim,
                output_dim=self.hparams.num_sent_classes
                )

        self.a_loss = CrossEntropyLoss()
        self.rec_loss = CrossEntropyLoss()
        
        # metric objects for calculating and averaging performance across batches
        metrics = MetricCollection(
            {
                "auc": AUROC(task="binary", num_classes=2),
                "mrr": RetrievalMRR(),
                "ndcg@3": RetrievalNormalizedDCG(k=3),
                "ndcg@6": RetrievalNormalizedDCG(k=6),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        polit_div_metrics = MetricCollection({
            'polit_div@3': Diversity(num_classes=self.hparams.num_polit_classes, k=3),
            'polit_div@6': Diversity(num_classes=self.hparams.num_polit_classes, k=6) 
            })
        categ_div_metrics = MetricCollection({
            'categ_div@3': Diversity(num_classes=self.hparams.num_categ_classes, k=3),
            'categ_div@6': Diversity(num_classes=self.hparams.num_categ_classes, k=6) 
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
            'categ_pers@3': Personalization(num_classes=self.hparams.num_categ_classes, k=3),
            'categ_pers@6': Personalization(num_classes=self.hparams.num_categ_classes, k=6) 
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

        # for tracking best so far validation loss
        self.val_acc = AUROC(task="binary", num_classes=2)
        self.val_acc_best = MaxMetric()

    def forward(self, batch: NemigBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.generator(batch)

    def adversarial_loss(self, preds, targets):
        y_true = torch.zeros(preds.shape, device=preds.device)
        for i in range(targets.shape[0]):
            y_true[i, targets[i]-1] = 1.0

        return self.a_loss(preds, y_true)

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_metrics.reset()

    def model_step(self, batch: NemigBatch) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor]:

        _, bias_free_scores, _, _, _ = self.generator(batch)
        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])
        
        candidate_politic, _ = to_dense_batch(batch['x_cand']['politic'], batch['batch_cand'])
        candidate_categories, _ = to_dense_batch(batch["x_cand"]["category"], batch["batch_cand"])
        candidate_sentiments, _ = to_dense_batch(batch["x_cand"]["sentiment"], batch["batch_cand"])

        clicked_politic, mask_hist = to_dense_batch(batch['x_hist']['politic'], batch['batch_hist'])
        clicked_categories, mask_hist = to_dense_batch(
            batch["x_hist"]["category"], batch["batch_hist"]
        )
        clicked_sentiments, _ = to_dense_batch(batch["x_hist"]["sentiment"], batch["batch_hist"])
        
        # predictions, targets, indexes for metric computation
        preds = torch.cat(
            [bias_free_scores[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).detach()
        targets = torch.cat(
            [y_true[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).long()

        cand_news_size = torch.tensor(
            [torch.where(mask_cand[n])[0].shape[0] for n in range(mask_cand.shape[0])]
        )
        hist_news_size = torch.tensor(
            [torch.where(mask_hist[n])[0].shape[0] for n in range(mask_hist.shape[0])]
        )

        target_politic = torch.cat([candidate_politic[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0).long() 
        target_categories = torch.cat(
            [candidate_categories[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).long()
        target_sentiments = torch.cat(
            [candidate_sentiments[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).long()

        hist_politic = torch.cat([clicked_politic[n][mask_hist[n]] for n in range(mask_hist.shape[0])], dim=0).long() 
        hist_categories = torch.cat(
            [clicked_categories[n][mask_hist[n]] for n in range(mask_hist.shape[0])], dim=0
        ).long()
        hist_sentiments = torch.cat(
            [clicked_sentiments[n][mask_hist[n]] for n in range(mask_hist.shape[0])], dim=0
        ).long()

        return preds, targets, cand_news_size, hist_news_size, target_politic, target_categories, target_sentiments, hist_politic, hist_categories, hist_sentiments 

    def training_step(self, batch: NemigBatch, batch_idx: int):
        optimizer_g, optimizer_d = self.optimizers()

        # train generator
        self.toggle_optimizer(optimizer_g, optimizer_idx=0)
        combined_scores, bias_free_scores, loss_orth, clicked_news_vector, candidate_news_vector = self.generator(batch)
        predicted_clicked_senti, predicted_candidate_senti = self.discriminator(clicked_news_vector, candidate_news_vector)
        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])
        
        g_loss = self.rec_loss(combined_scores, y_true) + self.hparams.beta_coefficient * loss_orth - self.hparams.alpha_coefficient * (self.adversarial_loss(predicted_clicked_senti, batch['x_hist']['sentiment']) + self.adversarial_loss(predicted_candidate_senti, batch['x_cand']['sentiment']))
        
        self.log('g_loss', g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
            
        # train discriminator
        
        self.toggle_optimizer(optimizer_d, optimizer_idx=1)
        _, _, _, clicked_news_vector, candidate_news_vector = self.generator(batch)
        predicted_clicked_senti, predicted_candidate_senti = self.discriminator(clicked_news_vector, candidate_news_vector)
        d_loss = self.adversarial_loss(predicted_clicked_senti, batch['x_hist']['sentiment']) + self.adversarial_loss(predicted_candidate_senti, batch['x_cand']['sentiment'])
        
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        preds = torch.cat(
            [bias_free_scores[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).detach()
        targets = torch.cat(
            [y_true[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).long()
        cand_news_size = torch.tensor(
            [torch.where(mask_cand[n])[0].shape[0] for n in range(mask_cand.shape[0])]
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {
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
        preds, targets, cand_news_size, _, _, _, _, _, _, _ = self.model_step(batch)

        return {
                "preds": preds, 
                "targets": targets, 
                "cand_news_size": cand_news_size
                }

    def validation_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([o['preds'] for o in outputs])
        targets = torch.cat([o['targets'] for o in outputs])
        cand_news_size = torch.cat([o['cand_news_size'] for o in outputs])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        
        self.val_acc(preds, targets)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.val_metrics(preds, targets, **{"indexes": indexes})
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: NemigBatch, batch_idx: int):
        preds, targets, cand_news_size, hist_news_size, target_politic, target_categories, target_sentiments, hist_politic, hist_categories, hist_sentiments = self.model_step(batch)

        return {
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
        
        self.test_metrics(preds, targets, **{"indexes": cand_indexes})
        self.test_polit_div_metrics(preds, target_politic, cand_indexes)
        self.test_categ_div_metrics(preds, target_categories, cand_indexes)
        self.test_sent_div_metrics(preds, target_sentiments, cand_indexes)
        self.test_polit_pers_metrics(
            preds, target_politic, hist_politic, cand_indexes, hist_indexes
        )
        self.test_categ_pers_metrics(
            preds, target_categories, hist_categories, cand_indexes, hist_indexes
        )
        self.test_sent_pers_metrics(
            preds, target_sentiments, hist_sentiments, cand_indexes, hist_indexes
        )

        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            self.test_polit_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_categ_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_sent_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_polit_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_categ_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_sent_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        optimizer_generator = self.hparams.optimizer_generator(params=self.generator.parameters())
        optimizer_discriminator = self.hparams.optimizer_discriminator(params=self.discriminator.parameters())

        return [optimizer_generator, optimizer_discriminator]
