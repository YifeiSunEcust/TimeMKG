from math import sqrt
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from layers.Embed import PatchEmbedding, DataEmbedding_inverted
import transformers
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.StandardNorm import Normalize
import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="The current process just got forked")
transformers.logging.set_verbosity_error()

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.max_len = configs.max_len
        self.data_path = os.path.join(configs.root_path, configs.data_path)
        with open(self.data_path, 'r') as f:
            first_line = f.readline().strip()
        self.variables = first_line.split(',')[1:]
        self.prompt_bank = configs.prompt_path
        with open(self.prompt_bank, 'r') as f:
            lines = f.readlines()
        self.prompts = {}
        for line in lines:
            line = line.strip()
            if line:
                variable, prompt = line.split(':', 1)
                self.prompts[variable.strip()] = prompt.strip()

        model_path = "Qwen3_4B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.LLMmodel = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=None)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Time Series Encoder
        self.TS_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Prompt Encoder
        self.Prompt_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.sentenceAggregator = nn.Sequential(
            nn.Linear(self.max_len, self.max_len // 2),
            nn.ReLU(),
            nn.Linear(self.max_len // 2, 1)
        )
        self.LLMtodim = nn.Linear(self.d_llm, configs.d_model)

        # Cross Attention
        self.cross_attention = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        B, T, N = x_enc.size()

        prompts = []
        for b in range(B):
            for n in range(N):
                response = self.prompts[self.variables[n]]
                prompts.append(response)

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.LLMmodel.device)
        prompt_embeddings = self.LLMmodel.model.embed_tokens(inputs.input_ids)
        self.d_llm = prompt_embeddings.size(-1)
        prompt_embeddings = prompt_embeddings.view(B, N, -1, self.d_llm)
        prompt_embeddings = prompt_embeddings[:, :, :self.max_len, :]
        prompt_embeddings = prompt_embeddings.permute(0, 1, 3, 2).contiguous()

        prompt_out = self.sentenceAggregator(prompt_embeddings)
        prompt_out = prompt_out.squeeze(-1)
        prompt_out = self.LLMtodim(prompt_out)
        # Prompt Encoder
        prompt_out, attns = self.Prompt_encoder(prompt_out, attn_mask=None)

        # Time Series encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.TS_encoder(enc_out, attn_mask=None)

        # Cross Attention
        dec_out = self.cross_attention(enc_out, prompt_out, x_mask=None, cross_mask=None)

        # decoder
        dec_out, attn = self.decoder(dec_out, attn_mask=None)
        dec_out = self.projection(dec_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        B, T, N = x_enc.size()
        prompts = []
        for b in range(B):
            for n in range(N):
                response = self.prompts[self.variables[n]]
                prompts.append(response)
        save_path = "./prompt_embeddings.pt"
        if os.path.exists(save_path):
            prompt_embeddings = torch.load(save_path).to(x_enc.device)
        else:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.LLMmodel.device)
            prompt_embeddings = self.LLMmodel.model.embed_tokens(inputs.input_ids)
            torch.save(prompt_embeddings.cpu(), save_path)
            print(f"Prompt embeddings saved to {save_path}")

        self.d_llm = prompt_embeddings.size(-1)
        prompt_embeddings = prompt_embeddings.view(B, N, -1, self.d_llm)
        # Casual Prompt Encoder
        prompt_embeddings = prompt_embeddings[:, :, :self.max_len, :]
        prompt_embeddings = prompt_embeddings.permute(0, 1, 3, 2).contiguous()

        prompt_out = self.sentenceAggregator(prompt_embeddings)
        prompt_out = prompt_out.squeeze(-1)
        prompt_out = self.LLMtodim(prompt_out)
        # Prompt Encoder
        prompt_out, attns = self.Prompt_encoder(prompt_out, attn_mask=None)

        # Time Series encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.TS_encoder(enc_out, attn_mask=None)

        # Cross-Modality Attention
        dec_out = self.cross_attention(enc_out, prompt_out, x_mask=None, cross_mask=None)

        # decoder
        dec_out, attn = self.decoder(dec_out, attn_mask=None)

        # Output
        output = self.act(dec_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

