import argparse
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import torch
from models import *
from data import *
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FSCAP:
    def __init__(self, model_file):
        # self.context_encoder = MLPEncoder(2048, config).to(device)
        # self.query_encoder = MLPEncoder(2048, config).to(device)
        # self.predictor = Predictor(config['encoding_dim'] * 2, config).to(device)
        # context_encoder_dict, query_encoder_dict, predictor_dict = torch.load(model_file, map_location=device)
        # self.context_encoder.load_state_dict(context_encoder_dict)
        # self.query_encoder.load_state_dict(query_encoder_dict)
        # self.predictor.load_state_dict(predictor_dict)
        # self.context_encoder.eval()
        # self.query_encoder.eval()
        # self.predictor.eval()
        # encoding_dim은 train에서 사용한 d_model과 동일하게 맞춰야 합니다.
        # ----------------------------
        # Self-Attention 적용 수정 - NEW!!
        # ----------------------------
        d_model = config['encoding_dim']

        self.context_encoder = MLPEncoder(2048, config).to(device)
        self.query_encoder   = MLPEncoder(2048, config).to(device)
        self.predictor       = Predictor(d_model * 2, config).to(device)

        # ★ 컨텍스트 Self-Attention 모듈 추가
        # n_heads, n_layers는 train 설정과 동일하게 맞추세요.
        self.context_self_attn = ContextSelfAttention(
            d_model=d_model,
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 1),
            dropout=0.0,   # 추론 단계에서는 보통 dropout=0
        ).to(device)

        # ★ 4개의 state_dict을 로드 (train.py에서 저장한 구조와 동일하게)
        (context_encoder_dict,
         query_encoder_dict,
         predictor_dict,
         context_self_attn_dict) = torch.load(model_file, map_location=device)

        self.context_encoder.load_state_dict(context_encoder_dict)
        self.query_encoder.load_state_dict(query_encoder_dict)
        self.predictor.load_state_dict(predictor_dict)
        self.context_self_attn.load_state_dict(context_self_attn_dict)

        self.context_encoder.eval()
        self.query_encoder.eval()
        self.predictor.eval()
        self.context_self_attn.eval()



    # def predict(self, context_smiles, context_activities, queries):
    #     context_x = torch.tensor(np.array([self.featurize_mol(smile) for smile in context_smiles], dtype=bool)).unsqueeze(0)
    #     context_y = torch.tensor(np.array([self.clip_activity(math.log10(float(activity) + 1e-10)) for activity in context_activities])).unsqueeze(0)
    #     query_x = torch.tensor(np.array([self.featurize_mol(smile) for smile in queries], dtype=bool))
    #     context_x, context_y, query_x = context_x.to(dtype=torch.float32, device=device), context_y.to(dtype=torch.float32, device=device).unsqueeze(-1), query_x.to(dtype=torch.float32, device=device)
        
    #     context = torch.zeros((len(context_smiles), len(context_x), config['encoding_dim']), device=device)
    #     for j in range(len(context_smiles)):
    #         context[j] = self.context_encoder(context_x[:, j, :], context_y[:, j, :])
    #     context = context.mean(0)
    #     query = self.query_encoder(query_x)
        

    #     tiled_contexts = torch.zeros((len(queries), config['encoding_dim']), device=device)
    #     for i in range(len(queries)):
    #         tiled_contexts[i] = context
    #     x = torch.concat((tiled_contexts, query), dim=1)
    #     out = self.predictor(x)
    #     return (10 ** out.detach().cpu().flatten()).tolist()
    # ----------------------------
    # Self-Attention 적용 수정 - NEW!!
    # ----------------------------
    def predict(self, context_smiles, context_activities, queries):
        context_x = torch.tensor(
            np.array([self.featurize_mol(smile) for smile in context_smiles], dtype=bool)
        ).unsqueeze(0)
        context_y = torch.tensor(
            np.array([self.clip_activity(math.log10(float(activity) + 1e-10))
                      for activity in context_activities])
        ).unsqueeze(0)
        query_x = torch.tensor(
            np.array([self.featurize_mol(smile) for smile in queries], dtype=bool)
        )

        # dtype / device 정리
        context_x = context_x.to(dtype=torch.float32, device=device)
        context_y = context_y.to(dtype=torch.float32, device=device).unsqueeze(-1)  # (1, n_ctx, 1)
        query_x   = query_x.to(dtype=torch.float32, device=device)

        # ----------------------------
        # ① 컨텍스트 인코딩
        # ----------------------------
        n_ctx   = len(context_smiles)
        d_model = config['encoding_dim']

        # (n_ctx, batch, d_model) 형태로 쌓기 (여기서 batch=1)
        context = torch.zeros((n_ctx, len(context_x), d_model), device=device)
        for j in range(n_ctx):
            # context_x : (1, n_ctx, in_dim) → context_x[:, j, :] : (1, in_dim)
            # context_y : (1, n_ctx, 1)      → context_y[:, j, :] : (1, 1)
            context[j] = self.context_encoder(context_x[:, j, :], context_y[:, j, :])

        # (n_ctx, batch, d_model) → (batch, n_ctx, d_model)
        context = context.permute(1, 0, 2)        # (1, n_ctx, d_model)

        # ----------------------------
        # ② Self-Attention 적용 (ANP-style)
        # ----------------------------
        context = self.context_self_attn(context) # (1, n_ctx, d_model)

        # ----------------------------
        # ③ set representation으로 집계 (순서 불변성 유지)
        # ----------------------------
        context = context.mean(dim=1)             # (1, d_model)

        # ----------------------------
        # ④ Query 인코딩 & concat
        # ----------------------------
        query = self.query_encoder(query_x)       # (n_query, d_model)

        # 각 query마다 동일한 context를 붙여주기 위해 tile
        tiled_contexts = context.expand(len(queries), -1)  # (n_query, d_model)

        x   = torch.concat((tiled_contexts, query), dim=1) # (n_query, 2*d_model)
        out = self.predictor(x)
        return (10 ** out.detach().cpu().flatten()).tolist()


    def featurize_mol(self, smiles):
        if not ((10 <= len([char for char in smiles if char not in '()=@[]123456789']) <= 70) and MolFromSmiles(smiles)):
            raise ValueError('smiles is invalid, or too long/short')
        return np.array(GetMorganFingerprintAsBitVect(MolFromSmiles(smiles), 3))

    def clip_activity(self, val):
        if val < -2.5:
            val = -2.5
        if val > 6.5:
            val = 6.5
        return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context_smiles', type=str)
    parser.add_argument('--context_activities', type=str)
    parser.add_argument('--query_smiles', type=str)
    parser.add_argument('--model_file', type=str, default='model.pt')
    parser.add_argument('--encoding_dim', type=int, default=512)
    args = parser.parse_args()
    config = {'encoding_dim': args.encoding_dim}
    fscap = FSCAP(args.model_file)
    for prediction in fscap.predict(args.context_smiles.split(';'), args.context_activities.split(';'), args.query_smiles.split(';')):
        print(prediction)
