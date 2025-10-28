# eval_fs_cap.py
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from scipy.stats import linregress
from tqdm import tqdm

from models import MLPEncoder, Predictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
context_num = 8
model_path = "model.pt"

# --- FS-CAP Test Dataset ---
class FS_CAP_TestDataset(Dataset):
    def __init__(self, tsv_file, test_targets_list, context_num=8):
        self.context_num = context_num

        # TSV 파일 읽기
        df = pd.read_csv(tsv_file, sep='\t', low_memory=False)

        # 컬럼명 확인 후 선택
        df = df[['Target Name Assigned by Curator or DataSource', 'Ligand SMILES', 'IC50 (nM)', 'Ki (nM)']]
        df.columns = ['target', 'smiles', 'IC50', 'Ki']

        # activity 계산: Ki 우선, 없으면 IC50
        df['activity'] = df['Ki'].combine_first(df['IC50'])
        df = df.dropna(subset=['activity', 'smiles', 'target'])

        # test targets만 선택
        df = df[df['target'].isin(test_targets_list)].reset_index(drop=True)

        self.targets = df['target'].values
        self.smiles = df['smiles'].values
        self.activities = df['activity'].values.astype(float)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # fingerprint 계산
        mol = Chem.MolFromSmiles(self.smiles[idx])
        if mol is None:
            fp = np.zeros((2048,))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp = np.array(fp)
        return torch.tensor(fp, dtype=torch.float32), torch.tensor(self.activities[idx], dtype=torch.float32), self.targets[idx]

# --- 테스트용 타겟 리스트 (논문 기준 41개 예시) ---
test_targets_list = [
    'HIV-1 Protease', 'HIV-1 protease M1', 'Cytochrome P450 3A4',
    'Galactokinase (GALK)', 'Caspase-3', 'Caspase-1', 'Caspase-4',
    'HIV-1 protease M2', 'Caspase-7', 'Caspase-6', 'Caspase-5', 'Caspase-8',
    'Caspase-2', 'HIV-1 protease M3', 'HIV-2 Protease',
    'HIV-1 Protease Mutant (L23I)', 'HIV-1 Protease Mutant (L23V)',
    'HIV-1 Protease Mutant (V32I)', 'HIV-1 Protease Mutant (I47L)',
    'HIV-1 Protease Mutant (I50L)', 'HIV-1 Protease Mutant (L76M)',
    'HIV-1 Protease Mutant (V82I)', 'HIV-1 Protease Mutant (I84V)',
    'HIV-1 Protease Mutant (D30N)', 'HIV-1 Protease Mutant (M36I)',
    'HIV-1 Protease Mutant (A71V)', 'HIV-1 Protease Mutant (D30N/M36I)',
    'HIV-1 Protease Mutant (D30N/A71V)', 'HIV-1 Protease Mutant (M36I/A71V)',
    'HIV-1 Protease Mutant (D30N/M36I/A71V)', 'HIV-1 Protease Mutant (A71V/V82T/I84V)',
    'HIV-1 Protease Mutant, ANAM-11', 'HIV-1 Protease Mutant, A-1',
    'HIV-1 Protease Mutant, NAM-10', 'HIV-1 Reverse Transcriptase',
    'HIV-1 Reverse Transcriptase Mutant (Y181C)', 'Orexin receptor type 1 (OX1)',
    'Hormone-sensitive lipase (HSL)', 'Dual specificity mitogen-activated protein kinase kinase 1',
    '11-beta-hydroxysteroid dehydrogenase 1', 'Beta-lactamase (KPC-2)'
]

# --- 데이터셋 및 로더 ---
test_dataset = FS_CAP_TestDataset("Bindingdb_All.tsv", test_targets_list, context_num=context_num)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# --- 모델 로드 ---
checkpoint = torch.load(model_path, map_location=device)

# checkpoint가 tuple인지 dict인지 확인
if isinstance(checkpoint, tuple) or isinstance(checkpoint, list):
    model_state_dicts = checkpoint[0]
else:
    model_state_dicts = checkpoint

# 기본 config
config = {
    'encoder_batchnorm': True,
    'layer_width': 2048,
    'd_model': 512,
    'simple': False
}

# 모델 생성
context_encoder = MLPEncoder(2048, config).to(device)
query_encoder = MLPEncoder(2048, config).to(device)
predictor = Predictor(config['d_model']*2, config).to(device)

# state_dict 로드
if isinstance(model_state_dicts, dict):
    if 'context_encoder' in model_state_dicts:
        context_encoder.load_state_dict(model_state_dicts['context_encoder'])
        query_encoder.load_state_dict(model_state_dicts['query_encoder'])
        predictor.load_state_dict(model_state_dicts['predictor'])
    else:
        context_encoder.load_state_dict(model_state_dicts[0])
        query_encoder.load_state_dict(model_state_dicts[1])
        predictor.load_state_dict(model_state_dicts[2])
else:
    context_encoder.load_state_dict(model_state_dicts[0])
    query_encoder.load_state_dict(model_state_dicts[1])
    predictor.load_state_dict(model_state_dicts[2])

context_encoder.eval()
query_encoder.eval()
predictor.eval()

# --- 평가 ---
all_pred = []
all_real = []
target_to_pred = {}
target_to_real = {}

with torch.no_grad():
    for fp, act, target in tqdm(test_loader, desc="Evaluating"):
        fp = fp.to(device)
        act = act.to(device).unsqueeze(-1)
        # context 계산 (논문 방식 단순 평균)
        context = torch.zeros((context_num, fp.shape[0], config['d_model']), device=device)
        for j in range(context_num):
            context[j] = context_encoder(fp, act)
        context = context.mean(0)
        query = query_encoder(fp)
        x = torch.concat((context, query), dim=1)
        out = predictor(x)

        pred = out.cpu().numpy().flatten()
        real = act.cpu().numpy().flatten()
        all_pred.extend(pred)
        all_real.extend(real)
        for k, t in enumerate(target):
            if t not in target_to_pred:
                target_to_pred[t] = []
                target_to_real[t] = []
            target_to_pred[t].append(pred[k])
            target_to_real[t].append(real[k])

# --- 결과 출력 ---
overall_r = linregress(all_pred, all_real).rvalue
print(f"Overall correlation (r): {overall_r:.4f}")

per_target_r = []
print("\nPer-target correlation:")
for t in target_to_real:
    try:
        r = linregress(target_to_pred[t], target_to_real[t]).rvalue
    except:
        r = 0
    per_target_r.append((t, r))
    print(f"{t}: r = {r:.4f}")

mean_per_target_r = np.mean([r for t, r in per_target_r])
print(f"\nMean per-target correlation: {mean_per_target_r:.4f}")
