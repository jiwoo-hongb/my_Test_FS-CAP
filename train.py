import torch
import numpy as np
import random
from torch import optim
from scipy.stats import linregress
from models import *
from data import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import collections
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# 난수 시드 설정 함수
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # RDKit이나 dataloader의 worker seed까지 고정하면 더 좋습니다.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# 시드 설정: 논문 재현을 위해 고정된 시드 값 사용
FIXED_SEED = 42  # 데이터 분할에 사용된 시드와 동일하게 사용
set_seed(FIXED_SEED)

# ----------------------------
# 설정
# ----------------------------
# 학습용 컨텍스트 개수: 항상 8-shot
context_num = 8

# 평가할 n-shot 리스트 (1, 2, 4, 8-shot)
eval_n_shots = [1, 2, 4, 8]

config = {
    'run_name': f'new_ver_1_{FIXED_SEED}',
    # 논문처럼 pIC50 3~10 범위 사용
    'context_ranges': [(3.0, 10.0)] * context_num,  # unit is log10 nM
    'val_freq': 1024,
    'lr': 0.000040012,
    'layer_width': 2048,
    'd_model': 512,
    'batch_size': 1024,
    'warmup_steps': 128,
    'total_epochs': 2 ** 15,
    'n_heads': 16,
    'n_layers': 4,
    'affinity_embed_layers': 1,
    'init_range': 0.2,
    'scalar_dropout': 0.15766,
    'embed_dropout': 0.16668,
    'final_dropout': 0.10161,
    'pred_dropout': True,
    'pred_batchnorm': False,
    'pred_dropout_p': 0.1,
    'encoder_batchnorm': True,
    'simple': False
}

# dataloader_batch는 "학습용 컨텍스트 개수" 기준으로 설정
if config['simple']:
    config['dataloader_batch'] = 1024 // len(config['context_ranges'])
else:
    config['dataloader_batch'] = 128 // len(config['context_ranges'])

# ----------------------------
# 데이터 및 모델
# ----------------------------
# 학습은 항상 8-shot 기준으로만 진행
train_dataloader, _ = get_dataloaders(
    config['dataloader_batch'],
    config['context_ranges']
)

# n-shot 평가용 test_dataloader들 준비 (1/2/4/8-shot)
test_dataloaders = {}
for n_ctx in eval_n_shots:
    eval_context_ranges = [(3.0, 10.0)] * n_ctx
    _, test_loader_n = get_dataloaders(
        config['dataloader_batch'],  # 평가에서는 같은 batch 크기 사용
        eval_context_ranges
    )
    test_dataloaders[n_ctx] = test_loader_n

context_encoder = MLPEncoder(config['layer_width'], config).cuda()
query_encoder = MLPEncoder(config['layer_width'], config).cuda()
predictor = Predictor(config['d_model'] * 2, config).cuda()

# ----------------------------
# 옵티마이저 & 스케줄러
# ----------------------------
optimizer = optim.RAdam(
    list(context_encoder.parameters()) +
    list(query_encoder.parameters()) +
    list(predictor.parameters()), lr=config['lr']
)

warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.0001,
    end_factor=1.0,
    total_iters=config['warmup_steps']
)
annealing_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['total_epochs']
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    [warmup_scheduler, annealing_scheduler],
    milestones=[config['warmup_steps']]
)

# ----------------------------
# TensorBoard
# ----------------------------
writer = SummaryWriter('logs/' + config['run_name'])

# ----------------------------
# n-shot 평가 함수 (1/2/4/8-shot 공용)
# ----------------------------
def evaluate(context_encoder, query_encoder, predictor,
             test_loader, config, epoch, writer, suffix=""):
    """
    suffix: '_1shot', '_2shot', '_4shot', '_8shot' 같은 문자열
    """
    context_encoder.eval()
    query_encoder.eval()
    predictor.eval()

    val_loss = 0.0
    target_to_pred = {}
    target_to_real = {}
    all_pred = []
    all_real = []

    with torch.no_grad():
        for j, (context_x, context_y, query_x, query_y, targets) in enumerate(test_loader):
            context_x = context_x.to(dtype=torch.float32, device='cuda')
            context_y = context_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1)
            query_x = query_x.to(dtype=torch.float32, device='cuda')
            query_y = query_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1)

            # 현재 평가용 컨텍스트 개수 (1,2,4,8 등)
            num_ctx = context_x.shape[1]

            # 컨텍스트 인코딩 (train 루프와 동일 구조)
            context = torch.zeros((num_ctx, len(context_x), config['d_model']), device='cuda')
            for k in range(num_ctx):
                context[k] = context_encoder(context_x[:, k, :], context_y[:, k, :])
            context = context.mean(0)

            # 쿼리 인코딩
            query = query_encoder(query_x)

            # 예측
            x = torch.concat((context, query), dim=1)
            out = predictor(x)

            # MSE
            val_loss += torch.mean((out - query_y) ** 2).item()

            # numpy로 변환
            pred = out.cpu().numpy().flatten()
            real = query_y.cpu().numpy().flatten()
            all_pred.extend(pred)
            all_real.extend(real)

            # target별로 모으기 (per-target corr 계산용)
            for idx, target in enumerate(targets):
                if target not in target_to_real:
                    target_to_pred[target] = []
                    target_to_real[target] = []
                target_to_pred[target].append(pred[idx])
                target_to_real[target].append(real[idx])

    # 배치 수 j+1로 나눈 평균 loss
    writer.add_scalar(f'loss/test{suffix}', val_loss / (j + 1), epoch)

    # raw correlation (모든 샘플 기준)
    try:
        raw_corr = linregress(all_pred, all_real).rvalue
    except:
        raw_corr = 0.0
    writer.add_scalar(f'corr/raw{suffix}', raw_corr, epoch)

    # per-target correlation
    corrs = []
    for target in target_to_real:
        try:
            corrs.append(linregress(target_to_pred[target], target_to_real[target]).rvalue)
        except:
            corrs.append(0.0)
    writer.add_scalar(f'corr/per_target{suffix}', np.mean(corrs), epoch)

    # 학습 모드로 복귀
    context_encoder.train()
    query_encoder.train()
    predictor.train()

# ----------------------------
# 학습 루프
# ----------------------------
epoch = 0
while True:
    total_loss = 0.0
    count = 0
    for i, (context_x, context_y, query_x, query_y, _) in enumerate(train_dataloader):
        context_x = context_x.to(dtype=torch.float32, device='cuda')
        context_y = context_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1)
        query_x = query_x.to(dtype=torch.float32, device='cuda')
        query_y = query_y.to(dtype=torch.float32, device='cuda').unsqueeze(-1)

        # ----------------------------
        # 컨텍스트 인코딩 (항상 8-shot 기준)
        # ----------------------------
        num_ctx_train = len(config['context_ranges'])  # = 8
        context = torch.zeros((num_ctx_train, len(context_x), config['d_model']), device='cuda')
        for j in range(num_ctx_train):
            context[j] = context_encoder(context_x[:, j, :], context_y[:, j, :])
        context = context.mean(0)

        # ----------------------------
        # 쿼리 인코딩 & 예측
        # ----------------------------
        query = query_encoder(query_x)
        x = torch.concat((context, query), dim=1)
        loss = torch.mean((predictor(x) - query_y) ** 2)
        total_loss += loss.item()
        count += 1
        loss.backward()

        # ----------------------------
        # 학습 진행 상태 출력
        # ----------------------------
        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Train Loss: {total_loss / count:.6f}")

        # ----------------------------
        # n-shot 검증 (1/2/4/8-shot 전부 평가)
        # ----------------------------
        if i % (config['val_freq'] * (config['batch_size'] // config['dataloader_batch'])) == 0:
            # train loss 기록
            writer.add_scalar('loss/train', total_loss / count, epoch)

            # 각 n-shot에 대해 evaluate 실행
            for n_ctx in eval_n_shots:
                suffix = f'_{n_ctx}shot'
                evaluate(
                    context_encoder,
                    query_encoder,
                    predictor,
                    test_dataloaders[n_ctx],
                    config,
                    epoch,
                    writer,
                    suffix=suffix
                )

        # ----------------------------
        # 옵티마이저 업데이트
        # ----------------------------
        if i % (config['batch_size'] // config['dataloader_batch']) == 0:
            total_loss = 0.0
            count = 0
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # epoch 인자 제거
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
            epoch += 1

            if epoch == config['total_epochs']:
                torch.save(
                    (context_encoder.state_dict(), query_encoder.state_dict(), predictor.state_dict()),
                    f'model.pt'
                )
                exit()
