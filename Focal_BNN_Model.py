import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Import base classes from the main BNN model
from BNN_Model import (
    BinarizeWeightSTE, BinarizeActivationSTE, BinarizeActivation, sReLU,
    BinaryLinear, BinaryConv2d, ConstantScaling, CustomGAP,
    load_dataset_from_npy, create_sample_dataset
)

# --- Interactive Data Generator import ---
try:
    from BNN_Model import load_dataset_from_templates
    print("✅ Interactive Data Generator 관련 함수 로드 성공")
except ImportError:
    print("⚠️ Interactive Data Generator 관련 함수를 찾을 수 없습니다.")
    load_dataset_from_templates = None

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    """
    Focal Loss 구현
    논문: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    클래스 불균형 문제 해결을 위한 손실 함수
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: 클래스별 가중치 (list, tensor, or None)
        gamma: focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if isinstance(alpha, list):
            self.alpha = torch.FloatTensor(alpha)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logit 값
            targets: (N,) 정답 레이블
        """
        # Cross Entropy Loss 계산
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 소프트맥스 확률 계산 (p_t)
        p = torch.exp(-ce_loss)
        
        # Alpha 가중치 적용
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - p) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Quantization-Aware Training을 위한 STE ---
class TruncationSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for Truncation.
    - Forward pass: truncates the input tensor.
    - Backward pass: passes the gradient as is (identity function).
    """
    @staticmethod
    def forward(ctx, x):
        return torch.trunc(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# --- Focal Loss 기반 BNN 모델 ---
class FocalBNN(nn.Module):
    """
    Focal Loss를 사용하는 Binary Neural Network
    - 클래스 불균형 데이터셋에서 성능 향상을 위한 전용 모델
    - 기본 구조는 PaperInspiredBNN과 동일하지만 Focal Loss 최적화됨
    """
    def __init__(self, c1_channels=128, num_classes=10, scale_value=None):
        super(FocalBNN, self).__init__()
        
        if scale_value is None:
            scale_value = float(num_classes)
            
        # 첫 번째 컨볼루션 블록
        self.conv1 = BinaryConv2d(1, c1_channels, kernel_size=3, padding=0)
        self.srelu1 = sReLU()

        # 상수 스케일링 레이어
        self.constant_scaling = ConstantScaling(scale_value)

        # Custom Global Average Pooling (논문과 동일하게 32로 나누기)
        self.global_avg_pool = CustomGAP(divisor=32.0)

        # GAP 이후 sReLU 활성화 함수
        self.srelu_after_gap = sReLU()

        # 최종 분류 레이어
        self.fc = BinaryLinear(c1_channels, num_classes)
        
        print(f"🎯 FocalBNN 초기화 완료:")
        print(f" - 손실 함수: Focal Loss (클래스 불균형 대응)")
        print(f" - 상수 스케일링 값: {scale_value}")
        print(f" - 활성화 함수: sReLU (Conv 이후 + GAP 이후)")
        print(f" - 배치 정규화: 사용 안함")
        print(f" - MaxPooling: 사용 안함 (공간 정보 보존)")
        print(f" - Global Pooling: CustomGAP (divisor=32.0)")
        print(f" - 컨볼루션 레이어: 1개 (conv1)")
        print(f" - 패딩 값: -1.0 (데이터 일관성을 위해)")
        print(f" - 초기화: He/MSRA 초기화")

    def forward(self, x):
        # 입력 reshape: (batch_size, 64) -> (batch_size, 1, 8, 8)
        x = x.view(-1, 1, 8, 8)
        
        # 첫 번째 컨볼루션 블록
        x = self.conv1(x)
        x = self.srelu1(x)
        
        # 상수 스케일링
        x = self.constant_scaling(x)
        
        # Custom Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Quantization-Aware Training: GAP 이후 소수점 제거 (STE 적용)
        x = TruncationSTE.apply(x)
        
        # GAP 이후 STE 기반 이진화 활성화
        x = self.srelu_after_gap(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        # 최종 분류
        x = self.fc(x)
        
        return x

# --- 하이브리드 손실 함수 ---
class HybridLoss(nn.Module):
    def __init__(self, focal_weight=0.2, focal_alpha=None, focal_gamma=2.0):
        super(HybridLoss, self).__init__()
        self.focal_weight = focal_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, inputs, targets):
        focal_loss_val = self.focal_loss(inputs, targets)
        ce_loss_val = self.ce_loss(inputs, targets)
        total_loss = self.focal_weight * focal_loss_val + (1 - self.focal_weight) * ce_loss_val
        return total_loss

# --- 하이브리드 훈련 함수 ---
def train_hybrid_bnn_model(model, X_train, y_train, X_test, y_test, 
                          epochs=500, initial_lr=0.001, weight_decay=1e-4,
                          focal_weight=0.2, focal_gamma=2.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    criterion = HybridLoss(focal_weight=focal_weight, focal_gamma=focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=initial_lr/100)

    train_losses, train_accuracies, test_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            train_pred = torch.argmax(train_outputs, dim=1)
            train_acc = (train_pred == y_train_tensor).float().mean()

            test_outputs = model(X_test_tensor)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_acc = (test_pred == y_test_tensor).float().mean()

        scheduler.step()

        train_losses.append(loss.item())
        train_accuracies.append(train_acc.item())
        test_accuracies.append(test_acc.item())

        if (epoch + 1) % 50 == 0 or epoch < 10:
            print(f'Epoch [{epoch+1:3d}/{epochs}] | Loss: {loss.item():.4f} | '
                  f'Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')

    return train_losses, train_accuracies, test_accuracies, model

def plot_hybrid_results(train_losses, train_accuracies, test_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_title('Training Loss (Hybrid)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Accuracy (Hybrid Loss)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()

# --- 모델 평가 함수 ---
def evaluate_focal_model(model, X_test, y_test):
    """Focal Loss BNN 모델 상세 평가"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1)

    print("📊 Focal Loss BNN 모델 상세 분류 성능:")
    print(classification_report(y_test, predictions.cpu().numpy(),
                                target_names=[str(i) for i in range(10)]))
    
    # 클래스별 정확도 분석
    cm = confusion_matrix(y_test, predictions.cpu().numpy())
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    print("\n📈 클래스별 정확도:")
    for i, acc in enumerate(class_accuracies):
        print(f"  클래스 {i}: {acc:.4f}")
    
    return predictions.cpu().numpy()



if __name__ == "__main__":
    print("🎯 하이브리드 손실 기반 BNN 실험")
    print("📄 목적: Focal Loss + Cross-Entropy Loss 조합으로 안정적 학습")
    print("🎨 Using: Interactive Template Creator Dataset")
    print("="*80)

    # 1. 데이터셋 로드/생성
    print("1️⃣ 데이터셋 로드/생성 중...")
    
    # 방법 1: 저장된 .npy 파일에서 로드 시도
    X_train, X_test, y_train, y_test = load_dataset_from_npy()
    
    # 방법 2: .npy 파일이 없으면 templates.json에서 생성 시도  
    if X_train is None and load_dataset_from_templates is not None:
        print("\n🔄 .npy 파일이 없어서 templates.json에서 데이터셋 생성을 시도합니다...")
        X_train, X_test, y_train, y_test = load_dataset_from_templates(
            samples_per_digit=2500, test_size=0.2, random_state=42
        )
    
    # 방법 3: 둘 다 실패하면 샘플 데이터셋 생성
    if X_train is None:
        print("\n🎲 템플릿도 없어서 샘플 데이터셋을 생성합니다...")
        X_train, X_test, y_train, y_test = create_sample_dataset(samples_per_digit=500)
    
    if X_train is None:
        print("❌ 데이터셋 로드/생성에 실패했습니다. 프로그램을 종료합니다.")
        exit(1)

    # 2. 데이터셋 클래스 분포 확인
    print("\n2️⃣ 데이터셋 클래스 분포 확인...")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    
    print(f"📊 훈련 데이터 클래스별 분포:")
    for cls, count in zip(unique_train, counts_train):
        percentage = count / total_samples * 100
        print(f"  클래스 {cls}: {count}개 ({percentage:.1f}%)")

    # 3. 하이브리드 BNN 모델 훈련
    print("\n3️⃣ 하이브리드 BNN 모델 훈련...")
    c1_ch = 128
    scale_value = 10.0
    
    hybrid_model = FocalBNN(c1_channels=c1_ch, num_classes=10, scale_value=scale_value)
    
    total_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f" 총 파라미터 수: {total_params:,}")

    # 훈련 실행
    train_losses, train_acc, test_acc, trained_model = train_hybrid_bnn_model(
        hybrid_model, X_train, y_train, X_test, y_test,
        epochs=800, initial_lr=0.015, weight_decay=1e-4, focal_weight=0.2
    )

    # 4. 결과 시각화
    print("\n4️⃣ 결과 시각화...")
    plot_hybrid_results(train_losses, train_acc, test_acc)

    # 5. 모델 평가
    print("\n5️⃣ 모델 평가...")
    predictions = evaluate_focal_model(trained_model, X_test, y_test)

    # 6. 모델 저장
    model_save_path = "hybrid_bnn_model.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_config': {
            'c1_channels': c1_ch,
            'num_classes': 10,
            'scale_value': scale_value
        },
        'training_history': {
            'train_losses': train_losses,
            'train_accuracies': train_acc,
            'test_accuracies': test_acc
        },
        'final_test_accuracy': test_acc[-1],
        'best_test_accuracy': max(test_acc)
    }, model_save_path)
    print(f"\n✅ 모델 저장: {model_save_path}")

    # 7. 요약
    print(f"📈 최종 테스트 정확도: {test_acc[-1]:.4f}")
    print(f"🏆 최고 테스트 정확도: {max(test_acc):.4f}")

    print(f"\n🎉 하이브리드 BNN 실험 완료!")

    print(f"\nCUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"현재 GPU: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB") 