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

# --- Interactive Data Generator import ---
try:
    from interactive_data_generator import BinaryDigitDataGenerator, InteractiveTemplateCreator
    print("✅ Interactive Data Generator 모듈 로드 성공")
except ImportError:
    print("⚠️ Interactive Data Generator 모듈을 찾을 수 없습니다.")
    BinaryDigitDataGenerator = None

# --- 기존 EnhancedBinaryDigitGenerator 클래스 import ---
#from Gemini_code1 import EnhancedBinaryDigitGenerator, create_enhanced_binary_digit_dataset

# --- 1-bit 양자화를 위한 STE (Straight Through Estimator) ---
class BinarizeWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_real):
        binary_weight = weight_real.sign()
        binary_weight[binary_weight == 0] = 1  # 0인 경우 +1로 강제
        return binary_weight

    @staticmethod
    def backward(ctx, grad_output_binary_weight):
        return grad_output_binary_weight

class BinarizeActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation_real):
        ctx.save_for_backward(activation_real)
        return activation_real.sign()  # {-1, 1}로 이진화

    @staticmethod
    def backward(ctx, grad_output_binary_activation):
        activation_real, = ctx.saved_tensors
        grad_input = grad_output_binary_activation.clone()
        # STE의 일반적인 클리핑: 입력이 너무 크면 그래디언트 전달 안함
        grad_input[activation_real.abs() > 1.0] = 0
        return grad_input

class BinarizeActivation(nn.Module):
    def forward(self, x):
        return BinarizeActivationSTE.apply(x)

# --- sReLU (Shifted ReLU) 활성화 함수 ---
class sReLU(nn.Module):
    """
    Shifted ReLU: max(-1, x)
    논문: "Single-bit-per-weight deep convolutional neural networks without batch-normalization"
    BN-ReLU 조합을 대체하는 활성화 함수
    """
    def forward(self, x):
        return torch.clamp(x, min=-1.0)  # max(-1, x)

# --- Binary Linear Layer (He 초기화 적용) ---
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, use_bias=False):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        # He 초기화 (MSRA 초기화)
        std = np.sqrt(2.0 / in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        binary_weight = BinarizeWeightSTE.apply(self.weight)
        return F.linear(x, binary_weight, self.bias)

# --- Binary Convolutional Layer (He 초기화 적용) ---
class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=False, padding_value=-1.0):
        super(BinaryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.padding_value = padding_value
        
        # He 초기화 (MSRA 초기화)
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        binary_weight = BinarizeWeightSTE.apply(self.weight)
        
        # 패딩이 필요한 경우, 지정된 값으로 패딩 적용
        if self.padding > 0:
            # (left, right, top, bottom) 순서로 패딩
            pad = (self.padding, self.padding, self.padding, self.padding)
            x = F.pad(x, pad, mode='constant', value=self.padding_value)
            # 패딩을 이미 적용했으므로 conv2d에서는 padding=0
            return F.conv2d(x, binary_weight, self.bias, self.stride, padding=0)
        else:
            return F.conv2d(x, binary_weight, self.bias, self.stride, self.padding)

# --- 상수 스케일링 레이어 ---
class ConstantScaling(nn.Module):
    """
    논문에서 제안한 상수 스케일링 레이어
    마지막 컨볼루션 레이어와 GAP 사이에 위치
    모든 채널에 동일한 상수 값을 곱함 (클래스 수와 동일하게 설정)
    """
    def __init__(self, scale_value=10.0):
        super(ConstantScaling, self).__init__()
        self.scale_value = scale_value
        
    def forward(self, x):
        return x * self.scale_value

# --- 커스텀 Global Average Pooling ---
class CustomGAP(nn.Module):
    """
    사용자 정의 Global Average Pooling.
    공간 차원(H, W)에 대해 합산한 후, 지정된 값으로 나눕니다.
    """
    def __init__(self, divisor=16.0):
        super(CustomGAP, self).__init__()
        # 사용자가 요청한 16으로 나누기 위한 divisor
        self.divisor = float(divisor)
        
    def forward(self, x):
        # 공간 차원(H, W)에 대해 합산. 결과 shape: (N, C, 1, 1)
        spatial_sum = torch.sum(x, dim=(-2, -1), keepdim=True)
        # 지정된 값(16)으로 나눔
        return spatial_sum / self.divisor

# --- 논문 기반 BNN 모델 (BN 없음, sReLU 사용, MaxPool 제거, Sign 활성화 추가) ---
class PaperInspiredBNN(nn.Module):
    """
    "Single-bit-per-weight deep convolutional neural networks without batch-normalization" 논문 기반 모델
    - BN 레이어 제거
    - sReLU 활성화 함수 사용
    - sReLU 후 Sign 활성화 추가
    - 마지막 단에 상수 스케일링
    - He 초기화
    - MaxPooling 제거 (더 많은 공간 정보 보존)
    """
    def __init__(self, c1_channels=16, c2_channels=32, num_classes=10, scale_value=None):
        super(PaperInspiredBNN, self).__init__()
        
        if scale_value is None:
            scale_value = float(num_classes)  # 클래스 수와 동일하게 설정
            
        # 첫 번째 컨볼루션 블록 (MaxPool 제거)
        self.conv1 = BinaryConv2d(1, c1_channels, kernel_size=3, padding=0)
        self.srelu1 = sReLU()
        #self.sign1 = BinarizeActivation()  # sReLU 후 Sign 활성화 추가
        
        # 두 번째 컨볼루션 블록 (MaxPool 제거) - 주석 처리
        #self.conv2 = BinaryConv2d(c1_channels, c2_channels, kernel_size=2, padding=0)
        #self.srelu2 = sReLU()
        # self.sign2 = BinarizeActivation()  # sReLU 후 Sign 활성화 추가
        
        # 상수 스케일링 레이어 (논문에서 제안)
        self.constant_scaling = ConstantScaling(scale_value)
        
        # Custom Global Average Pooling (5x5=25 대신 16으로 나눔)
        self.global_avg_pool = CustomGAP(divisor=16.0)
        
        # GAP 이후 sReLU 활성화 함수
        #self.srelu_after_gap = sReLU()
        self.sign_after_gap = BinarizeActivation()
        
        # 최종 분류 레이어 - c1_channels 사용 (conv2 제거로 인해)
        self.fc = BinaryLinear(c1_channels, num_classes)
        
        print(f"📋 PaperInspiredBNN 초기화 완료:")
        print(f" - 상수 스케일링 값: {scale_value}")
        print(f" - 활성화 함수: sReLU (Conv 이후 + GAP 이후)")
        print(f" - 배치 정규화: 사용 안함")
        print(f" - MaxPooling: 사용 안함 (공간 정보 보존)")
        print(f" - Global Pooling: CustomGAP (divisor=16.0)")
        print(f" - GAP 이후 활성화: sReLU (max(-1, x))")
        print(f" - 컨볼루션 레이어: 1개 (conv1)")
        print(f" - 패딩 값: -1.0 (데이터 일관성을 위해)")
        print(f" - 초기화: He/MSRA 초기화")

    def forward(self, x):
        # 입력 reshape: (batch_size, 64) -> (batch_size, 1, 8, 8)
        x = x.view(-1, 1, 8, 8)
        
        # 첫 번째 컨볼루션 블록 (8x8 유지)
        x = self.conv1(x)
        #x = self.srelu1(x)
        #x = self.sign1(x)  # Sign 활성화 추가
        
        # 두 번째 컨볼루션 블록 (8x8 유지) - 주석 처리
        #x = self.conv2(x)
        #x = self.srelu2(x)
        #x = self.sign2(x)  # Sign 활성화 추가
        
        # 상수 스케일링 (논문의 핵심 기법)
        x = self.constant_scaling(x)
        
        # Custom Global Average Pooling (5x5 -> 1x1)
        x = self.global_avg_pool(x)
        
        # GAP 이후 sReLU 활성화 (max(-1, x))
        #x = self.srelu_after_gap(x)
        
        x = self.sign_after_gap(x)  # Sign 활성화 추가
        x = x.view(x.size(0), -1)  # Flatten
        
        # 최종 분류
        x = self.fc(x)
        
        return x

# --- Early Stopping을 포함한 개선된 훈련 함수 ---
def train_paper_inspired_model(model, X_train, y_train, X_test, y_test, 
                              epochs=300, initial_lr=0.001, weight_decay=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Adam 옵티마이저 사용 (논문에서 권장)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    # 코사인 학습률 감쇠 (논문에서 사용)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=initial_lr/100)

    train_losses, train_accuracies, test_accuracies = [], [], []
    best_test_acc = 0.0

    model_name = model.__class__.__name__
    print(f"🚀 {model_name} 훈련 시작!")
    print(f" 훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    print(f" 에포크: {epochs}, 초기 학습률: {initial_lr}")
    print(f" 학습률 스케줄: 코사인 감쇠 (eta_min: {initial_lr/100:.6f})")
    print(f" 가중치 감쇠: {weight_decay}")
    print(f" Early Stopping: 사용 안함 (전체 에포크 훈련)")

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


        if (epoch + 1) % 25 == 0 or epoch < 10:
            current_lr = scheduler.get_last_lr()[0]
            log_msg = f'Epoch [{epoch+1:3d}/{epochs}] | Loss: {loss.item():.4f} | ' \
                      f'Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | ' 

            print(log_msg)

    return train_losses, train_accuracies, test_accuracies, model

# --- 결과 시각화 함수 ---
def plot_paper_results(train_losses, train_accuracies, test_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 플롯
    ax1.plot(train_losses, 'b-', linewidth=2, alpha=0.8)
    ax1.set_title('Training Loss (Paper-Inspired BNN)', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 로그 스케일로 더 명확하게 보기
    
    # Accuracy 플롯
    ax2.plot(train_accuracies, 'b-', label='Train Accuracy', linewidth=2, alpha=0.8)
    ax2.plot(test_accuracies, 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)
    ax2.set_title('Accuracy (sReLU + Constant Scaling + GAP + sReLU)', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()

# --- 모델 평가 함수 ---
def evaluate_paper_model(model, X_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1)

    print("📊 논문 기반 모델 상세 분류 성능:")
    print(classification_report(y_test, predictions.cpu().numpy(),
                                target_names=[str(i) for i in range(10)]))
    
    return predictions.cpu().numpy()

# --- 데이터셋 로드 함수들 ---
def load_dataset_from_npy():
    """저장된 .npy 파일에서 데이터셋 로드 (prefix 자동 감지)"""
    
    # 가능한 prefix들을 우선순위대로 시도
    prefixes = ['micro_', 'shift_', '', 'augmented_', 'enhanced_']
    
    for prefix in prefixes:
        try:
            print(f"📁 .npy 파일에서 데이터셋 로드 중... (prefix: '{prefix}')")
            X_train = np.load(f'{prefix}X_train.npy')
            X_test = np.load(f'{prefix}X_test.npy')
            y_train = np.load(f'{prefix}y_train.npy')
            y_test = np.load(f'{prefix}y_test.npy')
            
            print(f"✅ 데이터셋 로드 성공! (prefix: '{prefix}')")
            print(f" - X_train: {X_train.shape}")
            print(f" - X_test: {X_test.shape}")
            print(f" - y_train: {y_train.shape}")
            print(f" - y_test: {y_test.shape}")
            
            # 클래스별 분포 확인
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            print(f" - 훈련 데이터 클래스별 분포: {dict(zip(unique_train, counts_train))}")
            print(f" - 테스트 데이터 클래스별 분포: {dict(zip(unique_test, counts_test))}")
            
            return X_train, X_test, y_train, y_test
            
        except FileNotFoundError:
            continue
    
    # 모든 prefix를 시도했지만 실패
    print(f"❌ 데이터셋 파일을 찾을 수 없습니다.")
    print("💡 다음 중 하나를 실행해서 데이터셋을 먼저 생성하세요:")
    print("   1. python shift_augmented_dataset.py --visualize")
    print("   2. python interactive_data_generator.py")
    return None, None, None, None

def load_dataset_from_templates(samples_per_digit=500, test_size=0.2, random_state=42):
    """templates.json에서 템플릿을 로드하여 새 데이터셋 생성"""
    try:
        print("📂 templates.json에서 템플릿 로드하여 데이터셋 생성 중...")
        
        # 템플릿 파일 존재 확인
        if not os.path.exists('templates.json'):
            print("❌ templates.json 파일을 찾을 수 없습니다.")
            print("💡 interactive_data_generator.py를 실행해서 템플릿을 먼저 생성하세요!")
            return None, None, None, None
        
        # InteractiveTemplateCreator로 템플릿 로드
        creator = InteractiveTemplateCreator()
        creator.load_templates('templates.json')
        
        if not creator.templates:
            print("❌ 템플릿이 비어있습니다.")
            return None, None, None, None
        
        # 데이터 생성기 생성
        generator = BinaryDigitDataGenerator(creator.templates, creator.template_params)
        
        # 데이터셋 생성
        X_train, X_test, y_train, y_test = generator.create_dataset(
            samples_per_digit=samples_per_digit, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"✅ 템플릿에서 데이터셋 생성 성공!")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ 템플릿에서 데이터셋 생성 실패: {e}")
        return None, None, None, None

def create_sample_dataset(samples_per_digit=200):
    """간단한 샘플 데이터셋 생성 (템플릿이 없을 때 테스트용)"""
    print("🎲 간단한 샘플 데이터셋 생성 중...")
    
    np.random.seed(42)
    all_X, all_y = [], []
    
    for digit in range(10):
        for _ in range(samples_per_digit):
            # 간단한 패턴 생성 (8x8 = 64차원)
            sample = np.random.choice([-1, 1], size=64, p=[0.7, 0.3])
            all_X.append(sample)
            all_y.append(digit)
    
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y)
    
    # 데이터 셔플
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # 훈련/테스트 분할
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ 샘플 데이터셋 생성 완료!")
    print(f" - 총 샘플: {len(X)}")
    print(f" - 훈련: {len(X_train)}, 테스트: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

# --- BNN 모델 테스터 클래스 ---
class BNNModelTester:
    """
    저장된 BNN 모델을 불러와서 테스트하는 클래스
    """
    def __init__(self, model_path="interactive_template_bnn_model.pth"):
        self.model_path = model_path
        self.model = None
        self.model_config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_model()
    
    def load_model(self):
        """저장된 모델 불러오기"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 모델 설정 정보 로드
            self.model_config = checkpoint['model_config']
            
            # 모델 인스턴스 생성
            self.model = PaperInspiredBNN(
                c1_channels=self.model_config['c1_channels'],
                c2_channels=self.model_config['c2_channels'],
                num_classes=self.model_config['num_classes'],
                scale_value=self.model_config['scale_value']
            )
            
            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ 모델 로드 성공!")
            print(f"📋 모델 설정: {self.model_config}")
            
            if 'final_test_accuracy' in checkpoint:
                print(f"📊 훈련 시 최종 정확도: {checkpoint['final_test_accuracy']:.4f}")
                print(f"🏆 훈련 시 최고 정확도: {checkpoint['best_test_accuracy']:.4f}")
                
        except FileNotFoundError:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {self.model_path}")
            print("💡 먼저 BNN_Model.py를 실행하여 모델을 훈련하고 저장하세요.")
            return
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}")
            return
    
    def preprocess_input(self, input_data):
        """
        입력 데이터 전처리
        input_data: 8x8 numpy array 또는 64-length vector
        """
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        
        # 8x8 형태로 reshape
        if input_data.shape == (64,):
            input_data = input_data.reshape(8, 8)
        elif input_data.shape != (8, 8):
            raise ValueError(f"입력 데이터 형태가 잘못되었습니다. 예상: (8,8) 또는 (64,), 실제: {input_data.shape}")
        
        # 이진화 (0 또는 1) → (-1 또는 1)로 변환 (훈련 데이터와 일치)
        input_data = np.where(input_data > 0.5, 1.0, -1.0).astype(np.float32)
        
        # PyTorch 텐서로 변환 (batch dimension 추가)
        input_tensor = torch.FloatTensor(input_data.flatten()).unsqueeze(0).to(self.device)
        
        return input_tensor, input_data
    
    def predict_single(self, input_data, show_visualization=True):
        """
        단일 8x8 이미지에 대한 예측
        """
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        try:
            # 입력 전처리
            input_tensor, processed_input = self.preprocess_input(input_data)
            
            # 예측 수행
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # 결과 출력
            print(f"\n🔮 예측 결과:")
            print(f"📊 예측된 숫자: {predicted_class}")
            print(f"📈 신뢰도: {confidence:.4f}")
            
            # 모든 클래스의 확률 출력
            print(f"\n📋 모든 클래스 확률:")
            for i in range(10):
                prob = probabilities[0, i].item()
                bar = "█" * int(prob * 20)
                print(f"  {i}: {prob:.4f} {bar}")
            
            # 시각화
            if show_visualization:
                self.visualize_prediction(processed_input, predicted_class, confidence, probabilities[0])
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy()
            }
            
        except Exception as e:
            print(f"❌ 예측 중 오류 발생: {e}")
            return None
    
    def visualize_prediction(self, input_image, predicted_class, confidence, probabilities):
        """예측 결과 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 입력 이미지 시각화
        ax1.imshow(input_image, cmap='gray', interpolation='nearest')
        ax1.set_title(f'입력 이미지 (8x8)\n예측: {predicted_class} (신뢰도: {confidence:.3f})', fontsize=12)
        ax1.set_xticks(range(8))
        ax1.set_yticks(range(8))
        ax1.grid(True, alpha=0.3)
        
        # 확률 분포 시각화
        probabilities_np = probabilities.cpu().numpy()
        bars = ax2.bar(range(10), probabilities_np, alpha=0.7)
        bars[predicted_class].set_color('red')  # 예측된 클래스 강조
        ax2.set_title('클래스별 확률 분포', fontsize=12)
        ax2.set_xlabel('숫자')
        ax2.set_ylabel('확률')
        ax2.set_xticks(range(10))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_batch(self, input_batch, labels=None):
        """
        여러 이미지에 대한 배치 예측
        input_batch: (N, 64) 또는 (N, 8, 8) 형태의 numpy array
        labels: ground truth labels (optional)
        """
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        try:
            # 입력 전처리
            if len(input_batch.shape) == 3:  # (N, 8, 8)
                input_batch = input_batch.reshape(input_batch.shape[0], -1)  # (N, 64)
            
            # 이진화: (0 또는 1) → (-1 또는 1)로 변환 (훈련 데이터와 일치)
            input_batch = np.where(input_batch > 0.5, 1.0, -1.0).astype(np.float32)
            input_tensor = torch.FloatTensor(input_batch).to(self.device)
            
            # 배치 예측
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)
            
            predictions = predicted_classes.cpu().numpy()
            
            print(f"\n🔮 배치 예측 결과 ({len(input_batch)}개 샘플):")
            print(f"📊 예측값: {predictions}")
            
            if labels is not None:
                accuracy = (predictions == labels).mean()
                print(f"📈 정확도: {accuracy:.4f}")
                print(f"\n📋 분류 리포트:")
                print(classification_report(labels, predictions, target_names=[str(i) for i in range(10)]))
            
            return {
                'predictions': predictions,
                'probabilities': probabilities.cpu().numpy()
            }
            
        except Exception as e:
            print(f"❌ 배치 예측 중 오류 발생: {e}")
            return None

def create_custom_test_images():
    """사용자 정의 테스트 이미지 생성 예제"""
    
    # 숫자 '0' 패턴 (-1, 1 값 사용)
    zero_pattern = np.array([
        [-1, 1, 1, 1, 1, 1, 1, -1],
        [1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, -1, -1, -1, -1, 1],
        [1, -1, -1, -1, -1, -1, -1, 1],
        [1, -1, -1, -1, -1, -1, -1, 1],
        [1, -1, -1, -1, -1, -1, -1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1],
        [-1, 1, 1, 1, 1, 1, 1, -1]
    ])
    
    # 숫자 '1' 패턴 (-1, 1 값 사용)
    one_pattern = np.array([
        [-1, -1, -1, 1, 1, -1, -1, -1],
        [-1, -1, 1, 1, 1, -1, -1, -1],
        [-1, 1, 1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1, -1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ])
    
    # 숫자 '2' 패턴 (-1, 1 값 사용)
    two_pattern = np.array([
        [-1, 1, 1, 1, 1, 1, 1, -1],
        [1, 1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1, -1],
        [-1, -1, -1, 1, 1, 1, -1, -1],
        [-1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ])
    
    return {
        'zero': zero_pattern,
        'one': one_pattern,
        'two': two_pattern
    }

def draw_8x8_image():
    """
    마우스로 8x8 이미지를 그릴 수 있는 간단한 GUI.
    왼쪽 클릭: 픽셀 ON(1), 오른쪽 클릭: 픽셀 OFF(0)
    완료 후 Enter를 누르면 numpy array 반환
    """
    
    img = np.zeros((8, 8), dtype=np.int32)
    fig, ax = plt.subplots()
    mat = ax.imshow(img, cmap='gray_r', vmin=0, vmax=1)
    ax.set_title("마우스로 8x8 이미지를 그리세요\n왼쪽 클릭: 1, 오른쪽 클릭: 0\n완료 후 창을 닫으세요")
    plt.xticks(range(8))
    plt.yticks(range(8))
    plt.grid(True, color='lightgray', linewidth=1)

    def onclick(event):
        if event.inaxes != ax:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= x < 8 and 0 <= y < 8:
            if event.button == 1:  # 왼쪽 클릭: 1
                img[y, x] = 1
            elif event.button == 3:  # 오른쪽 클릭: 0
                img[y, x] = 0
            mat.set_data(img)
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
    fig.canvas.mpl_disconnect(cid)
    return img

def interactive_test():
    """대화형 테스트 함수"""
    tester = BNNModelTester()
    
    if tester.model is None:
        return
    
    print("\n🎯 BNN 모델 대화형 테스트")
    print("="*50)
    
    while True:
        print("\n📝 테스트 옵션:")
        print("1. 미리 정의된 패턴 테스트")
        print("2. 사용자 정의 8x8 패턴 입력 (키보드)")
        print("3. 마우스로 8x8 이미지 그리기")
        print("4. 저장된 데이터셋으로 배치 테스트")
        print("5. 종료")
        
        choice = input("\n선택하세요 (1-5): ").strip()
        
        if choice == '1':
            # 미리 정의된 패턴 테스트
            test_patterns = create_custom_test_images()
            
            print("\n🔍 미리 정의된 패턴 테스트:")
            for name, pattern in test_patterns.items():
                print(f"\n--- {name.upper()} 패턴 테스트 ---")
                tester.predict_single(pattern, show_visualization=True)
                
        elif choice == '2':
            # 사용자 정의 입력
            print("\n✏️ 8x8 이진 패턴을 입력하세요 (0 또는 1, 공백으로 구분)")
            print("💡 입력 예시: 0 1 1 0 0 1 1 0")
            print("8줄을 입력하세요:")
            
            try:
                pattern = []
                for i in range(8):
                    line = input(f"줄 {i+1}: ").strip().split()
                    if len(line) != 8:
                        print("❌ 8개의 값을 입력해야 합니다.")
                        break
                    row = [1 if int(x) == 1 else -1 for x in line]  # 0을 -1로 변환
                    pattern.append(row)
                
                if len(pattern) == 8:
                    pattern = np.array(pattern, dtype=np.float32)
                    tester.predict_single(pattern, show_visualization=True)
                    
            except ValueError:
                print("❌ 올바른 숫자를 입력하세요 (0 또는 1).")
            except Exception as e:
                print(f"❌ 입력 처리 중 오류: {e}")
                
        elif choice == '3':
            print("\n🖱️ 마우스로 8x8 이미지를 그리세요!")
            img = draw_8x8_image()
            # 0을 -1로 변환
            img = np.where(img == 1, 1, -1).astype(np.float32)
            tester.predict_single(img, show_visualization=True)
            
        elif choice == '4':
            # 저장된 데이터셋으로 배치 테스트
            try:
                print("\n📊 저장된 테스트 데이터셋으로 배치 테스트...")
                X_test = np.load('X_test.npy')
                y_test = np.load('y_test.npy')
                
                # 처음 10개 샘플만 테스트
                sample_size = min(10, len(X_test))
                X_sample = X_test[:sample_size]
                y_sample = y_test[:sample_size]
                
                result = tester.predict_batch(X_sample, y_sample)
                
            except FileNotFoundError:
                print("❌ X_test.npy 또는 y_test.npy 파일을 찾을 수 없습니다.")
                print("💡 먼저 generate_dataset_no_noise.py를 실행하여 데이터셋을 생성하세요.")
            except Exception as e:
                print(f"❌ 배치 테스트 중 오류: {e}")
                
        elif choice == '5':
            print("👋 테스트를 종료합니다.")
            break
        else:
            print("❌ 올바른 옵션을 선택하세요.")

if __name__ == "__main__":
    print("🎯 Interactive Template 기반 BNN 실험")
    print("📄 Based on: Single-bit-per-weight deep CNNs without batch-normalization")
    print("🎨 Using: Interactive Template Creator Dataset")
    print("="*80)

    # 1. 데이터셋 로드/생성
    print("1️⃣ 데이터셋 로드/생성 중...")
    
    # 방법 1: 저장된 .npy 파일에서 로드 시도
    X_train, X_test, y_train, y_test = load_dataset_from_npy()
    
    # 방법 2: .npy 파일이 없으면 templates.json에서 생성 시도  
    if X_train is None and BinaryDigitDataGenerator is not None:
        print("\n🔄 .npy 파일이 없어서 templates.json에서 데이터셋 생성을 시도합니다...")
        X_train, X_test, y_train, y_test = load_dataset_from_templates(
            samples_per_digit=2500, test_size=0.2, random_state=42
        )
    
    # 방법 3: 둘 다 실패하면 샘플 데이터셋 생성
    if X_train is None:
        print("\n🎲 템플릿도 없어서 샘플 데이터셋을 생성합니다...")
        X_train, X_test, y_train, y_test = create_sample_dataset(samples_per_digit=300)
        print("⚠️ 이는 테스트용 랜덤 데이터입니다. 실제 실험을 위해서는:")
        print("   1. interactive_data_generator.py를 실행해서 템플릿 생성")
        print("   2. 'Generate Data' 버튼으로 데이터셋 저장")
    
    if X_train is None:
        print("❌ 데이터셋 로드/생성에 실패했습니다. 프로그램을 종료합니다.")
        exit(1)
 
    # 2. 논문 기반 모델 생성
    print("\n2️⃣ 논문 기반 모델 생성 중...")
    c1_ch, c2_ch = 128, 16  # 파라미터 수 줄임
    scale_value = 10.0
    
    paper_model = PaperInspiredBNN(c1_channels=c1_ch, c2_channels=c2_ch, 
                                   num_classes=10, scale_value=scale_value)
    
    total_params = sum(p.numel() for p in paper_model.parameters())
    print(f" 총 파라미터 수: {total_params:,}")

    # 3. 논문 기반 모델 훈련
    print("\n3️⃣ 논문 기반 모델 훈련 시작...")
    train_losses, train_accuracies, test_accuracies, trained_model = train_paper_inspired_model(
        paper_model, X_train, y_train, X_test, y_test,
        epochs=1000, initial_lr=0.001, weight_decay=1e-4  # 에포크 줄임
    )

    # 4. 결과 시각화
    print("\n4️⃣ 결과 시각화...")
    plot_paper_results(train_losses, train_accuracies, test_accuracies)

    # 5. 모델 평가
    print("\n5️⃣ 모델 평가...")
    predictions = evaluate_paper_model(trained_model, X_test, y_test)

    # 6. 모델 저장
    print("\n6️⃣ 모델 저장...")
    model_save_path = "interactive_template_bnn_model.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_config': {
            'c1_channels': c1_ch,
            'c2_channels': c2_ch,
            'num_classes': 10,
            'scale_value': scale_value
        },
        'training_history': {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        },
        'final_test_accuracy': test_accuracies[-1],
        'best_test_accuracy': max(test_accuracies)
    }, model_save_path)
    print(f"✅ 모델 저장 완료: {model_save_path}")

    # 7. 실험 요약
    print("\n" + "="*80)
    print("📋 Interactive Template 기반 BNN 실험 요약")
    print("="*80)
    print(f"🎨 데이터: Interactive Template Creator로 생성")
    print(f"📊 데이터셋: {len(X_train)} (훈련) + {len(X_test)} (테스트)")
    print(f"🏗️ 모델 구조: Conv({c1_ch})->sReLU->Scale({scale_value})->GAP->sReLU->FC(10)")
    print(f"⚙️ 총 파라미터 수: {total_params:,}")
    print(f"📈 최종 테스트 정확도: {test_accuracies[-1]:.4f}")
    print(f"🏆 최고 테스트 정확도: {max(test_accuracies):.4f}")
    print(f"💾 모델 저장 위치: {model_save_path}")

    print("\n🎉 Interactive Template 기반 BNN 실험 완료!")
    
    # 8. 대화형 테스트 옵션
    print("\n" + "="*80)
    print("🎯 훈련된 모델을 바로 테스트해볼까요?")
    print("="*80)
    
    while True:
        choice = input("\n테스트를 시작하시겠습니까? (y/n): ").strip().lower()
        if choice in ['y', 'yes', '예', 'ㅇ']:
            print("\n🚀 대화형 테스트 시작!")
            interactive_test()
            break
        elif choice in ['n', 'no', '아니오', 'ㄴ']:
            print("👋 테스트를 건너뜁니다.")
            break
        else:
            print("❌ 올바른 옵션을 선택하세요 (y/n).")
    
    print("\n🎯 BNN 모델 실험이 모두 완료되었습니다!")
    print("💡 나중에 모델을 테스트하려면 다음과 같이 실행하세요:")
    print("   from BNN_Model import BNNModelTester, interactive_test")
    print("   interactive_test()")

    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"현재 GPU: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB") 