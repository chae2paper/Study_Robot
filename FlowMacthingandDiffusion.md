# Flow Matching vs Diffusion: 완전 정복 가이드

> 이 문서는 Diffusion과 Flow Matching을 처음 접하는 사람도 이해할 수 있도록, 직관부터 수식, 그리고 실제 코드까지 단계별로 설명하려고 노력합니다
> 제가 공부하고자 만든 문서이기 때문에 내용이 부족하거나 오류가 있다면 issues 부탁드립니다 :)

---

## 목차

1. [생성 모델이란?](#1-생성-모델이란)
2. [Diffusion 이해하기](#2-diffusion-이해하기)
3. [Flow Matching 이해하기](#3-flow-matching-이해하기)
4. [Diffusion vs Flow Matching 비교](#4-diffusion-vs-flow-matching-비교)
5. [수식](#5-수식-완전-정복)
6. [코드로 이해](#6-코드로-이해)
7. [Meta의 flow_matching 라이브러리 사용법](#7-meta의-flow_matching-라이브러리-사용법)
8. [Q: 직선 경로의 이해](#8-Q-직선-경로의-이해)
9. [로보틱스 응용](#9-로보틱스-응용)

---

## 1. 생성 모델이란?

### 1.1 목표

Diffusion과 Flow Matching 모두 같은 목표를 가집니다:

```
노이즈 분포 ──────────────> 데이터 분포
(랜덤한 점들)              (의미있는 데이터: 이미지, 로봇 동작 등)
```

우리가 원하는 건 랜덤한 노이즈에서 시작해서 진짜 같은 데이터를 만들어내는 것입니다.

### 1.2 핵심 질문

"노이즈를 어떻게 데이터로 바꾸지?"

이 질문에 대한 답이 Diffusion과 Flow Matching에서 다릅니다:

| 방법 | 접근법 |
|------|--------|
| Diffusion | "데이터에 노이즈를 섞었을 때, 어떤 노이즈가 섞였는지 맞춰봐" |
| Flow Matching | "지금 위치에서 데이터 방향이 어디인지 맞춰봐" |

---

## 2. Diffusion 이해하기

### 2.1 Forward Process: 노이즈 섞기

Diffusion은 먼저 깨끗한 데이터에 노이즈를 점점 섞는 과정을 정의합니다:

```
깨끗한 이미지 → 약간 흐림 → 더 흐림 → ... → 완전한 노이즈
    x_0           x_1         x_2              x_T
```

수식으로 표현하면:

$$x_t = \sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{1-\bar\alpha_t} \cdot \epsilon$$

여기서:
- $x_0$: 원본 데이터 (예: 깨끗한 이미지)
- $\epsilon$: 순수한 노이즈 $\sim \mathcal{N}(0, I)$
- $\bar\alpha_t$: 시간에 따라 감소하는 값 (1에서 0으로)
- $x_t$: 시간 $t$에서의 noisy 데이터

### 2.2 학습: "어떤 노이즈가 섞였지?"

Diffusion 모델은 섞인 노이즈를 예측하도록 학습합니다:

```python
# Diffusion 학습 (pseudo-code)
for step in range(num_steps):
    x_0 = sample_data()                    # 원본 데이터 샘플
    epsilon = torch.randn_like(x_0)        # 노이즈 샘플 (정답!)
    t = torch.randint(0, T, (batch_size,)) # 랜덤 시간
    
    # Forward process: 노이즈 섞기
    x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * epsilon
    
    # 모델에게 문제 출제: "x_t에 어떤 노이즈가 섞였게?"
    epsilon_pred = model(x_t, t)
    
    # 정답과 비교
    loss = MSE(epsilon_pred, epsilon)
    loss.backward()
```

핵심: 학습할 때는 우리가 직접 노이즈를 섞었으므로 정답($\epsilon$)을 알고 있습니다.

### 2.3 샘플링: 노이즈 빼기

학습된 모델로 새 데이터를 생성할 때는 역방향으로 갑니다:

```python
# Diffusion 샘플링 (pseudo-code)
x = torch.randn(...)  # 순수 노이즈에서 시작

for t in reversed(range(T)):
    epsilon_pred = model(x, t)  # "여기에 섞인 노이즈는 이거일 듯"
    
    # 노이즈 제거 + 새 노이즈 추가 (SDE 특성)
    x = denoise_step(x, epsilon_pred, t) + sigma_t * torch.randn_like(x)
    
return x  # 생성된 데이터!
```

중요: 매 스텝마다 새로운 노이즈를 추가합니다 (확률적 과정)

---

## 3. Flow Matching 이해하기

### 3.1 핵심 아이디어: 직선 경로

Flow Matching은 노이즈에서 데이터로 가는 직선 경로를 정의합니다.

```
노이즈 (x_0) ─────────────────> 데이터 (x_1)
              직선으로 이동!
```

수식:

$$x_t = (1-t) \cdot x_0 + t \cdot x_1$$

여기서:
- $x_0$: 노이즈 $\sim \mathcal{N}(0, I)$
- $x_1$: 실제 데이터
- $t$: 시간 (0에서 1로)
- $x_t$: 시간 $t$에서의 중간 지점

### 3.2 Velocity 구하기: 학습 target

경로 $x_t = (1-t)x_0 + tx_1$를 정의했으니, 이제 모델이 학습할 target을 구해야 합니다.

Flow Matching에서 모델은 velocity (속도 벡터)를 예측합니다. "지금 위치에서 어느 방향으로 가야 하는지"를 알려주는 값입니다.

#### Velocity 유도

경로를 시간 $t$로 미분하면:

$$\frac{dx_t}{dt} = \frac{d}{dt}[(1-t) \cdot x_0 + t \cdot x_1]$$

$x_0$와 $x_1$은 상수 (샘플링하면 고정된 값)이므로:

$$= x_0 \cdot (-1) + x_1 \cdot (1) = x_1 - x_0$$

#### 결과의 의미

$$\text{velocity} = x_1 - x_0$$

이건 그냥 "시작점에서 끝점으로 향하는 방향"입니다. 

등속 직선 운동이라서 $t$가 식에 남지 않습니다. 출발할 때나 중간이나 도착 직전이나 같은 방향, 같은 속력으로 이동합니다.

이 $(x_1 - x_0)$가 바로 학습할 때 모델이 맞춰야 할 정답 (target)이 됩니다.

### 3.3 학습: "어느 방향으로 가야 하지?"

Flow Matching 모델은 velocity(방향)를 예측하도록 학습합니다:

```python
# Flow Matching 학습 (pseudo-code)
for step in range(num_steps):
    x_1 = sample_data()              # 목적지 (데이터)
    x_0 = torch.randn_like(x_1)      # 시작점 (노이즈)
    t = torch.rand(batch_size)       # 랜덤 시간 [0, 1]
    
    # 중간 지점 계산
    x_t = (1 - t) * x_0 + t * x_1
    
    # 정답 velocity
    target_velocity = x_1 - x_0
    
    # 모델에게 문제 출제: "x_t에서 어느 방향으로 가야 해?"
    velocity_pred = model(x_t, t)
    
    # 정답과 비교
    loss = MSE(velocity_pred, target_velocity)
    loss.backward()
```

핵심: 학습할 때는 $x_0$와 $x_1$ 모두 아니까 정답(velocity = $x_1 - x_0$)을 알고 있습니다.

### 3.4 문제: 샘플링할 때는 $x_1$을 모른다

학습할 때는 $x_1$을 알아서 정답을 계산할 수 있었습니다. 

그런데 샘플링할 때는 어떻게 하죠?_?

```python
x = torch.randn(...)  # 노이즈에서 시작
for t in steps:
    v = model(x, t)   # 여기서 뭘 예측해야 하지?
    x = x + v * dt
```

샘플링 시점에서 $x_1$ (목적지)을 모릅니다. 생성하려고 하는 중이라서요

그래서 모델은 "지금 위치 $x$에서, 목적지를 모르는 상태로, 평균적으로 어디로 가야 하는지"를 알려줘야 합니다. 이게 Marginal Velocity $v_t(x)$입니다:

$$v_t(x) = \mathbb{E}_{x_1 \sim p(x_1|x_t=x)}[u_t(x|x_1)]$$

#### 문제: Marginal Velocity는 계산 불가능

이걸 구하려면 $p(x_1|x_t=x)$, 즉 "지금 $x$에 있는데, 이게 어떤 데이터 $x_1$에서 온 거지?"를 알아야 합니다.

이걸 알려면 모든 데이터 포인트에 대해 "이 $x$가 여기서 왔을 확률"을 계산해야 하는데, 데이터가 수백만 개면 불가능합니다.

#### 해결: Conditional Velocity를 학습하면 된다

그런데 3.3에서 한 게 뭐였죠? 

$x_1$을 정해놓고 (샘플링해서), 그에 대한 velocity $(x_1 - x_0)$를 학습했습니다. 이게 바로 Conditional Velocity입니다.

$$u_t(x|x_1) = x_1 - x_0$$

$x_0$도 내가 샘플링했고, $x_1$도 내가 샘플링했으니까, 계산 가능합니다. 적분 필요 없음

#### 비유로 이해...!

| | Marginal | Conditional |
|---|----------|-------------|
| 질문 | "서울역에 있는 사람이 평균적으로 어디로 가나?" | "이 사람은 부산 간다고 정하면, 어디로 가야 하나?" |
| 계산 | 모든 사람의 목적지 조사 필요 (불가능) | 그냥 부산 방향 (바로 계산) |

#### 핵심 정리: 왜 Conditional을 학습해도 되는가?

CFM (Conditional Flow Matching) 정리:

$$\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$$

Conditional target으로 학습해도, 실제로는 Marginal을 학습하는 것과 같은 효과

직관적으로:
- 학습할 때: 랜덤하게 $(x_0, x_1)$ 쌍을 엄청 많이 뽑아서 각각의 방향 $(x_1 - x_0)$ 학습
- 충분히 학습하면: 모델이 자연스럽게 "그 위치에서의 평균 방향"을 배우게 됨
regression의 기본 성질입니다. 조건부 target들의 평균을 예측하도록 학습하면, 결국 marginal을 예측하게 됩니다.

#### Diffusion에서도 똑같은 문제가 있다

| | Diffusion | Flow Matching |
|---|-----------|---------------|
| 학습할 때 아는 것 | $x_0$ (데이터), $\epsilon$ (노이즈) | $x_0$ (노이즈), $x_1$ (데이터) |
| Conditional target | $\epsilon$ | $x_1 - x_0$ |
| 샘플링 때 모르는 것 | 원래 $x_0$가 뭐였는지 | 목적지 $x_1$이 뭔지 |
| 모델이 예측하는 것 | "평균적인" 노이즈 방향 | "평균적인" velocity |

Diffusion도 사실 conditional target ($\epsilon$)을 학습하지만, 충분히 학습하면 marginal하게 작동한다고 합니다.

### 3.5 샘플링: ODE 풀기

학습된 모델로 새 데이터를 생성:

```python
# Flow Matching 샘플링 (pseudo-code)
x = torch.randn(...)  # 노이즈에서 시작 (t=0)

num_steps = 20
dt = 1.0 / num_steps

for i in range(num_steps):
    t = i / num_steps
    v = model(x, t)    # "여기서 어느 방향으로 가야 해?"
    x = x + v * dt     # 그 방향으로 이동 (Euler method)
    
return x  # 생성된 데이터! (t=1)
```

중요: 새로운 노이즈를 추가하지 않습니다 (결정론적 과정)

---

## 4. Diffusion vs Flow Matching 비교

### 4.1 비교

| 구분 | Diffusion | Flow Matching |
|------|-----------|---------------|
| 수학적 기반 | SDE (확률미분방정식) | ODE (상미분방정식) |
| 학습 target | 노이즈 $\epsilon$ | Velocity $x_1 - x_0$ |
| 경로 | 곡선 (노이즈 스케줄 의존) | 직선 (Optimal Transport) |
| 샘플링 | 확률적 (매 스텝 노이즈 추가) | 결정론적 (노이즈 없음) |
| 재현성 | 같은 시작점 → 다른 결과 | 같은 시작점 → 같은 결과 |

### 4.2 시각적 비교

```
Diffusion (SDE):
x_0 ~~~~~~~~> 곡선 경로 ~~~~~~~~> x_T
     ↑ 매 스텝 노이즈 추가로 경로가 흔들림

Flow Matching (ODE):
x_0 ────────> 직선 경로 ────────> x_1
     깔끔하게 직진!
```

### 4.3 샘플링 스텝 비교

```python
# Diffusion 샘플링 한 스텝
for t in steps:
    pred = model(x, t)
    x = x + pred * scale + sigma * torch.randn_like(x)  # 노이즈 추가!
                           ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# Flow Matching 샘플링 한 스텝
for t in steps:
    v = model(x, t)
    x = x + v * dt  # 노이즈 없음!
```

### 4.4 왜 Flow Matching이 적은 스텝으로 되나?

Diffusion: 노이즈 스케줄에 의한 곡선 경로. Euler method로 근사하면 오차가 큼.

Flow Matching: 직선 경로. Euler method가 직선을 거의 완벽히 따라감.

```
곡선 경로 (Diffusion):
x_0 ----→ 실제 경로 (곡선)
    \
     \____  Euler 근사 (오차 발생)
          \
           → x_T

직선 경로 (Flow Matching):
x_0 --------→ x_1   ← 실제 경로
x_0 --------→ x_1   ← Euler 근사 (거의 일치!)
```

---

## 5. 수식

### 5.1 Diffusion 수식

Forward Process:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$$

Reparameterization:

$$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Loss Function:

$$\mathcal{L}_{\text{Diffusion}} = \mathbb{E}_{t, x_0, \epsilon}\left[ \| \epsilon_\theta(x_t, t) - \epsilon \|^2 \right]$$

### 5.2 Flow Matching 수식

Probability Path:

$$x_t = (1-t)x_0 + tx_1$$

여기서 $x_0 \sim \mathcal{N}(0, I)$, $x_1 \sim p_{\text{data}}$

Conditional Velocity:

$$u_t(x|x_1) = \frac{dx_t}{dt} = x_1 - x_0$$

Loss Function (Conditional Flow Matching):

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1}\left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

### 5.3 Marginal vs Conditional: 핵심 수식 정리

#### Marginal Velocity (우리가 진짜 원하는 것)

샘플링할 때 필요한 건 marginal velocity입니다:

$$v_t(x) = \int u_t(x|x_1) \cdot p(x_1|x_t=x) \, dx_1$$

하지만 $p(x_1|x_t=x)$ (역방향 조건부 확률)은 계산 불가능합니다.

#### Conditional Velocity (우리가 실제로 계산하는 것)

$x_1$을 고정하면:

$$u_t(x|x_1) = \frac{d}{dt}[(1-t)x_0 + tx_1] = x_1 - x_0$$

이건 시간 $t$에 무관하고, 알고 있는 값들로만 구성됩니다.

#### CFM 정리

Conditional Flow Matching Loss:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1}\left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

Flow Matching Loss (이론적):

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x}\left[ \| v_\theta(x, t) - v_t(x) \|^2 \right]$$

정리:

$$\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$$

두 loss의 gradient가 같으므로, conditional target으로 학습해도 marginal velocity를 학습하는 것과 동일한 효과!

#### 왜 이게 성립하는가?

Regression의 기본 성질:

$$\mathbb{E}[\|f(x) - Y\|^2]$$

를 최소화하면 $f(x) = \mathbb{E}[Y|x]$를 얻습니다.

Flow Matching에서:
- $Y = x_1 - x_0$ (conditional target)
- 최적의 $v_\theta(x_t, t) = \mathbb{E}[x_1 - x_0 | x_t]$ = marginal velocity!

충분히 많은 $(x_0, x_1)$ 쌍으로 학습하면, 모델은 자연스럽게 그 위치에서의 "평균 방향"을 배웁니다.

---

## 6. 코드로 이해

### 6.1 간단한 2D 예제: Flow Matching

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 간단한 MLP 모델
class VelocityNet(nn.Module):
    def __init__(self, dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t):
        # t를 x와 concat
        t = t.view(-1, 1) if t.dim() == 1 else t
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

# 2. 데이터 생성 (예: 2D 원형 분포)
def sample_data(n_samples):
    """원형 분포에서 샘플링"""
    theta = torch.rand(n_samples) * 2 * 3.14159
    r = 1.0 + 0.1 * torch.randn(n_samples)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)

# 3. Flow Matching 학습
def train_flow_matching(model, n_iterations=5000, batch_size=256, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i in range(n_iterations):
        optimizer.zero_grad()
        
        # 샘플링
        x_1 = sample_data(batch_size)           # 목적지 (데이터)
        x_0 = torch.randn_like(x_1)             # 시작점 (노이즈)
        t = torch.rand(batch_size)              # 시간 [0, 1]
        
        # 중간 지점 (interpolation)
        t_expanded = t.view(-1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # Target velocity
        target = x_1 - x_0
        
        # 예측
        pred = model(x_t, t)
        
        # Loss
        loss = ((pred - target)  2).mean()
        
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 1000 == 0:
            print(f"Step {i+1}, Loss: {loss.item():.4f}")
    
    return model

# 4. 샘플링 (ODE 풀기)
@torch.no_grad()
def sample(model, n_samples=1000, n_steps=50):
    x = torch.randn(n_samples, 2)  # 노이즈에서 시작
    dt = 1.0 / n_steps
    
    for i in range(n_steps):
        t = torch.full((n_samples,), i / n_steps)
        v = model(x, t)
        x = x + v * dt  # Euler step
    
    return x

# 5. 실행!
if __name__ == "__main__":
    model = VelocityNet()
    model = train_flow_matching(model)
    
    # 샘플 생성
    samples = sample(model)
    
    # 시각화
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    real_data = sample_data(1000)
    plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, s=1)
    plt.title("Real Data")
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
    plt.title("Generated Samples")
    plt.axis('equal')
    
    plt.savefig("flow_matching_result.png")
    plt.show()
```

### 6.2 간단한 2D 예제: Diffusion (비교용)

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 데이터 생성 함수 (Flow Matching과 동일)
def sample_data(n_samples):
    """원형 분포에서 샘플링"""
    theta = torch.rand(n_samples) * 2 * 3.14159
    r = 1.0 + 0.1 * torch.randn(n_samples)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)

class NoiseNet(nn.Module):
    """Diffusion용: 노이즈 예측 네트워크"""
    def __init__(self, dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t):
        t = t.view(-1, 1) if t.dim() == 1 else t
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

def get_alpha_schedule(T=1000):
    """노이즈 스케줄 생성"""
    beta = torch.linspace(1e-4, 0.02, T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return alpha_bar

def train_diffusion(model, n_iterations=5000, batch_size=256, T=1000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    alpha_bar = get_alpha_schedule(T)
    
    for i in range(n_iterations):
        optimizer.zero_grad()
        
        # 샘플링
        x_0 = sample_data(batch_size)           # 원본 데이터
        epsilon = torch.randn_like(x_0)         # 노이즈 (정답!)
        t = torch.randint(0, T, (batch_size,))  # 랜덤 시간 스텝
        
        # Forward process: 노이즈 섞기
        alpha_bar_t = alpha_bar[t].view(-1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        
        # 예측
        t_normalized = t.float() / T  # [0, 1]로 정규화
        epsilon_pred = model(x_t, t_normalized)
        
        # Loss
        loss = ((epsilon_pred - epsilon)  2).mean()
        
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 1000 == 0:
            print(f"Step {i+1}, Loss: {loss.item():.4f}")
    
    return model

@torch.no_grad()
def sample_diffusion(model, n_samples=1000, T=1000):
    alpha_bar = get_alpha_schedule(T)
    beta = torch.linspace(1e-4, 0.02, T)
    alpha = 1 - beta
    
    x = torch.randn(n_samples, 2)  # 순수 노이즈에서 시작
    
    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t)
        t_normalized = t_batch.float() / T
        
        epsilon_pred = model(x, t_normalized)
        
        # Denoising step
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_pred
        )
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta[t])
            x = mean + sigma * noise  # 노이즈 추가! (SDE 특성)
        else:
            x = mean
    
    return x

# 실행!
if __name__ == "__main__":
    model = NoiseNet()
    model = train_diffusion(model)
    
    # 샘플 생성
    samples = sample_diffusion(model)
    
    # 시각화
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    real_data = sample_data(1000)
    plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, s=1)
    plt.title("Real Data")
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
    plt.title("Generated Samples (Diffusion)")
    plt.axis('equal')
    
    plt.savefig("diffusion_result.png")
    plt.show()
```

---

## 7. Meta의 flow_matching 라이브러리 사용법

Meta (Facebook Research)에서 공식 Flow Matching 라이브러리를 사용하는 방법

### 7.1 설치

```bash
# pip 설치 (권장)
pip install flow-matching

# 또는 conda 환경 생성
git clone https://github.com/facebookresearch/flow_matching.git
cd flow_matching
conda env create -f environment.yml
conda activate flow_matching
```

### 7.2 라이브러리 구조

```
flow_matching/
├── path/              # Probability Path 정의
│   └── scheduler/     # 스케줄러 (CondOT, VP 등)
├── loss/              # Loss 함수
├── solver/            # ODE Solver
└── utils/             # 유틸리티
```

### 7.3 기본 사용법: 2D Flow Matching

공식 예제 (`examples/2d_flow_matching.ipynb`) 기반입니다.

```python
import torch
from torch import nn, Tensor

# flow_matching 라이브러리 import
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

import matplotlib.pyplot as plt

# 1. Velocity 모델 정의
class VelocityMLP(nn.Module):
    def __init__(self, dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: [batch, dim] - 위치
            t: [batch] - 시간
        Returns:
            velocity: [batch, dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        t = t.unsqueeze(-1)  # [batch, 1]
        return self.net(torch.cat([x, t], dim=-1))

# 2. ModelWrapper 상속 (공식 방식)
class WrappedModel(ModelWrapper):
    """ODESolver가 사용하는 인터페이스로 모델 래핑"""
    def forward(self, x: Tensor, t: Tensor, extras) -> Tensor:
        return self.model(x, t)

# 3. 데이터 생성 함수 (체커보드 패턴)
def sample_checkerboard(batch_size: int, device: str = 'cpu') -> Tensor:
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size,), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)
    return torch.stack([x1, x2], dim=1) * 2

# 4. 학습
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VelocityMLP().to(device)

# AffineProbPath: x_t = (1-t)*x_0 + t*x_1 형태의 경로
# CondOTScheduler: Conditional Optimal Transport 스케줄러
path = AffineProbPath(scheduler=CondOTScheduler())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 256
iterations = 5000

for i in range(iterations):
    optimizer.zero_grad()
    
    # 데이터 샘플링
    x_1 = sample_checkerboard(batch_size, device)  # 목적지 (데이터)
    x_0 = torch.randn_like(x_1)                     # 시작점 (노이즈)
    t = torch.rand(batch_size, device=device)       # 시간
    
    # path.sample(): x_t와 dx_t (velocity target) 계산
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
    
    # 모델 예측
    v_pred = model(path_sample.x_t, path_sample.t)
    
    # Flow Matching Loss
    loss = torch.pow(v_pred - path_sample.dx_t, 2).mean()
    
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 1000 == 0:
        print(f"Step {i+1}, Loss: {loss.item():.4f}")

# 5. 샘플링
wrapped_model = WrappedModel(model)
solver = ODESolver(velocity_model=wrapped_model)

# 노이즈에서 시작
n_samples = 2048
x_init = torch.randn(n_samples, 2, device=device)

# ODE 풀기 (t=0에서 t=1로)
step_size = 0.05
time_grid = torch.tensor([0.0, 1.0], device=device)

sol = solver.sample(
    time_grid=time_grid,
    x_init=x_init,
    method='midpoint',
    step_size=step_size
)

# 6. 시각화
samples = sol.cpu().numpy()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
real_data = sample_checkerboard(n_samples, 'cpu').numpy()
plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.3, s=1)
plt.title("Real Data (Checkerboard)")
plt.axis('equal')
plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.subplot(1, 2, 2)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
plt.title("Generated Samples")
plt.axis('equal')
plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.savefig("flow_matching_checkerboard.png")
plt.show()
```

### 7.4 path.sample() 반환값 설명

```python
path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

# path_sample 객체 내용:
# - path_sample.x_t: 시간 t에서의 위치 (interpolated)
# - path_sample.t: 시간 값
# - path_sample.dx_t: target velocity (학습에 사용할 정답)
```

### 7.5 다양한 스케줄러

```python
from flow_matching.path.scheduler import (
    CondOTScheduler,        # Conditional Optimal Transport (직선 경로)
    VPScheduler,            # Variance Preserving (Diffusion 유사)
    PolynomialConvexScheduler  # 다항식 스케줄
)

# CondOT (기본, 권장)
# x_t = (1-t)*x_0 + t*x_1
path_ot = AffineProbPath(scheduler=CondOTScheduler())

# VP (Diffusion 경로와 유사)
path_vp = AffineProbPath(scheduler=VPScheduler())
```

### 7.6 ODESolver 옵션

```python
from flow_matching.solver import ODESolver

solver = ODESolver(velocity_model=wrapped_model)

# 다양한 solver method
samples = solver.sample(
    time_grid=torch.linspace(0, 1, 21),
    x_init=x_init,
    method='euler',      # 가장 빠름, 정확도 낮음
    # method='midpoint',  # 균형
    # method='heun3',     # 더 정확
    # method='dopri5',    # Adaptive step, 가장 정확
    step_size=0.05,
    return_intermediates=False  # True면 모든 중간 결과 반환
)
```

### 7.7 이미지 생성 예제 (CIFAR-10 스타일)

이미지 생성 예제입니다. 간단한 UNet 구조를 포함합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

import matplotlib.pyplot as plt

# ============================================
# 1. 간단한 UNet 정의
# ============================================

class SinusoidalPosEmb(nn.Module):
    """시간 t를 위한 sinusoidal position embedding"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Time-conditioned Residual Block"""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]  # time conditioning
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    """
    간단한 UNet for 32x32 이미지
    - Encoder: 32 -> 16 -> 8
    - Decoder: 8 -> 16 -> 32
    """
    def __init__(self, in_channels: int = 3, base_dim: int = 64, time_dim: int = 128):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        
        self.down1 = ResBlock(base_dim, base_dim, time_dim)
        self.down2 = ResBlock(base_dim, base_dim * 2, time_dim)
        self.pool1 = nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1)
        self.pool2 = nn.Conv2d(base_dim * 2, base_dim * 2, 3, stride=2, padding=1)
        
        # Bottleneck
        self.mid = ResBlock(base_dim * 2, base_dim * 2, time_dim)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_dim * 4, base_dim, time_dim)  # skip connection
        self.up2 = nn.ConvTranspose2d(base_dim, base_dim, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base_dim * 2, base_dim, time_dim)  # skip connection
        
        self.conv_out = nn.Conv2d(base_dim, in_channels, 3, padding=1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] 이미지
            t: [B] 시간 (0~1)
        Returns:
            velocity: [B, C, H, W]
        """
        # Time embedding
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        t_emb = self.time_mlp(t)
        
        # Encoder
        h = self.conv_in(x)                    # [B, 64, 32, 32]
        h1 = self.down1(h, t_emb)              # [B, 64, 32, 32]
        h = self.pool1(h1)                     # [B, 64, 16, 16]
        h2 = self.down2(h, t_emb)              # [B, 128, 16, 16]
        h = self.pool2(h2)                     # [B, 128, 8, 8]
        
        # Bottleneck
        h = self.mid(h, t_emb)                 # [B, 128, 8, 8]
        
        # Decoder with skip connections
        h = self.up1(h)                        # [B, 128, 16, 16]
        h = torch.cat([h, h2], dim=1)          # [B, 256, 16, 16]
        h = self.dec1(h, t_emb)                # [B, 64, 16, 16]
        h = self.up2(h)                        # [B, 64, 32, 32]
        h = torch.cat([h, h1], dim=1)          # [B, 128, 32, 32]
        h = self.dec2(h, t_emb)                # [B, 64, 32, 32]
        
        return self.conv_out(h)                # [B, 3, 32, 32]


# ============================================
# 2. ModelWrapper (ODESolver용)
# ============================================

class WrappedUNet(ModelWrapper):
    def forward(self, x: Tensor, t: Tensor, extras) -> Tensor:
        return self.model(x, t)


# ============================================
# 3. 학습 코드
# ============================================

def train_flow_matching_cifar():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 데이터 로드 (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]로 정규화
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # 모델 초기화
    model = SimpleUNet(in_channels=3, base_dim=64).to(device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 학습
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()
            
            x_1 = images.to(device)                    # 목적지 (데이터)
            x_0 = torch.randn_like(x_1)                # 시작점 (노이즈)
            t = torch.rand(x_1.shape[0], device=device)  # 시간
            
            # Flow Matching
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            v_pred = model(path_sample.x_t, path_sample.t)
            loss = (v_pred - path_sample.dx_t).pow(2).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


# ============================================
# 4. 샘플링 코드
# ============================================

@torch.no_grad()
def sample_images(model, n_samples: int = 16, steps: int = 50, device: str = 'cuda'):
    model.eval()
    
    wrapped_model = WrappedUNet(model)
    solver = ODESolver(velocity_model=wrapped_model)
    
    # 노이즈에서 시작
    x_init = torch.randn(n_samples, 3, 32, 32, device=device)
    
    # ODE 풀기
    time_grid = torch.tensor([0.0, 1.0], device=device)
    step_size = 1.0 / steps
    
    samples = solver.sample(
        time_grid=time_grid,
        x_init=x_init,
        method='midpoint',
        step_size=step_size
    )
    
    # [-1, 1] -> [0, 1]로 변환
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    
    return samples


def visualize_samples(samples: Tensor, nrow: int = 4):
    """생성된 이미지 시각화"""
    n_samples = samples.shape[0]
    ncol = (n_samples + nrow - 1) // nrow
    
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < n_samples:
            img = samples[i].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("generated_cifar10.png")
    plt.show()


# ============================================
# 5. 실행
# ============================================

if __name__ == "__main__":
    # 학습
    model = train_flow_matching_cifar()
    
    # 샘플링
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    samples = sample_images(model, n_samples=16, steps=50, device=device)
    
    # 시각화
    visualize_samples(samples)
    
    print("Done! Generated images saved to generated_cifar10.png")
```

코드 구조 요약:

| 컴포넌트 | 설명 |
|----------|------|
| `SinusoidalPosEmb` | 시간 $t$를 고차원 벡터로 임베딩 |
| `ResBlock` | Time-conditioned residual block |
| `SimpleUNet` | 32x32 이미지용 간단한 UNet (Encoder-Bottleneck-Decoder) |
| `WrappedUNet` | `ModelWrapper` 상속하여 ODESolver와 호환 |
| `train_flow_matching_cifar` | CIFAR-10으로 학습 |
| `sample_images` | 학습된 모델로 이미지 생성 |

---

## 8. Q: 직선 경로의 이해

### 8.1 왜 직선 경로여도 되는가?

Flow Matching에서 직선 경로 $x_t = (1-t)x_0 + tx_1$를 사용하는데, "왜 하필 직선이어도 되는 거지?"라는 의문이 들 수 있습니다.

#### 핵심: 우리가 원하는 건 "도착지"지, "경로"가 아니다

목표를 다시 보면:

$$p_0 = \mathcal{N}(0, I) \quad \longrightarrow \quad p_1 = p_{\text{data}}$$

우리가 신경 쓰는 건:
- $t=0$에서 노이즈 분포
- $t=1$에서 데이터 분포

중간에 어떤 경로로 가든, 최종 분포만 맞으면 됩니다!

#### 비유: 서울에서 부산 가기

| 경로 | 설명 |
|------|------|
| 곡선 (Diffusion) | 대전 거쳐서, 대구 거쳐서... |
| 직선 (Flow Matching) | 직선으로 쭉! |

둘 다 부산에 도착하면 목표 달성이에요. 어떤 경로로 갔는지는 최종 결과에 영향을 주지 않습니다.

#### 수학적으로: Continuity Equation만 만족하면 된다

확률 분포 $p_t$와 velocity field $v_t$가 연속 방정식을 만족하면 OK:

$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v_t) = 0$$

이 조건을 만족하는 $(p_t, v_t)$ 쌍은 무한히 많습니다. Diffusion의 곡선 경로도 만족하고, Flow Matching의 직선 경로도 만족해요.

#### 그런데 왜 하필 직선을 선택하나?

Optimal Transport 관점: 두 분포 사이에서 mass를 옮기는 최소 비용 경로는?

$$\min \mathbb{E}[\|x_1 - x_0\|^2]$$

답: 직선, 돌아가거나 굽이치면 비용(거리)이 늘어남.

| 직선의 장점 | 이유 |
|------------|------|
| 적은 스텝 | 직선이라 Euler integration 오차 적음 |
| 빠른 학습 | 경로가 단순해서 velocity 예측이 쉬움 |
| 안정적 | 곡선 경로의 복잡한 dynamics 없음 |

### 8.2 학습할 때 1:1 매칭인가?

아니요! 

#### 학습할 때: 랜덤 매칭

```python
x_0 = torch.randn(batch_size, dim)   # 노이즈들
x_1 = sample_data(batch_size)         # 데이터들

# x_0[0]과 x_1[0]이 쌍이 됨
# x_0[1]과 x_1[1]이 쌍이 됨
# ...
```

매 배치마다 랜덤하게 쌍이 맺어집니다. 고정된 1:1 매칭이 아니에요.

같은 $x_1$ (데이터 포인트)이 다음 배치에서는 다른 $x_0$ (노이즈)와 쌍이 될 수 있습니다.

#### 시각적으로

```
배치 1:                    배치 2:
x_0[a] -----> x_1[고양이]    x_0[c] -----> x_1[고양이]  (다른 노이즈!)
x_0[b] -----> x_1[강아지]    x_0[d] -----> x_1[강아지]
```

#### 샘플링할 때

샘플링할 때는 $x_0$ (노이즈)만 주어지고, 모델이 평균적인 방향을 예측해서 어딘가에 도착합니다.

어떤 특정 $x_1$으로 가는 게 아니라, 학습된 "평균 velocity field"를 따라가는 거예요. 이게 바로 앞서 설명한 Marginal Velocity입니다.

### 8.3 직선 경로의 단점: 경로 교차 (Path Crossing)

직선 경로의 가장 큰 단점입니다!

#### 문제 상황

```
x_0[a] --------\    /-------> x_1[고양이]
                \  /
                 \/  ← 경로가 교차!
                 /\
                /  \
x_0[b] --------/    \-------> x_1[강아지]
```

두 직선 경로가 중간에서 교차할 수 있어요.

교차 지점에서:
- 같은 위치 $x_t$인데
- 가야 할 방향이 완전히 다름

모델 입장에서 혼란스럽습니다! 같은 입력인데 다른 출력을 내야 하니까요.

#### 이로 인한 문제들

| 문제 | 설명 |
|------|------|
| 학습 불안정 | 같은 $x_t$에 대해 상반된 gradient |
| 흐릿한 생성 | 모델이 "평균"을 내버림 (양쪽 방향의 중간) |
| Mode collapse | 특정 영역에서 방향을 못 정함 |

#### 이미지 생성 예시

```
노이즈 A -----> 고양이 이미지
              \
               X (교차!)
              /
노이즈 B -----> 강아지 이미지
```

교차 지점의 $x_t$에서 모델이 "고양이 방향? 강아지 방향?" 혼란 → 흐릿한 중간 이미지 생성 가능

### 8.4 해결책: Mini-batch Optimal Transport

랜덤 매칭 대신 최적 매칭을 하면 경로 교차가 줄어듭니다!

#### 기본 방식 vs OT 방식

```python
# 랜덤 매칭 (기본)
x_0 = torch.randn(batch_size, dim)
x_1 = sample_data(batch_size)
# x_0[i]와 x_1[i]가 그냥 순서대로 쌍이 됨

# Mini-batch OT 매칭 (개선)
x_0 = torch.randn(batch_size, dim)
x_1 = sample_data(batch_size)
assignment = compute_ot_assignment(x_0, x_1)  # 최적 매칭 계산
x_1 = x_1[assignment]  # 재배열해서 쌍 맺기
```

#### OT 매칭이 하는 일

"어떤 노이즈가 어떤 데이터로 가야 전체 이동 거리가 최소인가?"

```
OT 매칭 전 (랜덤):           OT 매칭 후:
x_0[a] ---\  /---> x_1[먼 것]    x_0[a] -----> x_1[가까운 것]
           \/                                              (교차 최소화!)
           /\
x_0[b] ---/  \---> x_1[먼 것]    x_0[b] -----> x_1[가까운 것]
```

가까운 것끼리 매칭하면 직선들이 덜 교차합니다!

#### 구현 예시

```python
from scipy.optimize import linear_sum_assignment

def compute_ot_assignment(x_0, x_1):
    """Mini-batch OT: 최적 매칭 계산"""
    # 거리 행렬 계산
    cost_matrix = torch.cdist(x_0, x_1, p=2)  # [batch, batch]
    
    # Hungarian algorithm으로 최적 매칭
    _, assignment = linear_sum_assignment(cost_matrix.cpu().numpy())
    
    return torch.tensor(assignment)

# 사용
x_0 = torch.randn(batch_size, dim)
x_1 = sample_data(batch_size)
assignment = compute_ot_assignment(x_0, x_1)
x_1_matched = x_1[assignment]  # 최적으로 재배열된 데이터

# 이제 x_0[i]와 x_1_matched[i]가 최적의 쌍!
x_t = (1 - t) * x_0 + t * x_1_matched
target = x_1_matched - x_0
```

### 8.5 Diffusion은 이 문제가 적다

Diffusion의 곡선 경로는 노이즈를 점점 추가하는 방식이라 경로 교차가 덜 심합니다.

| 방법 | 경로 교차 | 스텝 수 | Trade-off |
|------|----------|---------|-----------|
| Flow Matching (직선) | 있음 | 적음 (10-50) | 빠르지만 교차 문제 |
| Diffusion (곡선) | 적음 | 많음 (50-1000) | 느리지만 안정적 |
| Flow Matching + OT | 줄어듦 | 적음 | 최적의 균형 |

### 8.6 다른 경로도 가능하다

Flow Matching은 직선만 되는 게 아닙니다. 다른 경로도 선택 가능함:

```python
from flow_matching.path.scheduler import (
    CondOTScheduler,   # 직선 (Optimal Transport)
    VPScheduler,       # Diffusion 스타일 곡선
)

# 직선 경로
path_linear = AffineProbPath(scheduler=CondOTScheduler())

# 곡선 경로 (Diffusion 유사)
path_curved = AffineProbPath(scheduler=VPScheduler())
```

직선이 "유일한 정답"이 아니라 "효율적인 선택"라고 볼 수 있습니다.

### 8.7 정리

| 질문 | 답 |
|------|-----|
| 직선이어도 되나? | 네, 경계 조건($t=0$, $t=1$)만 맞으면 됨 |
| 왜 직선을 선택하나? | Optimal Transport 해 = 가장 효율적 |
| 1:1 매칭인가? | 아니요, 매 배치 랜덤 매칭 |
| 직선의 단점? | 경로 교차 → 학습 혼란, 흐릿한 생성 가능 |
| 해결책? | Mini-batch OT로 최적 매칭 |
| 곡선은 안 되나? | 됨! VP 스케줄러 등 선택 가능 |

---

## 9. 로보틱스 응용

### 9.1 Diffusion Policy vs Flow Matching Policy

로봇 제어에서 action sequence를 생성할 때:

| 구분 | Diffusion Policy | Flow Matching Policy |
|------|------------------|----------------------|
| 접근법 | Action을 denoising으로 생성 | Action을 ODE flow로 생성 |
| Inference 속도 | 느림 (50-100 스텝) | 빠름 (10-20 스텝) |
| 재현성 | 확률적 (매번 다름) | 결정론적 (같은 결과) |
| Real-time 적합성 | 어려움 | 적합 |

### 9.2 Flow Matching for Robot Action Generation

```python
import torch
import torch.nn as nn
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver

class RobotFlowPolicy(nn.Module):
    """
    Observation → Action Sequence 생성
    """
    def __init__(self, obs_dim, action_dim, horizon, hidden_dim=256):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Velocity network (conditioned on observation)
        self.velocity_net = nn.Sequential(
            nn.Linear(action_dim * horizon + hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * horizon)
        )
        
        self.path = AffineProbPath(scheduler=CondOTScheduler())
    
    def forward(self, action_t, t, obs_embed):
        """Velocity 예측"""
        # action_t: [B, horizon * action_dim]
        # t: [B, 1]
        # obs_embed: [B, hidden_dim]
        x = torch.cat([action_t, obs_embed, t], dim=-1)
        return self.velocity_net(x)
    
    def compute_loss(self, obs, actions):
        """
        Args:
            obs: [B, obs_dim] - 관측
            actions: [B, horizon, action_dim] - 목표 action sequence
        """
        batch_size = obs.shape[0]
        
        # Flatten actions
        x_1 = actions.view(batch_size, -1)  # [B, horizon * action_dim]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, 1, device=obs.device)
        
        # Path sample
        path_sample = self.path.sample(t=t.squeeze(-1), x_0=x_0, x_1=x_1)
        
        # Observation embedding
        obs_embed = self.obs_encoder(obs)
        
        # Velocity prediction
        v_pred = self.forward(path_sample.x_t, t, obs_embed)
        
        # Loss
        loss = (v_pred - path_sample.dx_t).pow(2).mean()
        return loss
    
    @torch.no_grad()
    def sample_actions(self, obs, n_steps=20):
        """
        Observation으로부터 action sequence 생성
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # Observation embedding
        obs_embed = self.obs_encoder(obs)
        
        # 노이즈에서 시작
        x = torch.randn(batch_size, self.horizon * self.action_dim, device=device)
        
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.full((batch_size, 1), i / n_steps, device=device)
            v = self.forward(x, t, obs_embed)
            x = x + v * dt  # Euler step
        
        # Reshape to [B, horizon, action_dim]
        actions = x.view(batch_size, self.horizon, self.action_dim)
        return actions

# 사용 예시
obs_dim = 64
action_dim = 7  # 7-DoF robot arm
horizon = 16    # 16 timesteps ahead

policy = RobotFlowPolicy(obs_dim, action_dim, horizon)

# 학습
obs = torch.randn(32, obs_dim)
actions = torch.randn(32, horizon, action_dim)
loss = policy.compute_loss(obs, actions)

# 추론 (20 스텝으로 빠르게!)
obs_test = torch.randn(1, obs_dim)
generated_actions = policy.sample_actions(obs_test, n_steps=20)
print(f"Generated actions: {generated_actions.shape}")  # [1, 16, 7]
```

### 9.3 왜 로보틱스에서 Flow Matching이 유리한가?

1. 빠른 inference: Real-time control에 필수 (10-20 스텝 vs 50-100 스텝)

2. 결정론적 출력: 같은 observation → 같은 action (디버깅 용이)

3. 부드러운 trajectory: Optimal Transport 경로로 자연스러운 동작 생성

4. Consistency: Diffusion의 확률적 특성으로 인한 action jittering 없음

---

## 참고 자료

### 논문
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., 2022)
- [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) (Meta, 2024)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) (Chi et al., 2023)

### 코드
- [Meta flow_matching 라이브러리](https://github.com/facebookresearch/flow_matching)
- [Flow Matching 공식 문서](https://facebookresearch.github.io/flow_matching/)

### 추가 학습
- [Cambridge MLG Blog: An Introduction to Flow Matching](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)

---

## 요약

| 개념 | Diffusion | Flow Matching |
|------|-----------|---------------|
| 학습 목표 | 노이즈 예측 | Velocity 예측 |
| 경로 | 곡선 (노이즈 스케줄) | 직선 (Optimal Transport) |
| 샘플링 | 확률적 (SDE) | 결정론적 (ODE) |
| 스텝 수 | 많음 (50-1000) | 적음 (10-50) |
| 재현성 | 없음 | 있음 |
| 로보틱스 적합성 | 보통 | 높음 |

Flow Matching은 "노이즈에서 데이터까지 직선으로 가는 방향"을 학습하는 방법입니다.
