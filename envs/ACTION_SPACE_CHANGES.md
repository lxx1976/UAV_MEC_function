# 动作空间恢复：2维 → 3维

## ✅ 已完成的修改

### 1. **env_core.py - 动作维度声明**
```python
# 修改前：
self.action_dim = 2

# 修改后：
self.action_dim = 3
```

### 2. **env_core.py - 类文档字符串**
```python
# 修改前：
"""
Action (continuous 2D per UAV):
- action[0]: mapped to movement action
- action[1]: mapped to terminal selection
"""

# 修改后：
"""
Action (continuous 3D per UAV):
- action[0]: mapped to movement action (7 bins)
- action[1]: mapped to service decision (0=不服务, 1=服务)
- action[2]: mapped to terminal selection (num_terminals bins)
"""
```

### 3. **env_core.py - 动作向量填充**
```python
# 修改前：
if action_vec.shape[0] < 2:
    pad = np.zeros(2 - action_vec.shape[0], dtype=np.float32)

# 修改后：
if action_vec.shape[0] < 3:
    pad = np.zeros(3 - action_vec.shape[0], dtype=np.float32)
```

### 4. **env_core.py - 动作解码**
```python
# 修改前：
movement_action, horizontal_action, vertical_action, terminal_id = decode_action_vector(...)

# 修改后：
movement_action, horizontal_action, vertical_action, service_decision, terminal_id = decode_action_vector(...)
```

### 5. **env_core.py - 服务逻辑**
```python
# 修改前：直接处理选中的终端
terminal = self.terminals[terminal_id]
if not invalid_service:
    # 计算处理量...

# 修改后：先检查是否决定服务
if service_decision > 0:  # 只有当决定服务时才处理
    terminal = self.terminals[terminal_id]
    if not invalid_service:
        # 计算处理量...
else:
    # 不服务任何终端
    invalid_service = False
```

### 6. **function.py - decode_action_vector 函数**
```python
# 修改前：
def decode_action_vector(...) -> Tuple[int, int, int, int]:
    """Decode a 2D continuous action vector"""
    movement_action = continuous_to_discrete_index(action_vector[0], movement_bins)
    terminal_id = continuous_to_discrete_index(action_vector[1], max(num_terminals, 1))
    return movement_action, horizontal_action, vertical_action, terminal_id

# 修改后：
def decode_action_vector(...) -> Tuple[int, int, int, int, int]:
    """Decode a 3D continuous action vector"""
    movement_action = continuous_to_discrete_index(action_vector[0], movement_bins)
    service_decision = continuous_to_discrete_index(action_vector[1], 2)  # 0 or 1
    terminal_id = continuous_to_discrete_index(action_vector[2], max(num_terminals, 1))
    return movement_action, horizontal_action, vertical_action, service_decision, terminal_id
```

### 7. **env_core.py - info 字典**
添加了 `service_decision` 字段到 info 中，方便调试和分析。

---

## 📊 动作空间对比

### 2维动作空间（旧版本）
```
动作维度: 2
- action[0] ∈ [-1, 1] → 移动动作 (0-6)
- action[1] ∈ [-1, 1] → 终端选择 (0-5)

问题：UAV 每步都必须服务某个终端，无法选择"不服务"
```

### 3维动作空间（新版本）✅
```
动作维度: 3
- action[0] ∈ [-1, 1] → 移动动作 (0-6)
  - 0: 向上(y+) + 悬停
  - 1: 向下(y-) + 悬停
  - 2: 向左(x-) + 悬停
  - 3: 向右(x+) + 悬停
  - 4: 不动 + 上升
  - 5: 不动 + 下降
  - 6: 不动 + 悬停

- action[1] ∈ [-1, 1] → 是否服务 (0/1)
  - 0: 不服务任何终端（只移动）
  - 1: 服务选中的终端

- action[2] ∈ [-1, 1] → 终端选择 (0-5)
  - 选择要服务的终端ID

优势：
1. UAV 可以选择"只移动不服务"，更灵活
2. 避免无意义的服务（如距离太远、终端已完成等）
3. 更符合实际场景的决策过程
```

---

## 🎯 行为变化

### 场景1：UAV 决定不服务
```python
action = [0.5, -0.8, 0.3]  # 移动=3(向右), 服务=0(不服务), 终端=2
→ UAV 向右移动，但不处理任何终端
→ processed_bits = 0
→ 只消耗推进能耗，无计算和通信能耗
```

### 场景2：UAV 决定服务
```python
action = [0.5, 0.8, 0.3]  # 移动=3(向右), 服务=1(服务), 终端=2
→ UAV 向右移动，并服务终端2
→ processed_bits > 0 (如果终端未完成)
→ 消耗推进+计算+通信能耗
```

### 场景3：服务已完成的终端
```python
action = [0.5, 0.8, 0.3]  # 服务=1, 终端=2
→ 如果终端2已完成
→ processed_bits = 0
→ invalid_service = True
→ 受到惩罚 (self.invalid_service_penalty = 0.5)
```

---

## ⚠️ 需要注意的地方

### 1. 训练算法配置
确保你的 MAPPO 算法配置中动作空间维度设置为 3：
```python
# 在算法配置文件中
action_dim = 3  # 不是 2！
```

### 2. 奖励设计
现在 UAV 可以选择不服务，需要确保奖励设计合理：
- 不服务时：只有推进能耗惩罚
- 服务时：数据奖励 + 服务奖励 - 能耗惩罚
- 服务已完成终端：额外惩罚

### 3. 策略学习
agent 需要学习：
- 什么时候应该服务（距离近、终端未完成、电量充足）
- 什么时候应该只移动（距离远、电量不足、终端已完成）

---

## 🔍 调试建议

查看 info 字典中的 `service_decision` 字段：
```python
info = {
    'selected_terminal': 2,
    'service_decision': 1,  # ← 新增字段
    'processed_bits': 1500000.0,
    'invalid_service': False,
    ...
}
```

统计每个 episode 中：
- 服务次数 vs 不服务次数
- 无效服务次数（服务已完成终端）
- 平均处理量

这可以帮助你判断 agent 是否学到了合理的服务策略。
