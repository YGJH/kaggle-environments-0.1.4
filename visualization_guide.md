# ConnectX 可视化指南

## 问题解决方案

### 1. 中文字体缺失警告
**问题**: matplotlib显示中文时出现字体缺失警告
**解决方案**: 
- 已将所有可视化文本改为英文
- 添加了字体配置，使用系统可用字体

### 2. 目录不存在错误  
**问题**: 保存文件时目录不存在
**解决方案**:
- 自动创建必要的目录（game_visualizations, game_videos）
- 增加了错误处理

## 可视化选项

### 1. Text Visualization (推荐)
```python
# 在配置文件中设置
visualization:
  type: "text"
```
- **优点**: 无字体问题，快速，资源占用少
- **缺点**: 只显示最终状态和移动历史
- **适用**: 快速调试，服务器环境

### 2. Matplotlib Visualization  
```python
visualization:
  type: "matplotlib"
```
- **优点**: 静态图片，显示多个游戏状态
- **缺点**: 可能有字体问题，需要GUI环境
- **适用**: 详细分析，报告生成

### 3. Video Visualization
```python
visualization:
  type: "video"
```
- **优点**: 动态显示整个游戏过程
- **缺点**: 需要opencv，文件较大
- **适用**: 演示，详细观察策略

### 4. All (全部)
```python
visualization:
  type: "all"
```
- 同时生成文本、图片和视频
- 适用于完整记录

## 使用方法

### 配置文件设置 (config.yaml)
```yaml
# 添加可视化配置
visualization:
  type: "text"  # 选择: "text", "matplotlib", "video", "all"
  
# 其他现有配置...
training:
  num_episodes: 100000
  # ...
```

### 手动调用
```python
trainer = ConnectXTrainer("config.yaml")

# 文本可视化（推荐，无字体问题）
trainer.demo_game_with_visualization(episode_num=100, visualization_type="text")

# 视频可视化
trainer.demo_game_with_visualization(episode_num=100, visualization_type="video")

# 所有类型
trainer.demo_game_with_visualization(episode_num=100, visualization_type="all")
```

## 输出文件位置

- **文本输出**: 直接显示在控制台
- **图片文件**: `game_visualizations/episode_XXX_AgentA_vs_AgentB_timestamp.png`
- **视频文件**: `game_videos/episode_XXX_AgentA_vs_AgentB_timestamp.mp4`

## 推荐设置

### 服务器环境 (无GUI)
```yaml
visualization:
  type: "text"
```

### 本地开发环境
```yaml
visualization:
  type: "video"  # 或 "all"
```

### 调试模式
```yaml
visualization:
  type: "text"  # 快速，无额外依赖
```

## 依赖安装

```bash
# matplotlib可视化 (可选)
pip install matplotlib

# 视频可视化 (可选)  
pip install opencv-python

# 文本可视化无需额外依赖
```

## 故障排除

1. **字体警告**: 使用 `type: "text"` 避免
2. **目录错误**: 已自动解决，会自动创建目录
3. **OpenCV错误**: 安装 `pip install opencv-python`
4. **显示问题**: 在服务器上使用 `type: "text"`

现在训练将使用文本可视化作为默认选项，避免字体问题！
