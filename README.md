# 图像处理美颜与质量评价系统

## 项目简介

这是一个基于Python + PyQt5开发的图像处理美颜与质量评价系统，专为数字图像处理课程或毕业设计项目设计。系统采用面向对象和插件化架构，支持4种基础美颜算法和10种质量评价指标。

## 功能特性

### 美颜功能（4种基础算法）
- **双边滤波磨皮**：皮肤平滑处理，保留边缘细节
- **皮肤美白**：HSV空间调整亮度和饱和度
- **图像锐化**：非锐化掩模增强图像细节
- **局部对比度增强**：CLAHE算法改善图像质量

### 质量评价（10项基础指标）
- **图像模糊度**：Laplacian方差检测
- **图像亮度均值**：整体亮度评估
- **图像对比度**：像素值标准差
- **图像锐度**：Tenengrad梯度方法
- **高光暗区比例**：过曝和欠曝区域检测
- **人脸检测**：人脸数量、位置识别
- **人脸区域清晰度**：人脸区域Laplacian方差
- **人脸亮度对比度**：人脸区域亮度和对比度
- **人脸居中程度**：人脸在图像中的位置评估
- **背景均匀度**：背景区域均匀程度检测

## 技术架构

### 核心类设计
1. **ImageLoader**：图片加载、保存、格式转换
2. **BeautyBase**：美颜算法基类，支持插件注册
3. **QualityBase**：质量评价基类，支持插件注册
4. **MainWindow**：PyQt5主界面与业务逻辑

### 插件化机制
- 使用字典管理插件：`beauty_plugins`、`quality_plugins`
- 提供注册、注销、获取插件方法
- 新增算法无需修改主逻辑，直接注册即可

## 安装说明

### 环境要求
- Python 3.6+
- Windows/Linux/macOS

### 一键安装依赖
```bash
pip install -r requirements.txt
```

### 手动安装依赖
```bash
pip install PyQt5 opencv-python numpy Pillow
```

## 使用方法

### 运行程序
```bash
python main.py
```

### 操作流程
1. **打开图片**：点击"打开图片"按钮选择图像文件
2. **选择美颜效果**：勾选需要的美颜选项（可多选）
3. **执行美颜**：点击"执行美颜"按钮应用效果
4. **质量评价**：点击"质量评价"按钮查看图像质量分析
5. **保存结果**：点击"保存结果"按钮保存处理后的图像
6. **重置系统**：点击"重置"按钮清空所有状态

### 界面布局
- **顶部按钮区**：功能操作按钮
- **中间图片显示区**：左右双图显示（原图/效果图）
- **美颜选项区**：4种美颜效果复选框
- **底部结果显示区**：质量评价分数与结论滚动显示

## 项目结构

```
图像处理大作业/
├── main.py                 # 主程序入口
├── image_loader.py         # 图片加载器类
├── beauty_plugins.py       # 美颜算法插件
├── quality_plugins.py      # 质量评价插件
├── requirements.txt        # 依赖包列表
├── README.md              # 项目说明文档
└── 人像照片/              # 测试图片目录
    ├── 101ab5858186dccc3ebfe7fe230128da.png
    ├── 6499cac3398c839c9457c87cb1259260.jpg
    └── 7497020bfb6b4508869d16be3912777e.jpg
```

## 扩展开发

### 添加新的美颜算法
1. 创建新类继承`BeautyBase`
2. 实现`process()`方法
3. 在`beauty_plugins.py`中注册插件

```python
class NewBeautyPlugin(BeautyBase):
    def __init__(self):
        super().__init__()
        self.name = "新美颜算法"
        
    def process(self, image):
        # 实现美颜逻辑
        return processed_image

# 注册插件
register_beauty_plugin("new_beauty", NewBeautyPlugin)
```

### 添加新的质量评价指标
1. 创建新类继承`QualityBase`
2. 实现`evaluate()`方法
3. 在`quality_plugins.py`中注册插件

```python
class NewQualityPlugin(QualityBase):
    def __init__(self):
        super().__init__()
        self.name = "新质量指标"
        
    def evaluate(self, image):
        # 实现评价逻辑
        return {'score': score, 'assessment': '评价结果'}

# 注册插件
register_quality_plugin("new_quality", NewQualityPlugin)
```

## 注意事项

1. **图片格式支持**：JPEG、PNG、BMP、TIFF等常见格式
2. **图片大小限制**：建议使用2MB以内的图片以获得最佳性能
3. **人脸检测要求**：正面人脸检测效果最佳，侧脸可能无法识别
4. **异常处理**：系统包含完整的异常捕获和用户提示

## 故障排除

### 常见问题
1. **无法导入PyQt5**：检查Python版本和PyQt5安装
2. **OpenCV导入错误**：重新安装opencv-python
3. **人脸检测失败**：确保图片中包含清晰的人脸
4. **内存不足**：处理过大图片时可能出现，建议缩小图片尺寸

### 调试模式
如需查看详细错误信息，可修改代码中的异常处理部分。

## 版本信息

- **版本**：v1.0.0
- **发布日期**：2025-04-05
- **作者**：Python图像处理工程师
- **许可证**：仅供学习使用

## 联系方式

如有问题或建议，请联系项目维护者。

---

**注意**：本项目仅供学习和课程设计使用，请勿用于商业用途。