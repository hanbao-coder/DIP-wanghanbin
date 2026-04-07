#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美颜算法插件模块
包含美颜算法基类和4种基础美颜插件
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod


class BeautyBase(ABC):
    """美颜算法基类 - 所有美颜插件必须继承此类"""
    
    def __init__(self):
        self.name = "基础美颜"
        self.description = "美颜算法基类"
        self.intensity = 0.5  # 默认强度 0.5（中等）
    
    def set_intensity(self, intensity):
        """
        设置美颜强度
        
        参数:
            intensity: 强度值，范围0.0-1.0
        """
        self.intensity = max(0.0, min(1.0, intensity))
    
    def get_intensity(self):
        """
        获取当前美颜强度
        
        返回:
            强度值，范围0.0-1.0
        """
        return self.intensity
    
    @abstractmethod
    def process(self, image, intensity=None):
        """
        美颜处理主方法
        
        参数:
            image: 输入图像（RGB格式）
            intensity: 强度值，如果为None则使用self.intensity
            
        返回:
            处理后的图像（RGB格式）
        """
        pass


class BilateralFilterBeauty(BeautyBase):
    """双边滤波磨皮算法"""
    
    def __init__(self):
        super().__init__()
        self.name = "双边滤波磨皮"
        self.description = "使用双边滤波进行皮肤平滑处理，保留边缘细节"
    
    def process(self, image, intensity=None):
        """
        双边滤波磨皮处理
        
        参数:
            image: 输入图像（RGB格式）
            intensity: 强度值，如果为None则使用self.intensity
            
        返回:
            磨皮后的图像
        """
        try:
            if intensity is None:
                intensity = self.intensity
            
            # 根据强度调整参数
            # 强度0.0: 轻微磨皮，强度1.0: 强烈磨皮
            base_d = 5
            base_sigma_color = 25
            base_sigma_space = 25
            
            max_d = 15
            max_sigma_color = 150
            max_sigma_space = 150
            
            d = int(base_d + intensity * (max_d - base_d))
            sigma_color = base_sigma_color + intensity * (max_sigma_color - base_sigma_color)
            sigma_space = base_sigma_space + intensity * (max_sigma_space - base_sigma_space)
            
            # 确保d为奇数
            if d % 2 == 0:
                d += 1
            
            # 应用双边滤波
            smoothed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
            # 根据强度与原图混合
            if intensity < 1.0:
                # 线性混合：强度越高，磨皮效果越明显
                result = cv2.addWeighted(image, 1.0 - intensity, smoothed, intensity, 0)
            else:
                result = smoothed
            
            return result
            
        except Exception as e:
            print(f"双边滤波磨皮失败: {str(e)}")
            return image


class SkinWhiteningBeauty(BeautyBase):
    """皮肤美白算法"""
    
    def __init__(self):
        super().__init__()
        self.name = "皮肤美白"
        self.description = "通过HSV空间调整亮度和饱和度实现皮肤美白"
    
    def process(self, image, intensity=None):
        """
        皮肤美白处理
        
        参数:
            image: 输入图像（RGB格式）
            intensity: 强度值，如果为None则使用self.intensity
            
        返回:
            美白后的图像
        """
        try:
            if intensity is None:
                intensity = self.intensity
            
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 分离通道
            h, s, v = cv2.split(hsv)
            
            # 根据强度调整参数
            # 强度0.0: 轻微美白，强度1.0: 强烈美白
            base_brightness = 5
            base_saturation_reduction = 2
            
            max_brightness = 30
            max_saturation_reduction = 15
            
            brightness_boost = int(base_brightness + intensity * (max_brightness - base_brightness))
            saturation_reduction = int(base_saturation_reduction + intensity * (max_saturation_reduction - base_saturation_reduction))
            
            # 调整亮度和饱和度
            # 增加亮度（美白效果）
            v = cv2.add(v, brightness_boost)
            v = np.clip(v, 0, 255)
            
            # 降低饱和度（使皮肤更自然）
            s = cv2.subtract(s, saturation_reduction)
            s = np.clip(s, 0, 255)
            
            # 合并通道
            hsv_enhanced = cv2.merge([h, s, v])
            
            # 转换回RGB
            result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
            
            # 根据强度与原图混合
            if intensity < 0.8:
                # 线性混合：强度越高，美白效果越明显
                result = cv2.addWeighted(image, 1.0 - intensity, result, intensity, 0)
            
            return result
            
        except Exception as e:
            print(f"皮肤美白失败: {str(e)}")
            return image


class ImageSharpeningBeauty(BeautyBase):
    """图像锐化算法"""
    
    def __init__(self):
        super().__init__()
        self.name = "图像锐化"
        self.description = "使用非锐化掩模(Unsharp Mask)增强图像细节"
    
    def process(self, image, intensity=None):
        """
        图像锐化处理
        
        参数:
            image: 输入图像（RGB格式）
            intensity: 强度值，如果为None则使用self.intensity
            
        返回:
            锐化后的图像
        """
        try:
            if intensity is None:
                intensity = self.intensity
            
            # 根据强度调整参数
            # 强度0.0: 轻微锐化，强度1.0: 强烈锐化
            base_sigma = 1.0
            max_sigma = 5.0
            
            sigma = base_sigma + intensity * (max_sigma - base_sigma)
            
            # 高斯模糊（低通滤波）
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # 根据强度调整锐化参数
            # 强度0.0: 原图，强度1.0: 强烈锐化
            amount = intensity * 2.0  # 0.0-2.0范围
            
            # 非锐化掩模：原图 + amount * (原图 - 模糊图)
            unsharp_mask = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
            
            # 限制像素值在合理范围内
            result = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
            
            # 根据强度与原图混合（避免过度锐化）
            if intensity < 0.7:
                # 线性混合：强度越高，锐化效果越明显
                result = cv2.addWeighted(image, 1.0 - intensity, result, intensity, 0)
            
            return result
            
        except Exception as e:
            print(f"图像锐化失败: {str(e)}")
            return image


class CLAHEEnhancementBeauty(BeautyBase):
    """局部对比度增强算法"""
    
    def __init__(self):
        super().__init__()
        self.name = "局部对比度增强"
        self.description = "使用CLAHE算法增强局部对比度，改善图像质量"
    
    def process(self, image, intensity=None):
        """
        局部对比度增强处理
        
        参数:
            image: 输入图像（RGB格式）
            intensity: 强度值，如果为None则使用self.intensity
            
        返回:
            增强后的图像
        """
        try:
            if intensity is None:
                intensity = self.intensity
            
            # 根据强度调整CLAHE参数
            # 强度0.0: 轻微增强，强度1.0: 强烈增强
            base_clip_limit = 1.0
            max_clip_limit = 4.0
            
            clip_limit = base_clip_limit + intensity * (max_clip_limit - base_clip_limit)
            
            # 转换为Lab颜色空间
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # 分离通道
            l, a, b = cv2.split(lab)
            
            # 对亮度通道应用CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # 合并通道
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            
            # 转换回RGB
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            # 根据强度与原图混合
            if intensity < 0.6:
                # 线性混合：强度越高，增强效果越明显
                result = cv2.addWeighted(image, 1.0 - intensity, result, intensity, 0)
            
            return result
            
        except Exception as e:
            print(f"局部对比度增强失败: {str(e)}")
            return image


# 美颜插件注册管理器
beauty_plugins = {}


def register_beauty_plugin(name, plugin_class):
    """
    注册美颜插件
    
    参数:
        name: 插件名称（唯一标识）
        plugin_class: 插件类
    """
    if issubclass(plugin_class, BeautyBase):
        beauty_plugins[name] = plugin_class
        print(f"美颜插件注册成功: {name}")
    else:
        print(f"注册失败: {plugin_class} 不是BeautyBase的子类")


def unregister_beauty_plugin(name):
    """
    注销美颜插件
    
    参数:
        name: 插件名称
    """
    if name in beauty_plugins:
        del beauty_plugins[name]
        print(f"美颜插件注销成功: {name}")
    else:
        print(f"注销失败: 未找到插件 {name}")


def get_all_beauty_plugins():
    """
    获取所有已注册的美颜插件
    
    返回:
        插件字典
    """
    return beauty_plugins.copy()


# 自动注册4种基础美颜插件
register_beauty_plugin("bilateral_filter", BilateralFilterBeauty)
register_beauty_plugin("skin_whitening", SkinWhiteningBeauty)
register_beauty_plugin("image_sharpening", ImageSharpeningBeauty)
register_beauty_plugin("clahe_enhancement", CLAHEEnhancementBeauty)


# 测试代码
if __name__ == "__main__":
    # 测试插件注册
    print("已注册的美颜插件:")
    for name, plugin_class in beauty_plugins.items():
        plugin_instance = plugin_class()
        print(f"- {name}: {plugin_instance.name} - {plugin_instance.description}")
    
    # 测试图像处理
    import numpy as np
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("\n测试美颜处理:")
    for name, plugin_class in beauty_plugins.items():
        try:
            plugin = plugin_class()
            result = plugin.process(test_image)
            print(f"{plugin.name}: 处理成功，输出尺寸 {result.shape}")
        except Exception as e:
            print(f"{plugin.name}: 处理失败 - {str(e)}")