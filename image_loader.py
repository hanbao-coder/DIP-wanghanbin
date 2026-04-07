#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片加载器类
负责图片的加载、保存和格式转换
"""

import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageLoader:
    """图片加载器类 - 负责图片的加载、保存和格式转换"""
    
    def __init__(self):
        pass
    
    def load_image(self, file_path):
        """
        加载图片文件
        
        参数:
            file_path: 图片文件路径
            
        返回:
            numpy数组格式的图片，失败返回None
        """
        try:
            # 方法1：使用Pillow加载图片（更好的中文路径支持）
            from PIL import Image
            pil_image = Image.open(file_path)
            
            # 转换为RGB模式（处理RGBA等格式）
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为numpy数组
            image = np.array(pil_image)
            
            # 方法2：如果Pillow失败，尝试使用OpenCV的改进方法
            if image is None or image.size == 0:
                # 使用numpy.fromfile方法解决中文路径问题
                image_array = np.fromfile(file_path, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is not None:
                    # 转换颜色空间 BGR -> RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            print(f"加载图片失败: {str(e)}")
            # 尝试最后的备用方法
            try:
                # 使用OpenCV的imdecode方法
                image_array = np.fromfile(file_path, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            except:
                return None
    
    def save_image(self, image, file_path):
        """
        保存图片文件
        
        参数:
            image: numpy数组格式的图片
            file_path: 保存路径
            
        返回:
            成功返回True，失败返回False
        """
        try:
            # 转换颜色空间 RGB -> BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_to_save = image
            
            # 方法1：使用OpenCV的imencode方法解决中文路径问题
            file_extension = file_path.split('.')[-1].lower()
            
            # 根据文件扩展名选择编码格式
            if file_extension in ['jpg', 'jpeg']:
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
            elif file_extension == 'png':
                encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            else:
                encode_param = []
            
            # 使用imencode保存图片
            success, encoded_image = cv2.imencode('.' + file_extension, image_to_save, encode_param)
            if success:
                encoded_image.tofile(file_path)
                return True
            else:
                # 方法2：如果imencode失败，尝试直接保存
                success = cv2.imwrite(file_path, image_to_save)
                return success
            
        except Exception as e:
            print(f"保存图片失败: {str(e)}")
            return False
    
    def cv2_to_qpixmap(self, cv_image):
        """
        将OpenCV图像转换为QPixmap
        
        参数:
            cv_image: OpenCV格式的图像（RGB）
            
        返回:
            QPixmap对象
        """
        try:
            # 获取图像尺寸和通道数
            height, width, channel = cv_image.shape
            
            # 计算每行字节数
            bytes_per_line = 3 * width
            
            # 创建QImage
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 转换为QPixmap
            pixmap = QPixmap.fromImage(q_image)
            
            return pixmap
            
        except Exception as e:
            print(f"图像格式转换失败: {str(e)}")
            # 返回空QPixmap
            return QPixmap()
    
    def qpixmap_to_cv2(self, pixmap):
        """
        将QPixmap转换为OpenCV图像
        
        参数:
            pixmap: QPixmap对象
            
        返回:
            OpenCV格式的图像（RGB）
        """
        try:
            # 转换为QImage
            q_image = pixmap.toImage()
            
            # 转换为numpy数组
            ptr = q_image.bits()
            ptr.setsize(q_image.byteCount())
            
            # 重新形状为图像
            arr = np.array(ptr).reshape(q_image.height(), q_image.width(), 4)
            
            # 转换为RGB（去掉alpha通道）
            cv_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            
            return cv_image
            
        except Exception as e:
            print(f"QPixmap转OpenCV失败: {str(e)}")
            return None
    
    def resize_image(self, image, max_width=800, max_height=600):
        """
        调整图片尺寸，保持宽高比
        
        参数:
            image: 输入图像
            max_width: 最大宽度
            max_height: 最大高度
            
        返回:
            调整后的图像
        """
        try:
            height, width = image.shape[:2]
            
            # 计算缩放比例
            scale = min(max_width / width, max_height / height)
            
            # 如果图片已经小于最大尺寸，不进行缩放
            if scale >= 1:
                return image
            
            # 计算新尺寸
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 使用插值方法进行缩放
            resized_image = cv2.resize(image, (new_width, new_height), 
                                     interpolation=cv2.INTER_AREA)
            
            return resized_image
            
        except Exception as e:
            print(f"调整图片尺寸失败: {str(e)}")
            return image
    
    def convert_to_grayscale(self, image):
        """
        将彩色图像转换为灰度图像
        
        参数:
            image: 输入彩色图像
            
        返回:
            灰度图像
        """
        try:
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                return gray_image
            else:
                return image
                
        except Exception as e:
            print(f"转换为灰度图像失败: {str(e)}")
            return image
    
    def get_image_info(self, image):
        """
        获取图像基本信息
        
        参数:
            image: 输入图像
            
        返回:
            包含图像信息的字典
        """
        try:
            info = {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2] if len(image.shape) == 3 else 1,
                'dtype': str(image.dtype),
                'size': f"{image.shape[1]}x{image.shape[0]}"
            }
            
            return info
            
        except Exception as e:
            print(f"获取图像信息失败: {str(e)}")
            return {}


# 测试代码
if __name__ == "__main__":
    # 创建加载器实例
    loader = ImageLoader()
    
    # 测试加载图片
    test_image = loader.load_image("test.jpg")
    if test_image is not None:
        print("图片加载成功")
        print(f"图片尺寸: {test_image.shape}")
        
        # 测试获取图像信息
        info = loader.get_image_info(test_image)
        print(f"图像信息: {info}")
        
        # 测试转换为灰度图
        gray_image = loader.convert_to_grayscale(test_image)
        print(f"灰度图尺寸: {gray_image.shape}")
        
        # 测试调整尺寸
        resized_image = loader.resize_image(test_image, 400, 300)
        print(f"调整后尺寸: {resized_image.shape}")
        
    else:
        print("图片加载失败")