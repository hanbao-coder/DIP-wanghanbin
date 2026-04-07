#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理美颜与质量评价系统
作者：Python图像处理工程师
日期：2025-04-05
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QCheckBox, 
                            QTextEdit, QFileDialog, QMessageBox, QScrollArea,
                            QGroupBox, QGridLayout, QSlider)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from image_loader import ImageLoader
from beauty_plugins import BeautyBase, beauty_plugins
from quality_plugins import QualityBase, quality_plugins


class MainWindow(QMainWindow):
    """主窗口类 - 负责界面布局和业务逻辑"""
    
    def __init__(self):
        super().__init__()
        self.image_loader = ImageLoader()
        self.original_image = None
        self.processed_image = None
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("图像处理美颜与质量评价系统")
        self.setGeometry(100, 100, 1200, 700)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部按钮区域
        self.create_top_buttons(main_layout)
        
        # 中间图片显示区域
        self.create_image_display(main_layout)
        
        # 美颜选项区域
        self.create_beauty_options(main_layout)
        
        # 底部评价结果显示区域
        self.create_result_display(main_layout)
        
    def create_top_buttons(self, layout):
        """创建顶部按钮区域"""
        button_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("打开图片")
        self.open_btn.clicked.connect(self.open_image)
        
        self.process_btn = QPushButton("执行美颜")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        
        self.quality_btn = QPushButton("质量评价")
        self.quality_btn.clicked.connect(self.evaluate_quality)
        self.quality_btn.setEnabled(False)
        
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_all)
        
        button_layout.addWidget(self.open_btn)
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.quality_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def create_image_display(self, layout):
        """创建图片显示区域"""
        image_layout = QHBoxLayout()
        
        # 原图显示区域
        original_group = QGroupBox("原图")
        original_layout = QVBoxLayout()
        self.original_label = QLabel("请打开图片")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(500, 300)
        self.original_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        original_layout.addWidget(self.original_label)
        original_group.setLayout(original_layout)
        
        # 处理结果显示区域
        processed_group = QGroupBox("效果图")
        processed_layout = QVBoxLayout()
        self.processed_label = QLabel("处理结果将显示在这里")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(500, 300)
        self.processed_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        processed_layout.addWidget(self.processed_label)
        processed_group.setLayout(processed_layout)
        
        image_layout.addWidget(original_group)
        image_layout.addWidget(processed_group)
        layout.addLayout(image_layout)
    
    def create_beauty_options(self, layout):
        """创建美颜选项区域"""
        beauty_group = QGroupBox("美颜选项")
        beauty_layout = QVBoxLayout()
        
        self.beauty_checkboxes = {}
        self.beauty_sliders = {}
        self.beauty_labels = {}  # 添加beauty_labels属性初始化
        
        # 创建4种基础美颜复选框和强度滑块
        beauty_options = [
            ("双边滤波磨皮", "bilateral_filter"),
            ("皮肤美白", "skin_whitening"),
            ("图像锐化", "image_sharpening"),
            ("局部对比度增强", "clahe_enhancement")
        ]
        
        for text, key in beauty_options:
            # 创建水平布局用于每个美颜选项
            option_layout = QHBoxLayout()
            
            # 创建复选框
            checkbox = QCheckBox(text)
            checkbox.setMinimumWidth(120)
            self.beauty_checkboxes[key] = checkbox
            option_layout.addWidget(checkbox)
            
            # 创建强度标签
            intensity_label = QLabel("强度: 50%")
            intensity_label.setMinimumWidth(60)
            option_layout.addWidget(intensity_label)
            self.beauty_labels[key] = intensity_label  # 保存标签引用
            
            # 创建强度滑块
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(50)  # 默认50%
            slider.setMinimumWidth(200)
            slider.setMaximumWidth(300)
            
            # 连接滑块值改变信号
            slider.valueChanged.connect(lambda value, label=intensity_label: 
                                      label.setText(f"强度: {value}%"))
            
            self.beauty_sliders[key] = slider
            option_layout.addWidget(slider)
            
            # 添加重置按钮
            reset_btn = QPushButton("重置")
            reset_btn.setMaximumWidth(60)
            reset_btn.clicked.connect(lambda checked, s=slider, l=intensity_label: 
                                     self.reset_slider(s, l))
            option_layout.addWidget(reset_btn)
            
            option_layout.addStretch()
            beauty_layout.addLayout(option_layout)
        
        beauty_group.setLayout(beauty_layout)
        layout.addWidget(beauty_group)
    
    def create_result_display(self, layout):
        """创建评价结果显示区域"""
        result_group = QGroupBox("质量评价结果")
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(250)  # 增加最小高度
        self.result_text.setMaximumHeight(400)  # 增加最大高度
        self.result_text.setPlaceholderText("质量评价结果将显示在这里...")
        
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
    
    def open_image(self):
        """打开图片文件 - 支持无限次更换图片"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", 
                "", 
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
            )
            
            if file_path:
                # 先加载新图片，再重置状态
                self.original_image = self.image_loader.load_image(file_path)
                if self.original_image is not None:
                    # 完全重置所有图像数据和状态（但保留新图片）
                    self._reset_image_state()
                    
                    # 显示原图
                    pixmap = self.image_loader.cv2_to_qpixmap(self.original_image)
                    self.original_label.setPixmap(pixmap.scaled(
                        500, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    
                    # 启用相关按钮
                    self.process_btn.setEnabled(True)
                    self.quality_btn.setEnabled(True)
                    
                    # 重置所有美颜滑块到默认值
                    self._reset_all_sliders()
                    
                    QMessageBox.information(self, "成功", f"图片加载成功: {os.path.basename(file_path)}")
                else:
                    QMessageBox.warning(self, "错误", "图片加载失败，请检查文件格式")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开图片时发生错误: {str(e)}")
    
    def _reset_all_image_data(self):
        """完全重置所有图像数据和缓存（用于重置按钮）"""
        # 重置图像数据
        self.original_image = None
        self.processed_image = None
        
        # 重置界面显示
        self.original_label.clear()
        self.original_label.setText("请打开图片")
        self.processed_label.clear()
        self.processed_label.setText("处理结果将显示在这里")
        self.result_text.clear()
        
        # 重置按钮状态
        self.process_btn.setEnabled(False)
        self.quality_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        # 强制垃圾回收，释放内存
        import gc
        gc.collect()
        
        print("所有图像数据和缓存已完全重置")
    
    def _reset_image_state(self):
        """重置图像状态（用于图片更换，保留当前图片）"""
        # 重置处理结果
        self.processed_image = None
        self.processed_label.clear()
        self.processed_label.setText("处理结果将显示在这里")
        self.result_text.clear()
        
        # 重置按钮状态
        self.save_btn.setEnabled(False)
        
        # 强制垃圾回收，释放内存
        import gc
        gc.collect()
        
        print("图像状态已重置")
    
    def _reset_all_sliders(self):
        """重置所有美颜滑块到默认值"""
        for slider in self.beauty_sliders.values():
            slider.setValue(50)  # 默认50%
        
        # 更新滑块标签
        for label in self.beauty_labels.values():
            label.setText("强度: 50%")
        
        print("所有美颜滑块已重置到默认值")
    
    def process_image(self):
        """执行美颜处理"""
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        try:
            # 复制原图进行处理
            self.processed_image = self.original_image.copy()
            
            # 获取选中的美颜选项
            selected_beauties = [key for key, checkbox in self.beauty_checkboxes.items() 
                               if checkbox.isChecked()]
            
            if not selected_beauties:
                QMessageBox.warning(self, "警告", "请至少选择一种美颜效果")
                return
            
            # 按顺序应用选中的美颜效果（带强度参数）
            for beauty_key in selected_beauties:
                if beauty_key in beauty_plugins:
                    beauty_plugin = beauty_plugins[beauty_key]()
                    
                    # 获取对应的强度值（滑块值转换为0.0-1.0范围）
                    intensity_slider = self.beauty_sliders[beauty_key]
                    intensity_value = intensity_slider.value() / 100.0  # 转换为0.0-1.0
                    
                    # 设置强度并处理图片
                    beauty_plugin.set_intensity(intensity_value)
                    self.processed_image = beauty_plugin.process(self.processed_image)
            
            # 显示处理结果
            pixmap = self.image_loader.cv2_to_qpixmap(self.processed_image)
            self.processed_label.setPixmap(pixmap.scaled(
                500, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # 启用保存按钮
            self.save_btn.setEnabled(True)
            
            QMessageBox.information(self, "成功", "美颜处理完成")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理图片时发生错误: {str(e)}")
    
    def evaluate_quality(self):
        """执行质量评价"""
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        try:
            # 选择要评价的图片（优先使用处理后的图片）
            image_to_evaluate = self.processed_image if self.processed_image is not None else self.original_image
            
            result_text = "=== 图像质量评价结果 ===\n\n"
            
            # 先执行人脸检测，获取结果
            face_detection_result = None
            face_detection_plugin = None
            
            # 执行所有质量评价插件
            for quality_key, quality_class in quality_plugins.items():
                try:
                    quality_plugin = quality_class()
                    
                    # 如果是人脸检测模块，保存结果
                    if quality_key == "face_detection":
                        face_detection_result = quality_plugin.evaluate(image_to_evaluate)
                        face_detection_plugin = quality_plugin
                        
                        # 检查是否为大指标（人脸检测）
                        if face_detection_result.get('is_major_metric', False):
                            # 大指标：分层显示
                            result_text += f"【{quality_plugin.name}】\n"
                            result_text += f"得分: {face_detection_result['score']:.2f}\n"
                            result_text += f"评价:\n{face_detection_result['assessment']}\n\n"
                        else:
                            # 普通指标：原有格式
                            result_text += f"【{quality_plugin.name}】\n"
                            result_text += f"得分: {face_detection_result['score']:.2f}\n"
                            result_text += f"评价: {face_detection_result['assessment']}\n\n"
                    
                    # 如果是人脸状态模块，传递人脸检测结果
                    elif quality_key == "face_state":
                        if face_detection_result is not None:
                            result = quality_plugin.evaluate(image_to_evaluate, face_detection_result)
                        else:
                            result = quality_plugin.evaluate(image_to_evaluate)
                        
                        # 检查是否为大指标（人脸状态）
                        if result.get('is_major_metric', False):
                            # 大指标：分层显示
                            result_text += f"【{quality_plugin.name}】\n"
                            result_text += f"得分: {result['score']:.2f}\n"
                            result_text += f"评价:\n{result['assessment']}\n\n"
                        else:
                            # 普通指标：原有格式
                            result_text += f"【{quality_plugin.name}】\n"
                            result_text += f"得分: {result['score']:.2f}\n"
                            result_text += f"评价: {result['assessment']}\n\n"
                    
                    # 其他模块正常执行
                    else:
                        result = quality_plugin.evaluate(image_to_evaluate)
                        
                        # 检查是否为大指标
                        if result.get('is_major_metric', False):
                            # 大指标：分层显示
                            result_text += f"【{quality_plugin.name}】\n"
                            result_text += f"得分: {result['score']:.2f}\n"
                            result_text += f"评价:\n{result['assessment']}\n\n"
                        else:
                            # 普通指标：原有格式
                            result_text += f"【{quality_plugin.name}】\n"
                            result_text += f"得分: {result['score']:.2f}\n"
                            result_text += f"评价: {result['assessment']}\n\n"
                        
                except Exception as e:
                    result_text += f"【{quality_key}】评价失败: {str(e)}\n\n"
            
            self.result_text.setText(result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"质量评价时发生错误: {str(e)}")
    
    def save_image(self):
        """保存处理后的图片"""
        if self.processed_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的处理结果")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图片", 
                "processed_image.jpg", 
                "JPEG文件 (*.jpg);;PNG文件 (*.png);;所有文件 (*.*)"
            )
            
            if file_path:
                success = self.image_loader.save_image(self.processed_image, file_path)
                if success:
                    QMessageBox.information(self, "成功", f"图片保存成功: {file_path}")
                else:
                    QMessageBox.warning(self, "错误", "图片保存失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存图片时发生错误: {str(e)}")
    
    def reset_slider(self, slider, label):
        """重置单个滑块到默认值"""
        slider.setValue(50)
        label.setText("强度: 50%")
    
    def reset_all(self):
        """重置所有状态"""
        self.original_image = None
        self.processed_image = None
        
        self.original_label.setText("请打开图片")
        self.processed_label.setText("处理结果将显示在这里")
        
        # 重置所有复选框和滑块
        for checkbox in self.beauty_checkboxes.values():
            checkbox.setChecked(False)
        
        for slider in self.beauty_sliders.values():
            slider.setValue(50)
        
        self.process_btn.setEnabled(False)
        self.quality_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        self.result_text.clear()
        
        QMessageBox.information(self, "提示", "系统已重置")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()