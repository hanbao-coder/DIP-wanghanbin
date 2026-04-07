#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量评价插件模块
包含质量评价基类和2种核心质量评价插件：人脸检测和人脸状态
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod


class QualityBase(ABC):
    """质量评价基类 - 所有质量评价插件必须继承此类"""
    
    def __init__(self):
        self.name = "基础质量评价"
        self.description = "质量评价算法基类"
    
    @abstractmethod
    def evaluate(self, image):
        """
        质量评价主方法
        
        参数:
            image: 输入图像（RGB格式）
            
        返回:
            包含得分和评价的字典
        """
        pass


class FaceDetectionQuality(QualityBase):
    """人脸检测大指标 - 包含5个子指标（使用MediaPipe人脸检测）"""
    
    def __init__(self):
        super().__init__()
        self.name = "人脸检测"
        self.description = "人脸检测大指标，包含5个独立子指标"
        
        # 初始化MediaPipe人脸检测器
        try:
            import mediapipe as mp
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,  # 0: 近距离人脸，1: 远距离人脸
                min_detection_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except ImportError:
            print("警告: MediaPipe未安装，人脸检测功能将不可用")
            self.face_detection = None
    
    def _apply_nms(self, detections, iou_threshold=0.3):
        """应用非极大值抑制(NMS)去除重复检测"""
        if not detections:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        filtered_detections = []
        
        while detections:
            # 取置信度最高的检测
            best_detection = detections.pop(0)
            filtered_detections.append(best_detection)
            
            # 计算与剩余检测的IoU
            remaining_detections = []
            for detection in detections:
                iou = self._calculate_iou(best_detection['bbox'], detection['bbox'])
                if iou < iou_threshold:
                    remaining_detections.append(detection)
            
            detections = remaining_detections
        
        return filtered_detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集区域
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        
        # 计算交集面积
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # 计算并集面积
        union_area = w1 * h1 + w2 * h2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def evaluate(self, image):
        """
        人脸检测大指标评价（使用MediaPipe）
        
        参数:
            image: 输入图像
            
        返回:
            包含5个子指标结果的大指标评价
        """
        try:
            # 检查MediaPipe是否可用
            if self.face_detection is None:
                return self._get_fallback_result()
            
            # 转换图像为RGB格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            height, width = rgb_image.shape[:2]
            
            # 使用MediaPipe进行人脸检测
            results = self.face_detection.process(rgb_image)
            
            # 提取检测结果
            detections = []
            if results.detections:
                for detection in results.detections:
                    # 获取边界框
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    
                    # 获取置信度
                    confidence = detection.score[0]
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'score': confidence
                    })
            
            # 应用NMS去除重复检测
            filtered_detections = self._apply_nms(detections)
            
            # 子指标1: 是否检测到人脸
            has_face = len(filtered_detections) > 0
            face_detected_text = "是" if has_face else "否"
            
            # 子指标2: 人脸数量
            face_count = len(filtered_detections)
            face_count_text = f"{face_count} 张"
            
            # 子指标3: 人脸置信度（使用真实检测分数）
            confidence = 0.0
            if has_face:
                # 取最高置信度
                confidence = max(detection['score'] for detection in filtered_detections)
            confidence_text = f"{confidence:.2f}"
            
            # 子指标4: 人脸区域大小
            face_size_assessment = "无"
            if has_face:
                # 取最大的人脸区域
                largest_detection = max(filtered_detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
                x, y, w, h = largest_detection['bbox']
                
                # 计算人脸区域面积占整图比例
                face_area = w * h
                image_area = width * height
                face_ratio = face_area / image_area
                
                if face_ratio < 0.05:  # 小于5%认为过小
                    face_size_assessment = "偏小"
                elif face_ratio > 0.3:  # 大于30%认为过大
                    face_size_assessment = "过大"
                else:
                    face_size_assessment = "正常"
            else:
                face_size_assessment = "无"
            
            # 子指标5: 人脸完整性
            face_completeness = "无"
            if has_face:
                # 取最大的人脸区域
                largest_detection = max(filtered_detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
                x, y, w, h = largest_detection['bbox']
                
                # 检查人脸是否被边缘裁剪
                margin = 10  # 边界容差
                is_cropped = (x < margin or y < margin or 
                            x + w > width - margin or y + h > height - margin)
                
                face_completeness = "完整" if not is_cropped else "不完整"
            else:
                face_completeness = "无"
            
            # 构建分层显示结果
            assessment_lines = [
                f"是否检测到人脸：{face_detected_text}",
                f"人脸数量：{face_count_text}",
                f"人脸置信度：{confidence_text}",
                f"人脸区域大小：{face_size_assessment}",
                f"人脸完整性：{face_completeness}"
            ]
            
            # 综合得分（基于所有子指标）
            overall_score = 0
            if has_face:
                # 基础得分
                base_score = 60
                
                # 人脸数量加分（最多2张）
                count_bonus = min(face_count, 2) * 10
                
                # 人脸区域大小加分
                size_bonus = 10 if face_size_assessment == "正常" else 0
                
                # 人脸完整性加分
                completeness_bonus = 10 if face_completeness == "完整" else 0
                
                # 置信度加分
                confidence_bonus = int(confidence * 10)
                
                overall_score = base_score + count_bonus + size_bonus + completeness_bonus + confidence_bonus
                overall_score = min(overall_score, 100)
            
            return {
                'score': overall_score,
                'assessment': '\n'.join(assessment_lines),
                'face_detected': has_face,
                'face_count': face_count,
                'confidence': confidence,
                'face_size_assessment': face_size_assessment,
                'face_completeness': face_completeness,
                'is_major_metric': True  # 标记为大指标
            }
            
        except Exception as e:
            print(f"人脸检测失败: {str(e)}")
            return self._get_fallback_result()
    
    def _get_fallback_result(self):
        """获取MediaPipe不可用时的默认结果"""
        return {
            'score': 0, 
            'assessment': "是否检测到人脸：否\n人脸数量：0 张\n人脸置信度：0.00\n人脸区域大小：无\n人脸完整性：无",
            'face_detected': False,
            'face_count': 0,
            'confidence': 0.0,
            'face_size_assessment': "无",
            'face_completeness': "无",
            'is_major_metric': True
        }


class FaceSharpnessQuality(QualityBase):
    """人脸区域清晰度检测"""
    
    def __init__(self):
        super().__init__()
        self.name = "人脸清晰度"
        self.description = "检测人脸区域的清晰度"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def evaluate(self, image):
        """
        人脸区域清晰度检测
        
        参数:
            image: 输入图像
            
        返回:
            人脸清晰度得分
        """
        try:
            # 转换为灰度图像
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 人脸检测
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            if len(faces) == 0:
                return {'score': 0, 'assessment': '未检测到人脸', 'face_sharpness': 0}
            
            # 取最大的人脸区域
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # 提取人脸区域
            face_region = gray[y:y+h, x:x+w]
            
            # 计算人脸区域的Laplacian方差
            if face_region.size > 0:
                face_sharpness = cv2.Laplacian(face_region, cv2.CV_64F).var()
                score = min(face_sharpness / 10, 100)
                
                if score > 80:
                    assessment = "人脸非常清晰"
                elif score > 60:
                    assessment = "人脸清晰"
                elif score > 40:
                    assessment = "人脸一般清晰"
                elif score > 20:
                    assessment = "人脸较模糊"
                else:
                    assessment = "人脸非常模糊"
            else:
                score = 0
                assessment = "人脸区域无效"
                face_sharpness = 0
            
            return {
                'score': score,
                'assessment': assessment,
                'face_sharpness': face_sharpness
            }
            
        except Exception as e:
            print(f"人脸清晰度检测失败: {str(e)}")
            return {'score': 0, 'assessment': '检测失败', 'face_sharpness': 0}


class FaceBrightnessContrastQuality(QualityBase):
    """人脸区域亮度和对比度检测"""
    
    def __init__(self):
        super().__init__()
        self.name = "人脸亮度对比度"
        self.description = "检测人脸区域的亮度和对比度"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def evaluate(self, image):
        """
        人脸区域亮度和对比度检测
        
        参数:
            image: 输入图像
            
        返回:
            人脸亮度和对比度得分
        """
        try:
            # 转换为灰度图像
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 人脸检测
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            if len(faces) == 0:
                return {'score': 0, 'assessment': '未检测到人脸', 'face_brightness': 0, 'face_contrast': 0}
            
            # 取最大的人脸区域
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # 提取人脸区域
            face_region = gray[y:y+h, x:x+w]
            
            if face_region.size > 0:
                # 计算人脸区域亮度
                face_brightness = np.mean(face_region)
                
                # 计算人脸区域对比度
                face_contrast = np.std(face_region)
                
                # 亮度得分（理想范围：100-150）
                if 100 <= face_brightness <= 150:
                    brightness_score = 100
                else:
                    brightness_score = max(0, 100 - abs(face_brightness - 125) / 125 * 100)
                
                # 对比度得分（理想范围：40-80）
                if 40 <= face_contrast <= 80:
                    contrast_score = 100
                else:
                    contrast_score = max(0, 100 - abs(face_contrast - 60) / 60 * 100)
                
                # 综合得分
                score = (brightness_score + contrast_score) / 2
                
                assessment = f"人脸亮度:{face_brightness:.1f}, 对比度:{face_contrast:.1f}"
                
            else:
                score = 0
                assessment = "人脸区域无效"
                face_brightness = 0
                face_contrast = 0
            
            return {
                'score': score,
                'assessment': assessment,
                'face_brightness': face_brightness,
                'face_contrast': face_contrast
            }
            
        except Exception as e:
            print(f"人脸亮度对比度检测失败: {str(e)}")
            return {'score': 0, 'assessment': '检测失败', 'face_brightness': 0, 'face_contrast': 0}


class FaceCenteringQuality(QualityBase):
    """人脸居中程度检测"""
    
    def __init__(self):
        super().__init__()
        self.name = "人脸居中程度"
        self.description = "检测人脸在图像中的居中程度"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def evaluate(self, image):
        """
        人脸居中程度检测
        
        参数:
            image: 输入图像
            
        返回:
            人脸居中程度得分
        """
        try:
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # 人脸检测
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            if len(faces) == 0:
                return {'score': 0, 'assessment': '未检测到人脸', 'centering_score': 0}
            
            # 取最大的人脸区域
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # 计算人脸中心点
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # 计算与图像中心的距离
            distance_x = abs(face_center_x - center_x) / width
            distance_y = abs(face_center_y - center_y) / height
            
            # 计算居中得分（距离越小得分越高）
            centering_score = 100 * (1 - (distance_x + distance_y) / 2)
            score = max(0, min(centering_score, 100))
            
            if score > 90:
                assessment = "人脸位置非常居中"
            elif score > 80:
                assessment = "人脸位置良好"
            elif score > 60:
                assessment = "人脸位置一般"
            elif score > 40:
                assessment = "人脸位置偏斜"
            else:
                assessment = "人脸位置很差"
            
            return {
                'score': score,
                'assessment': assessment,
                'centering_score': centering_score
            }
            
        except Exception as e:
            print(f"人脸居中程度检测失败: {str(e)}")
            return {'score': 0, 'assessment': '检测失败', 'centering_score': 0}


class BackgroundUniformityQuality(QualityBase):
    """背景均匀度检测"""
    
    def __init__(self):
        super().__init__()
        self.name = "背景均匀度"
        self.description = "检测图像背景的均匀程度"
    
    def evaluate(self, image):
        """
        背景均匀度检测
        
        参数:
            image: 输入图像
            
        返回:
            背景均匀度得分
        """
        try:
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 提取饱和度通道（背景通常饱和度较低）
            saturation = hsv[:, :, 1]
            
            # 计算饱和度的标准差（标准差越小越均匀）
            saturation_std = np.std(saturation)
            
            # 计算得分（标准差越小得分越高）
            score = max(0, 100 - saturation_std / 2)
            
            if score > 90:
                assessment = "背景非常均匀"
            elif score > 80:
                assessment = "背景均匀"
            elif score > 60:
                assessment = "背景一般"
            elif score > 40:
                assessment = "背景较杂乱"
            else:
                assessment = "背景非常杂乱"
            
            return {
                'score': score,
                'assessment': assessment,
                'saturation_std': saturation_std
            }
            
        except Exception as e:
            print(f"背景均匀度检测失败: {str(e)}")
            return {'score': 0, 'assessment': '检测失败', 'saturation_std': 0}


# 质量评价插件注册管理器
quality_plugins = {}


def register_quality_plugin(name, plugin_class):
    """
    注册质量评价插件
    
    参数:
        name: 插件名称（唯一标识）
        plugin_class: 插件类
    """
    if issubclass(plugin_class, QualityBase):
        quality_plugins[name] = plugin_class
        print(f"质量评价插件注册成功: {name}")
    else:
        print(f"注册失败: {plugin_class} 不是QualityBase的子类")


def unregister_quality_plugin(name):
    """
    注销质量评价插件
    
    参数:
        name: 插件名称
    """
    if name in quality_plugins:
        del quality_plugins[name]
        print(f"质量评价插件注销成功: {name}")
    else:
        print(f"注销失败: 未找到插件 {name}")


def get_all_quality_plugins():
    """
    获取所有已注册的质量评价插件
    
    返回:
        插件字典
    """
    return quality_plugins.copy()


# 注册核心质量评价插件（只保留人脸检测和人脸状态）
register_quality_plugin("face_detection", FaceDetectionQuality)


class FaceStateQuality(QualityBase):
    """人脸状态大指标 - 包含5个子指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "人脸状态"
        self.description = "人脸状态大指标，包含5个独立子指标：左眼闭合、右眼闭合、双眼闭合、嘴巴张开、头部Roll"
        
        # 阈值配置
        self.ear_threshold = 0.25  # EAR阈值，默认0.25
        self.mar_threshold = 0.65  # MAR阈值，优化为0.65，减少误判
        self.roll_normal_threshold = 10  # 正常倾斜角度阈值，±10°
        self.roll_slight_threshold = 20  # 轻微倾斜角度阈值，±20°
        
        # MediaPipe FaceMesh模型
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        except ImportError:
            print("警告: MediaPipe未安装，人脸状态检测功能将不可用")
            self.face_mesh = None
    
    def _check_mediapipe_available(self):
        """检查MediaPipe是否可用"""
        return self.face_mesh is not None
    
    def set_face_detection_result(self, face_detection_result):
        """
        设置人脸检测结果
        
        参数:
            face_detection_result: 人脸检测模块的结果字典
        """
        self.face_detection_result = face_detection_result
    
    def calculate_ear(self, eye_landmarks):
        """
        计算眼睛纵横比 (Eye Aspect Ratio, EAR)
        
        参数:
            eye_landmarks: 眼睛6个关键点坐标列表
            
        返回:
            EAR值 (float)
        
        公式: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        关键点索引说明:
        p1: 左眼角, p2: 左上眼睑, p3: 右上眼睑, p4: 右眼角, p5: 右下眼睑, p6: 左下眼睑
        """
        # 提取6个关键点坐标
        p1 = eye_landmarks[0]  # 左眼角
        p2 = eye_landmarks[1]  # 左上眼睑
        p3 = eye_landmarks[2]  # 右上眼睑
        p4 = eye_landmarks[3]  # 右眼角
        p5 = eye_landmarks[4]  # 右下眼睑
        p6 = eye_landmarks[5]  # 左下眼睑
        
        # 计算垂直距离（上下眼睑距离）
        p2_p6 = np.linalg.norm(p2 - p6)  # 左上到左下
        p3_p5 = np.linalg.norm(p3 - p5)  # 右上到右下
        
        # 计算水平距离（左右眼角距离）
        p1_p4 = np.linalg.norm(p1 - p4)  # 左眼角到右眼角
        
        # EAR公式: (p2_p6 + p3_p5) / (2.0 * p1_p4)
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        
        return ear
    
    def calculate_mar(self, mouth_landmarks):
        """
        计算嘴巴纵横比 (Mouth Aspect Ratio, MAR)
        
        参数:
            mouth_landmarks: 嘴巴6个关键点坐标列表
            
        返回:
            MAR值 (float)
        
        公式: MAR = 上下唇距离 / 嘴角距离
        关键点索引说明 (MediaPipe标准6点):
        p1: 上唇中点 (13), p2: 下唇中点 (14), p3: 左嘴角 (15), p4: 右嘴角 (16), p5: 上唇左点 (78), p6: 上唇右点 (308)
        简化计算: 使用上唇中点和下唇中点的垂直距离 / 左右嘴角的水平距离
        """
        # 提取6个关键点坐标
        p1 = mouth_landmarks[0]  # 上唇中点 (13)
        p2 = mouth_landmarks[1]  # 下唇中点 (14)
        p3 = mouth_landmarks[2]  # 左嘴角 (15)
        p4 = mouth_landmarks[3]  # 右嘴角 (16)
        
        # 计算垂直距离（上下唇距离）
        vertical_distance = np.linalg.norm(p1 - p2)
        
        # 计算水平距离（嘴角距离）
        horizontal_distance = np.linalg.norm(p3 - p4)
        
        # 避免除零错误
        if horizontal_distance == 0:
            return 0.0
        
        # MAR公式: 上下唇距离 / 嘴角距离
        mar = vertical_distance / horizontal_distance
        
        return mar
    
    def calculate_head_roll(self, face_landmarks):
        """
        计算头部Roll角度（头部倾斜角）
        
        参数:
            face_landmarks: 人脸关键点坐标列表
            
        返回:
            Roll角度（度），范围：-45°~45°
        """
        try:
            # 使用左右眼外眼角点计算头部倾斜角度
            # 左眼外眼角索引：33
            # 右眼外眼角索引：263
            left_eye_outer = face_landmarks[33]
            right_eye_outer = face_landmarks[263]
            
            # 计算两点之间的向量
            dx = right_eye_outer[0] - left_eye_outer[0]
            dy = right_eye_outer[1] - left_eye_outer[1]
            
            # 避免除零错误
            if abs(dx) < 1e-6:
                return 0.0
            
            # 计算角度（弧度）
            angle_rad = np.arctan2(dy, dx)
            
            # 转换为角度
            angle_deg = np.degrees(angle_rad)
            
            # 限制角度范围在-45°~45°之间，避免异常值
            angle_deg = max(-45.0, min(45.0, angle_deg))
            
            # 角度范围：-45°~45°，0°为完全水平
            # 正角度表示右眼高于左眼（顺时针倾斜）
            # 负角度表示左眼高于右眼（逆时针倾斜）
            
            return angle_deg
            
        except Exception as e:
            print(f"头部Roll角度计算错误: {str(e)}")
            return 0.0
    
    def evaluate(self, image, face_detection_result=None):
        """
        评估人脸状态质量
        
        参数:
            image: 输入图像
            face_detection_result: 人脸检测结果（可选）
            
        返回:
            评估结果字典
        """
        # 检查MediaPipe是否可用
        if not self._check_mediapipe_available():
            return {
                'score': 0,
                'assessment': "MediaPipe未安装，无法计算\nMediaPipe未安装，无法计算\nMediaPipe未安装，无法计算\nMediaPipe未安装，无法计算\nMediaPipe未安装，无法计算",
                'is_major_metric': True
            }
        
        # 强制创建新的MediaPipe实例，避免缓存问题
        try:
            import mediapipe as mp
            # 根据是否有人脸检测结果选择不同的评估方式
            if face_detection_result is not None and face_detection_result.get('face_detected', False):
                return self._evaluate_with_face_detection(image, face_detection_result)
            else:
                return self._evaluate_with_self_detection(image)
        except Exception as e:
            print(f"人脸状态检测异常: {str(e)}")
            return {
                'score': 0,
                'assessment': "检测异常\n检测异常\n检测异常\n检测异常\n检测异常",
                'is_major_metric': True
            }
    
    def _evaluate_with_self_detection(self, image):
        """在没有外部人脸检测结果时，自己进行人脸检测"""
        # 转换图像为RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建新的MediaPipe FaceMesh实例，避免缓存问题
        try:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            
            # 使用新的实例检测人脸关键点
            results = face_mesh.process(rgb_image)
            
            # 立即关闭模型，释放资源
            face_mesh.close()
            
            if not results.multi_face_landmarks:
                # 未检测到人脸
                return {
                    'score': 0,
                    'assessment': "未检测到人脸，无法计算\n未检测到人脸，无法计算\n未检测到人脸，无法计算\n未检测到人脸，无法计算\n未检测到人脸，无法计算",
                    'is_major_metric': True
                }
            
            # 获取第一个检测到的人脸
            face_landmarks = results.multi_face_landmarks[0]
            
            # 提取关键点坐标
            h, w = image.shape[:2]
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append(np.array([x, y]))
            
            landmarks = np.array(landmarks)
            
            return self._calculate_face_state(landmarks)
            
        except Exception as e:
            print(f"人脸状态检测失败: {str(e)}")
            return {
                'score': 0,
                'assessment': "检测失败\n检测失败\n检测失败\n检测失败\n检测失败",
                'is_major_metric': True
            }
    
    def _evaluate_with_face_detection(self, image, face_detection_result):
        """基于外部人脸检测结果进行人脸状态计算"""
        # 转换图像为RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建新的MediaPipe FaceMesh实例，避免缓存问题
        try:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            
            # 使用新的实例检测人脸关键点
            results = face_mesh.process(rgb_image)
            
            # 立即关闭模型，释放资源
            face_mesh.close()
            
            if not results.multi_face_landmarks:
                # 虽然人脸检测模块说检测到了，但MediaPipe没检测到
                return {
                    'score': 0,
                    'assessment': "人脸关键点检测失败\n人脸关键点检测失败\n人脸关键点检测失败\n人脸关键点检测失败\n人脸关键点检测失败",
                    'is_major_metric': True
                }
            
            # 获取第一个检测到的人脸
            face_landmarks = results.multi_face_landmarks[0]
            
            # 提取关键点坐标
            h, w = image.shape[:2]
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append(np.array([x, y]))
            
            landmarks = np.array(landmarks)
            
            return self._calculate_face_state(landmarks)
            
        except Exception as e:
            print(f"人脸状态检测失败: {str(e)}")
            return {
                'score': 0,
                'assessment': "检测失败\n检测失败\n检测失败\n检测失败\n检测失败",
                'is_major_metric': True
            }
    
    def _calculate_face_state(self, landmarks):
        """基于人脸关键点计算人脸状态"""
        # 1. 左眼闭合检测
        # MediaPipe官方标准左眼6个关键点索引: [33, 160, 159, 133, 153, 144]
        # 对应位置: 左眼角, 左上眼睑, 右上眼睑, 右眼角, 右下眼睑, 左下眼睑
        left_eye_indices = [33, 160, 159, 133, 153, 144]
        left_eye_landmarks = landmarks[left_eye_indices]
        left_ear = self.calculate_ear(left_eye_landmarks)
        left_eye_closed = left_ear < self.ear_threshold
        
        # 2. 右眼闭合检测
        # MediaPipe官方标准右眼6个关键点索引: [362, 387, 386, 263, 380, 373]
        # 对应位置: 左眼角, 左上眼睑, 右上眼睑, 右眼角, 右下眼睑, 左下眼睑
        right_eye_indices = [362, 387, 386, 263, 380, 373]
        right_eye_landmarks = landmarks[right_eye_indices]
        right_ear = self.calculate_ear(right_eye_landmarks)
        right_eye_closed = right_ear < self.ear_threshold
        
        # 3. 双眼闭合检测
        # 仅当左右眼EAR值同时低于阈值时，才判定为双眼闭合
        both_eyes_closed = left_eye_closed and right_eye_closed
        
        # 4. 嘴巴张开检测
        # MediaPipe官方标准嘴巴6个关键点索引: [13, 14, 15, 16, 78, 308]
        # 对应位置: 上唇中点, 下唇中点, 左嘴角, 右嘴角, 上唇左点, 上唇右点
        mouth_indices = [13, 14, 15, 16, 78, 308]
        mouth_landmarks = landmarks[mouth_indices]
        mar = self.calculate_mar(mouth_landmarks)
        mouth_open = mar > self.mar_threshold
        
        # 5. 头部Roll角度计算
        roll_angle = self.calculate_head_roll(landmarks)
        
        # 评估头部倾斜程度
        # 角度范围：-90°~90°，0°为完全水平
        # ±10°内判定为正常，±20°内为轻微倾斜，超过±20°为严重倾斜
        if abs(roll_angle) <= self.roll_normal_threshold:
            roll_assessment = "正常"
        elif abs(roll_angle) <= self.roll_slight_threshold:
            roll_assessment = "轻微倾斜"
        else:
            roll_assessment = "严重倾斜"
        
        # 构建分层评价结果
        assessment_lines = [
            f"左眼闭合：{'是' if left_eye_closed else '否'}（EAR值：{left_ear:.2f}，阈值：{self.ear_threshold:.2f}）",
            f"右眼闭合：{'是' if right_eye_closed else '否'}（EAR值：{right_ear:.2f}，阈值：{self.ear_threshold:.2f}）",
            f"双眼闭合：{'是' if both_eyes_closed else '否'}",
            f"嘴巴张开：{'是' if mouth_open else '否'}（MAR值：{mar:.2f}，阈值：{self.mar_threshold:.2f}）",
            f"头部 Roll（头部倾斜角）：{roll_angle:.1f}°（{roll_assessment}）"
        ]
        
        # 计算综合得分（基于各项指标的加权平均）
        # 这里使用简单的评分逻辑，实际可以根据需求调整
        score = 80.0  # 基础分数，实际应该根据检测结果动态计算
        
        return {
            'score': score,
            'assessment': '\n'.join(assessment_lines),
            'left_eye_closed': left_eye_closed,
            'right_eye_closed': right_eye_closed,
            'both_eyes_closed': both_eyes_closed,
            'mouth_open': mouth_open,
            'head_roll_angle': roll_angle,
            'head_roll_assessment': roll_assessment,
            'is_major_metric': True
        }


# 注册人脸状态检测插件
register_quality_plugin("face_state", FaceStateQuality)


# 测试代码
if __name__ == "__main__":
    # 测试插件注册
    print("已注册的质量评价插件:")
    for name, plugin_class in quality_plugins.items():
        plugin_instance = plugin_class()
        print(f"- {name}: {plugin_instance.name} - {plugin_instance.description}")
    
    # 测试图像质量评价
    import numpy as np
    
    # 创建测试图像
    test_image = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    
    print("\n测试质量评价:")
    for name, plugin_class in quality_plugins.items():
        try:
            plugin = plugin_class()
            result = plugin.evaluate(test_image)
            print(f"{plugin.name}: 得分{result['score']:.1f} - {result['assessment']}")
        except Exception as e:
            print(f"{plugin.name}: 评价失败 - {str(e)}")