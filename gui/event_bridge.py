"""GUIイベントブリッジ - GUIアクションをYANAに通知"""

from pathlib import Path
from yana.session import SessionManager, Event, WorkPhase


class GUIEventBridge:
    """GUIアクションをセッションに通知"""
    
    def __init__(self, session_manager: SessionManager):
        self.session = session_manager
    
    def on_capture_started(self):
        """撮影開始"""
        self.session.state.phase = WorkPhase.DATA_COLLECTION
        return self.session.record_event("gui", "capture_started")
    
    def on_capture_stopped(self, frame_count: int):
        """撮影停止"""
        return self.session.record_event("gui", "capture_stopped", {
            "frame_count": frame_count
        })
    
    def on_frame_captured(self, frame_path: str, frame_number: int):
        """フレーム撮影"""
        self.session.state.total_frames = frame_number
        return self.session.record_event("gui", "frame_captured", {
            "path": frame_path,
            "number": frame_number
        })
    
    def on_image_selected(self, image_path: str):
        """画像選択"""
        return self.session.record_event("gui", "image_selected", {
            "path": image_path
        })
    
    def on_image_deleted(self, image_path: str, reason: str = ""):
        """画像削除"""
        return self.session.record_event("gui", "image_deleted", {
            "path": image_path,
            "reason": reason
        })
    
    def on_annotation_started(self):
        """アノテーション開始"""
        self.session.state.phase = WorkPhase.ANNOTATION
        return self.session.record_event("gui", "annotation_started")
    
    def on_annotation_updated(self, image_path: str, annotation: dict):
        """アノテーション更新"""
        self.session.state.annotated_count += 1
        return self.session.record_event("gui", "annotation_updated", {
            "path": image_path,
            "annotation": annotation
        })
    
    def on_road_mapping_changed(self, mapping: dict):
        """ROADマッピング変更"""
        self.session.state.road_mapping = mapping
        return self.session.record_event("gui", "road_mapping_changed", {
            "mapping": mapping,
            "class_count": len(mapping)
        })
    
    def on_session_directory_changed(self, directory: str):
        """セッションディレクトリ変更"""
        self.session.state.collection_dir = directory
        return self.session.record_event("gui", "directory_changed", {
            "directory": directory
        })
    
    def on_training_started(self, config: dict = None):
        """訓練開始"""
        self.session.state.phase = WorkPhase.TRAINING
        self.session.state.training_epoch = 0
        return self.session.record_event("gui", "training_started", {
            "config": config or {}
        })
    
    def on_training_progress(self, epoch: int, loss: float):
        """訓練進捗"""
        self.session.state.training_epoch = epoch
        self.session.state.training_loss = loss
        return self.session.record_event("gui", "training_progress", {
            "epoch": epoch,
            "loss": loss
        })
    
    def on_training_completed(self, metrics: dict = None):
        """訓練完了"""
        self.session.state.phase = WorkPhase.EVALUATION
        return self.session.record_event("gui", "training_completed", {
            "metrics": metrics or {}
        })
    
    def on_export_started(self, output_dir: str):
        """エクスポート開始"""
        return self.session.record_event("gui", "export_started", {
            "output_dir": output_dir
        })
    
    def on_export_completed(self, output_dir: str, file_count: int):
        """エクスポート完了"""
        return self.session.record_event("gui", "export_completed", {
            "output_dir": output_dir,
            "file_count": file_count
        })
