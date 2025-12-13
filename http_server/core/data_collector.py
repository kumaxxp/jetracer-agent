"""データ収集モジュール

走行中の画像と制御値を収集して保存する。
"""
import asyncio
import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading
import cv2
import numpy as np


@dataclass
class CollectionConfig:
    """収集設定"""
    fps: float = 1.0                    # 収集頻度
    base_dir: str = "data/sessions"     # 保存先ディレクトリ
    save_ground_camera: bool = False    # 足元カメラも保存
    save_segmentation: bool = False     # セグメンテーション結果も保存
    image_format: str = "jpg"           # 画像形式
    image_quality: int = 85             # JPEG品質


@dataclass
class CollectionSession:
    """収集セッション"""
    session_id: str
    started_at: float
    ended_at: Optional[float] = None
    frame_count: int = 0
    config: CollectionConfig = field(default_factory=CollectionConfig)
    
    # パス
    session_dir: Path = None
    frames_dir: Path = None
    log_path: Path = None
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "frame_count": self.frame_count,
            "duration_sec": (self.ended_at or time.time()) - self.started_at,
            "config": {
                "fps": self.config.fps,
                "save_ground_camera": self.config.save_ground_camera,
                "save_segmentation": self.config.save_segmentation,
            }
        }


class DataCollector:
    """データ収集エンジン"""
    
    def __init__(self, config: CollectionConfig = None):
        self.config = config or CollectionConfig()
        
        self._session: Optional[CollectionSession] = None
        self._collecting = False
        self._loop_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._lock = threading.Lock()
        
        # ログファイル
        self._log_file = None
        self._csv_writer = None
        
        # コンポーネント（遅延取得）
        self._camera_manager = None
        self._controller = None
    
    def _init_components(self):
        """コンポーネントを取得"""
        from .camera_manager import camera_manager
        from .autonomous_controller import autonomous_controller
        
        self._camera_manager = camera_manager
        self._controller = autonomous_controller
    
    def start_session(self, session_id: str = None) -> CollectionSession:
        """収集セッションを開始"""
        if self._collecting:
            print("[DataCollector] Already collecting")
            return self._session
        
        self._init_components()
        
        # セッションID生成
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ディレクトリ作成
        base_dir = Path(self.config.base_dir)
        session_dir = base_dir / session_id
        frames_dir = session_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.save_ground_camera:
            (session_dir / "ground_frames").mkdir(exist_ok=True)
        
        if self.config.save_segmentation:
            (session_dir / "segmentation").mkdir(exist_ok=True)
        
        # セッション作成
        self._session = CollectionSession(
            session_id=session_id,
            started_at=time.time(),
            config=self.config,
            session_dir=session_dir,
            frames_dir=frames_dir,
            log_path=session_dir / "log.csv"
        )
        
        # メタデータ保存
        self._save_metadata()
        
        # ログファイル初期化
        self._init_log_file()
        
        print(f"[DataCollector] Session started: {session_id}")
        return self._session
    
    async def start_collecting(self) -> bool:
        """収集ループを開始"""
        if self._session is None:
            print("[DataCollector] No session started")
            return False
        
        if self._collecting:
            print("[DataCollector] Already collecting")
            return False
        
        self._collecting = True
        self._stop_event.clear()
        self._loop_task = asyncio.create_task(self._collection_loop())
        
        print(f"[DataCollector] Collection started at {self.config.fps} fps")
        return True
    
    async def stop_collecting(self):
        """収集ループを停止"""
        if not self._collecting:
            return
        
        self._stop_event.set()
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        
        self._collecting = False
        print("[DataCollector] Collection stopped")
    
    def end_session(self) -> Optional[CollectionSession]:
        """セッションを終了"""
        if self._session is None:
            return None
        
        # 収集中なら停止
        if self._collecting:
            asyncio.create_task(self.stop_collecting())
        
        # ログファイルクローズ
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._csv_writer = None
        
        # 終了時刻記録
        self._session.ended_at = time.time()
        
        # メタデータ更新
        self._save_metadata()
        
        session = self._session
        self._session = None
        
        print(f"[DataCollector] Session ended: {session.session_id} ({session.frame_count} frames)")
        return session
    
    async def _collection_loop(self):
        """収集ループ"""
        interval = 1.0 / self.config.fps
        
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            try:
                self._collect_frame()
            except Exception as e:
                print(f"[DataCollector] Collection error: {e}")
                import traceback
                traceback.print_exc()
            
            # インターバル待機
            elapsed = time.time() - loop_start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)
    
    def _collect_frame(self):
        """1フレーム収集"""
        if not self._session or not self._camera_manager:
            return
        
        timestamp = time.time()
        frame_id = self._session.frame_count + 1
        frame_name = f"{frame_id:06d}"
        
        # 正面カメラ画像取得
        frame = self._camera_manager.read(0)
        if frame is None:
            print(f"[DataCollector] Failed to capture frame {frame_id}")
            return
        
        # 画像保存
        frame_path = self._session.frames_dir / f"{frame_name}.{self.config.image_format}"
        if self.config.image_format == "jpg":
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality])
        else:
            cv2.imwrite(str(frame_path), frame)
        
        # 足元カメラ（オプション）
        if self.config.save_ground_camera:
            ground_frame = self._camera_manager.read(1)
            if ground_frame is not None:
                ground_path = self._session.session_dir / "ground_frames" / f"{frame_name}.{self.config.image_format}"
                cv2.imwrite(str(ground_path), ground_frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality])
        
        # 制御値取得
        state = self._controller.get_state() if self._controller else {}
        control = state.get("control", {})
        sensors = state.get("sensors", {})
        imu = sensors.get("imu", {})
        
        # ログ記録
        record = {
            "timestamp": timestamp,
            "frame_id": frame_name,
            "steering": control.get("steering", 0.0),
            "throttle": control.get("throttle", 0.0),
            "mode": state.get("mode", "unknown"),
            "road_ratio": sensors.get("road_ratio", 0.0),
            "lidar_min_mm": sensors.get("lidar_min_mm", 9999),
            "heading": imu.get("heading", 0.0),
            "roll": imu.get("roll", 0.0),
            "pitch": imu.get("pitch", 0.0),
        }
        self._write_log(record)
        
        self._session.frame_count = frame_id
    
    def _init_log_file(self):
        """ログファイル初期化"""
        if self._session is None:
            return
        
        self._log_file = open(self._session.log_path, 'w', newline='')
        self._csv_writer = csv.writer(self._log_file)
        
        # ヘッダー
        headers = [
            "timestamp", "frame_id", "steering", "throttle", "mode",
            "road_ratio", "lidar_min_mm", "heading", "roll", "pitch"
        ]
        self._csv_writer.writerow(headers)
    
    def _write_log(self, record: dict):
        """ログ書き込み"""
        if self._csv_writer is None:
            return
        
        row = [
            record["timestamp"],
            record["frame_id"],
            record["steering"],
            record["throttle"],
            record["mode"],
            record["road_ratio"],
            record["lidar_min_mm"],
            record["heading"],
            record["roll"],
            record["pitch"],
        ]
        self._csv_writer.writerow(row)
        self._log_file.flush()
    
    def _save_metadata(self):
        """メタデータ保存"""
        if self._session is None:
            return
        
        metadata = {
            "session_id": self._session.session_id,
            "started_at": datetime.fromtimestamp(self._session.started_at).isoformat(),
            "ended_at": datetime.fromtimestamp(self._session.ended_at).isoformat() if self._session.ended_at else None,
            "frame_count": self._session.frame_count,
            "config": {
                "fps": self.config.fps,
                "save_ground_camera": self.config.save_ground_camera,
                "save_segmentation": self.config.save_segmentation,
                "image_format": self.config.image_format,
                "image_quality": self.config.image_quality,
            }
        }
        
        metadata_path = self._session.session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def get_session_info(self) -> Optional[dict]:
        """現在のセッション情報を取得"""
        if self._session is None:
            return None
        return self._session.to_dict()
    
    def is_collecting(self) -> bool:
        """収集中か確認"""
        return self._collecting
    
    def list_sessions(self) -> List[dict]:
        """セッション一覧を取得"""
        sessions = []
        base_dir = Path(self.config.base_dir)
        
        if not base_dir.exists():
            return sessions
        
        for session_dir in sorted(base_dir.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue
            
            metadata_path = session_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        sessions.append(metadata)
                except Exception:
                    pass
        
        return sessions
    
    def update_config(self, **kwargs):
        """設定を更新"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_config(self) -> dict:
        """現在の設定を取得"""
        return {
            "fps": self.config.fps,
            "base_dir": self.config.base_dir,
            "save_ground_camera": self.config.save_ground_camera,
            "save_segmentation": self.config.save_segmentation,
            "image_format": self.config.image_format,
            "image_quality": self.config.image_quality,
        }


# シングルトンインスタンス
data_collector = DataCollector()
