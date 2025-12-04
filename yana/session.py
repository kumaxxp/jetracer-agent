"""YANAセッション状態管理"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import json

from .config import YANA_DATA_DIR, SESSION_FILE, HISTORY_FILE


class WorkPhase(Enum):
    """作業フェーズ"""
    IDLE = "idle"                       # 待機中
    DATA_COLLECTION = "data_collection" # データ収集中
    ANNOTATION = "annotation"           # アノテーション中
    TRAINING = "training"               # 訓練中
    EVALUATION = "evaluation"           # 評価中


class TaskStatus(Enum):
    """タスク状態"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """個別タスク"""
    id: str
    name: str
    status: TaskStatus
    created_at: str
    updated_at: str
    progress: float = 0.0  # 0.0 - 1.0
    details: dict = field(default_factory=dict)


@dataclass
class SessionState:
    """セッション状態"""
    session_id: str
    phase: WorkPhase
    started_at: str
    updated_at: str
    
    # データ収集関連
    collection_dir: Optional[str] = None
    total_frames: int = 0
    usable_frames: int = 0
    
    # アノテーション関連
    annotated_count: int = 0
    road_mapping: Optional[dict] = None
    
    # 訓練関連
    training_epoch: int = 0
    training_loss: float = 0.0
    
    # タスクキュー
    current_task: Optional[Task] = None
    pending_tasks: list = field(default_factory=list)
    completed_tasks: list = field(default_factory=list)
    
    # 最後のユーザー意図
    last_user_intent: str = ""


@dataclass
class Event:
    """イベント記録"""
    timestamp: str
    source: str  # "gui" | "yana" | "system"
    action: str
    details: dict = field(default_factory=dict)


class SessionManager:
    """セッション管理"""
    
    def __init__(self):
        YANA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.state: SessionState = self._load_or_create()
        self.event_handlers: list = []
    
    def _load_or_create(self) -> SessionState:
        """前回のセッションを読み込み、なければ新規作成"""
        if SESSION_FILE.exists():
            try:
                with open(SESSION_FILE) as f:
                    data = json.load(f)
                # Enumの復元
                data["phase"] = WorkPhase(data["phase"])
                if data.get("current_task"):
                    task_data = data["current_task"]
                    task_data["status"] = TaskStatus(task_data["status"])
                    data["current_task"] = Task(**task_data)
                return SessionState(**data)
            except Exception as e:
                print(f"セッション読み込みエラー: {e}")
        
        return self._create_new_session()
    
    def _create_new_session(self) -> SessionState:
        """新規セッション作成"""
        now = datetime.now().isoformat()
        return SessionState(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            phase=WorkPhase.IDLE,
            started_at=now,
            updated_at=now
        )
    
    def save(self):
        """状態を永続化"""
        self.state.updated_at = datetime.now().isoformat()
        
        # SessionStateをdict化（Enum対応）
        data = asdict(self.state)
        data["phase"] = self.state.phase.value
        if self.state.current_task:
            data["current_task"]["status"] = self.state.current_task.status.value
        
        with open(SESSION_FILE, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def record_event(self, source: str, action: str, details: dict = None):
        """イベントを記録"""
        event = Event(
            timestamp=datetime.now().isoformat(),
            source=source,
            action=action,
            details=details or {}
        )
        
        # 履歴ファイルに追記
        with open(HISTORY_FILE, 'a') as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + '\n')
        
        # ハンドラに通知
        for handler in self.event_handlers:
            handler(event)
        
        # 状態更新
        self.save()
        
        return event
    
    def on_event(self, handler):
        """イベントハンドラ登録"""
        self.event_handlers.append(handler)
    
    def start_new_collection(self, directory: str):
        """新しいデータ収集を開始（前回の作業をリセット）"""
        self.state = self._create_new_session()
        self.state.phase = WorkPhase.DATA_COLLECTION
        self.state.collection_dir = directory
        self.record_event("system", "new_collection_started", {"directory": directory})
    
    def get_context_for_yana(self) -> str:
        """YANA用のコンテキスト文字列を生成"""
        ctx = []
        ctx.append(f"現在のフェーズ: {self.state.phase.value}")
        
        if self.state.phase == WorkPhase.DATA_COLLECTION:
            ctx.append(f"収集ディレクトリ: {self.state.collection_dir}")
            ctx.append(f"撮影フレーム数: {self.state.total_frames}")
            ctx.append(f"使用可能フレーム: {self.state.usable_frames}")
        
        elif self.state.phase == WorkPhase.ANNOTATION:
            ctx.append(f"アノテーション済み: {self.state.annotated_count}")
            if self.state.road_mapping:
                ctx.append(f"ROADマッピング設定済み: {len(self.state.road_mapping)}クラス")
        
        elif self.state.phase == WorkPhase.TRAINING:
            ctx.append(f"訓練エポック: {self.state.training_epoch}")
            ctx.append(f"現在のLoss: {self.state.training_loss:.4f}")
        
        if self.state.current_task:
            task = self.state.current_task
            ctx.append(f"実行中タスク: {task.name} ({task.progress*100:.0f}%)")
        
        if self.state.pending_tasks:
            ctx.append(f"待機中タスク: {len(self.state.pending_tasks)}件")
        
        return "\n".join(ctx)
    
    def get_recent_events(self, count: int = 10) -> list[Event]:
        """直近のイベントを取得"""
        events = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE) as f:
                lines = f.readlines()
                for line in lines[-count:]:
                    try:
                        data = json.loads(line)
                        events.append(Event(**data))
                    except:
                        pass
        return events
    
    def is_resumable(self) -> bool:
        """前回の作業が再開可能か"""
        return self.state.phase != WorkPhase.IDLE
    
    def reset(self):
        """セッションをリセット"""
        self.state = self._create_new_session()
        self.save()
