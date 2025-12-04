"""YANA GUI アプリケーション"""

from pathlib import Path
from nicegui import ui, app

from yana.session import SessionManager
from gui.chat_panel import YANAChatPanel
from gui.event_bridge import GUIEventBridge


class YANAApp:
    """YANA GUIアプリケーション"""
    
    def __init__(self):
        self.session = SessionManager()
        self.event_bridge = GUIEventBridge(self.session)
        self.chat_panel: YANAChatPanel = None
        
        # イベントハンドラ登録
        self.session.on_event(self._on_session_event)
    
    def setup(self):
        """UIセットアップ"""
        # ダークモードトグル
        dark = ui.dark_mode()
        
        with ui.header().classes('bg-primary'):
            ui.label('YANA - JetRacer Control').classes('text-xl font-bold')
            ui.space()
            with ui.button(icon='dark_mode', on_click=dark.toggle).props('flat'):
                pass
        
        with ui.row().classes('w-full h-full p-4 gap-4'):
            # 左側: 操作パネル
            with ui.column().classes('w-2/3'):
                self._setup_camera_panel()
                self._setup_session_panel()
            
            # 右側: YANAチャット
            with ui.column().classes('w-1/3'):
                self.chat_panel = YANAChatPanel(self.session)
                self.chat_panel.setup()
    
    def _setup_camera_panel(self):
        """カメラパネル"""
        with ui.card().classes('w-full'):
            ui.label('カメラ').classes('text-lg font-bold')
            
            # カメラプレビュー（プレースホルダー）
            self.camera_view = ui.label('カメラプレビュー').classes(
                'w-full h-48 bg-gray-200 flex items-center justify-center'
            )
            
            with ui.row().classes('gap-2 mt-2'):
                ui.button('撮影開始', on_click=self._start_capture, icon='play_arrow')
                ui.button('撮影停止', on_click=self._stop_capture, icon='stop')
                ui.button('1枚撮影', on_click=self._capture_one, icon='camera')
    
    def _setup_session_panel(self):
        """セッション情報パネル"""
        with ui.card().classes('w-full mt-4'):
            ui.label('セッション情報').classes('text-lg font-bold')
            
            with ui.column().classes('gap-1'):
                self.session_id_label = ui.label(f'ID: {self.session.state.session_id}')
                self.phase_label = ui.label(f'フェーズ: {self.session.state.phase.value}')
                self.frames_label = ui.label(f'フレーム: {self.session.state.total_frames}枚')
            
            with ui.row().classes('gap-2 mt-2'):
                ui.button('新規セッション', on_click=self._new_session, icon='add')
                ui.button('ディレクトリ選択', on_click=self._select_directory, icon='folder')
    
    def _on_session_event(self, event):
        """セッションイベントハンドラ"""
        # UIを更新
        self.phase_label.text = f'フェーズ: {self.session.state.phase.value}'
        self.frames_label.text = f'フレーム: {self.session.state.total_frames}枚'
        
        # YANAに通知
        if self.chat_panel:
            self.chat_panel.receive_event(event)
    
    def _start_capture(self):
        """撮影開始"""
        self.event_bridge.on_capture_started()
        ui.notify('撮影を開始しました')
    
    def _stop_capture(self):
        """撮影停止"""
        self.event_bridge.on_capture_stopped(self.session.state.total_frames)
        ui.notify('撮影を停止しました')
    
    def _capture_one(self):
        """1枚撮影"""
        # 実際の撮影処理は別途実装
        frame_num = self.session.state.total_frames + 1
        self.event_bridge.on_frame_captured(f"/tmp/frame_{frame_num:04d}.jpg", frame_num)
        ui.notify(f'フレーム {frame_num} を撮影しました')
    
    def _new_session(self):
        """新規セッション"""
        self.session.reset()
        self.session_id_label.text = f'ID: {self.session.state.session_id}'
        ui.notify('新規セッションを開始しました')
    
    async def _select_directory(self):
        """ディレクトリ選択"""
        # 簡易実装（実際はファイルダイアログを使う）
        result = await ui.run_javascript(
            "prompt('データ保存ディレクトリを入力:', '/home/jetson/data')"
        )
        if result:
            self.event_bridge.on_session_directory_changed(result)
            ui.notify(f'ディレクトリを設定: {result}')


def create_app():
    """アプリケーション作成"""
    yana_app = YANAApp()
    
    @ui.page('/')
    def main_page():
        yana_app.setup()
    
    return yana_app


# スタンドアロン実行
if __name__ in {"__main__", "__mp_main__"}:
    create_app()
    ui.run(
        title='YANA - JetRacer Control',
        host='0.0.0.0',
        port=8080,
        reload=False
    )
