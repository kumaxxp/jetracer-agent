"""YANAチャットパネル - NiceGUI"""

import asyncio
from pathlib import Path
from nicegui import ui, context

from yana.session import SessionManager, Event
from agent.agent import YANAAgent


class YANAChatPanel:
    """YANAチャットパネル"""

    def __init__(self, session_manager: SessionManager):
        self.session = session_manager
        self.agent: YANAAgent = None
        self.pending_events: list[Event] = []
        self.chat_log = None
        self.input_field = None
        self.is_processing = False
        self._client = None  # クライアント参照を保持

    def _is_client_valid(self) -> bool:
        """クライアントがまだ有効かチェック"""
        try:
            if self._client is None:
                return False
            # クライアントが削除されていないかチェック
            return self._client.id in context.client.instances
        except Exception:
            return False
    
    def setup(self):
        """UIセットアップ"""
        with ui.column().classes('w-full h-full'):
            # ヘッダー
            with ui.row().classes('items-center gap-2 p-2 bg-gray-100 rounded-t'):
                # アバター画像（存在すれば）
                avatar_path = Path(__file__).parent.parent / "static" / "yana_avatar.png"
                if avatar_path.exists():
                    ui.image(str(avatar_path)).classes('w-10 h-10 rounded-full')
                else:
                    ui.label('🤖').classes('text-2xl')
                
                ui.label('YANA').classes('text-xl font-bold')
                ui.label('Your Autonomous Navigation Assistant').classes('text-xs text-gray-500')
            
            # チャットログ
            self.chat_log = ui.column().classes(
                'flex-grow overflow-auto bg-white p-3 border rounded'
            ).style('max-height: 400px; min-height: 300px;')
            
            # 入力エリア
            with ui.row().classes('w-full mt-2 gap-2'):
                self.input_field = ui.input(
                    placeholder='YANAに指示...'
                ).classes('flex-grow').on('keydown.enter', self._on_send)
                
                ui.button('送信', on_click=self._on_send).classes('px-4')
        
        # 起動シーケンスをスケジュール
        ui.timer(0.5, self._run_startup, once=True)
    
    async def _run_startup(self):
        """起動処理"""
        self._log_system("YANA を起動中...")
        
        try:
            # エージェント初期化
            self.agent = YANAAgent()
            self.agent.session_manager = self.session
            
            # コールバック設定
            self.agent.on_tool_call = self._on_tool_call
            
            # MCP接続
            await self.agent.connect_mcp()
            
            # 起動シーケンス実行
            async for chunk in self.agent.startup():
                self._log_assistant(chunk)
            
            # 保留中のイベントを処理
            for event in self.pending_events:
                await self._process_event(event)
            self.pending_events.clear()
            
        except Exception as e:
            self._log_system(f"起動エラー: {e}")
    
    async def _on_send(self):
        """送信ボタン/Enterキー"""
        if self.is_processing:
            return
        
        user_msg = self.input_field.value.strip()
        if not user_msg:
            return
        
        self.input_field.value = ''
        self._log_user(user_msg)
        
        if not self.agent:
            self._log_system("YANAはまだ起動中です。しばらくお待ちください。")
            return
        
        self.is_processing = True
        
        try:
            async for chunk in self.agent.run_with_tools(user_msg):
                self._log_assistant(chunk)
        except Exception as e:
            self._log_system(f"エラー: {e}")
        finally:
            self.is_processing = False
    
    def receive_event(self, event: Event):
        """GUIからのイベントを受信"""
        if self.agent:
            asyncio.create_task(self._process_event(event))
        else:
            self.pending_events.append(event)
    
    async def _process_event(self, event: Event):
        """イベントをYANAに通知"""
        if not self.agent:
            return
        
        notification = self.agent.receive_gui_event(event)
        response = await self.agent.process_notification(notification)
        
        if response:
            self._log_assistant(response)
    
    def _on_tool_call(self, tool_name: str, args: dict):
        """ツール呼び出し時のコールバック"""
        self._log_tool(tool_name, args)
    
    def _log_user(self, message: str):
        """ユーザーメッセージを表示"""
        with self.chat_log:
            with ui.row().classes('justify-end w-full'):
                ui.label(message).classes(
                    'bg-blue-100 text-blue-900 p-2 rounded-lg max-w-[80%]'
                )
    
    def _log_assistant(self, message: str):
        """YANAメッセージを表示"""
        with self.chat_log:
            with ui.row().classes('justify-start w-full'):
                ui.label(message).classes(
                    'bg-green-50 text-green-900 p-2 rounded-lg max-w-[80%] whitespace-pre-wrap'
                )
        # 自動スクロール
        ui.run_javascript('document.querySelector(".overflow-auto").scrollTop = document.querySelector(".overflow-auto").scrollHeight')
    
    def _log_tool(self, tool_name: str, args: dict):
        """ツール呼び出しを表示"""
        with self.chat_log:
            with ui.row().classes('justify-start w-full'):
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                ui.label(f"🔧 {tool_name}({args_str})").classes(
                    'bg-gray-100 text-gray-600 text-xs p-1 rounded font-mono'
                )
    
    def _log_system(self, message: str):
        """システムメッセージを表示"""
        with self.chat_log:
            ui.label(message).classes(
                'text-gray-500 text-sm italic w-full text-center py-1'
            )
