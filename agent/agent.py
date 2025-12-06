#!/usr/bin/env python3
"""
YANA Agent - Gemma 3 4B + Ollama + Hybrid アプローチ
ベンチマーク結果に基づく最適化版
"""

import asyncio
import base64
import json
import re
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# =============================================================================
# 設定
# =============================================================================

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "llm_config.yaml"
MCP_SERVER_PATH = Path(__file__).parent.parent / "mcp_server" / "server.py"
VENV_PYTHON = Path(__file__).parent.parent / "venv" / "bin" / "python3"


def load_config() -> dict:
    """設定ファイルを読み込む"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {
        "llm": {
            "model": "gemma3:4b",
            "ollama_url": "http://localhost:11434/api/generate",
            "approach": "hybrid",
            "temperature": 0.1,
            "max_tokens": 300,
        }
    }


# =============================================================================
# データ構造
# =============================================================================

@dataclass
class ToolCall:
    """ツール呼び出し情報"""
    name: str
    args: Dict[str, Any]


@dataclass
class AgentResponse:
    """エージェント応答"""
    tool_call: Optional[ToolCall]
    reply: Optional[str]
    raw_output: str


# =============================================================================
# YANAエージェント
# =============================================================================

@dataclass
class YANAAgent:
    """YANA Agent - Gemma 3 4B + Ollama版"""

    # 設定
    ollama_url: str = "http://localhost:11434/api/generate"
    model: str = "gemma3:4b"
    approach: str = "hybrid"
    temperature: float = 0.1
    max_tokens: int = 300

    # MCP
    mcp_session: Optional[ClientSession] = None
    _stdio_client: Any = field(default=None, repr=False)
    _read: Any = field(default=None, repr=False)
    _write: Any = field(default=None, repr=False)

    # ツール
    tools: Dict[str, dict] = field(default_factory=dict)

    # コールバック
    on_tool_call: Optional[Callable] = None

    def __post_init__(self):
        """設定ファイルから初期化"""
        config = load_config()
        llm_config = config.get("llm", {})

        self.ollama_url = llm_config.get("ollama_url", self.ollama_url)
        self.model = llm_config.get("model", self.model)
        self.approach = llm_config.get("approach", self.approach)
        self.temperature = llm_config.get("temperature", self.temperature)
        self.max_tokens = llm_config.get("max_tokens", self.max_tokens)

        logger.info(f"YANAAgent初期化: model={self.model}, approach={self.approach}")

    # -------------------------------------------------------------------------
    # Ollama接続
    # -------------------------------------------------------------------------

    def unload_model(self) -> bool:
        """Ollamaモデルをアンロードしてメモリを解放"""
        try:
            # keep_alive=0 でモデルを即座にアンロード
            payload = {
                "model": self.model,
                "keep_alive": 0
            }
            response = requests.post(
                self.ollama_url.replace("/api/generate", "/api/generate"),
                json=payload,
                timeout=10
            )
            logger.info(f"モデルアンロード: {self.model}")
            return True
        except Exception as e:
            logger.warning(f"モデルアンロード失敗: {e}")
            return False

    def check_ollama_connection(self) -> bool:
        """Ollama接続確認"""
        # Note: モデルのアンロードは明示的に呼び出す（頻繁なロード/アンロードを防止）
        try:
            response = requests.get(
                self.ollama_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama接続エラー: {e}")
            return False

    def _call_ollama(self, prompt: str, json_mode: bool = False) -> str:
        """Ollama API呼び出し"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            logger.warning("Ollama応答タイムアウト（120秒）")
            return "（応答生成に時間がかかっています）"
        except Exception as e:
            logger.error(f"Ollama呼び出しエラー: {e}")
            return ""

    def _call_ollama_vision(self, prompt: str, image_path: str) -> str:
        """Ollama Vision API呼び出し（マルチモーダル）"""
        # 画像をBase64エンコード
        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"画像読み込みエラー: {e}")
            return f"画像を読み込めませんでした: {e}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            logger.warning("Vision API応答タイムアウト（180秒）")
            return "（画像分析に時間がかかっています）"
        except Exception as e:
            logger.error(f"Vision API呼び出しエラー: {e}")
            return f"画像分析エラー: {e}"

    def describe_image(self, image_path: str, question: str = None) -> str:
        """画像をGemma 3で分析・説明"""
        if question:
            prompt = f"""この画像について質問に答えてください。
質問: {question}
日本語で簡潔に回答してください。"""
        else:
            prompt = """この画像に何が映っていますか？
以下の点を日本語で簡潔に説明してください：
- 主な被写体
- 場所や環境
- 注目すべき特徴"""

        return self._call_ollama_vision(prompt, image_path)

    # -------------------------------------------------------------------------
    # MCP接続
    # -------------------------------------------------------------------------

    async def connect_mcp(self):
        """MCPサーバーに接続"""
        # venv内のPythonを使用（MCPモジュールがvenvにインストールされている）
        python_cmd = str(VENV_PYTHON) if VENV_PYTHON.exists() else "python3"
        server_params = StdioServerParameters(
            command=python_cmd,
            args=[str(MCP_SERVER_PATH)],
        )

        self._stdio_client = stdio_client(server_params)
        self._read, self._write = await self._stdio_client.__aenter__()
        self.mcp_session = ClientSession(self._read, self._write)
        await self.mcp_session.__aenter__()
        await self.mcp_session.initialize()

        # ツール定義を取得してローカル登録
        tools_result = await self.mcp_session.list_tools()
        for tool in tools_result.tools:
            self.tools[tool.name] = {
                "description": tool.description,
                "schema": tool.inputSchema,
            }

        print(f"Connected to MCP server. {len(self.tools)} tools available.")

    async def disconnect_mcp(self):
        """MCPサーバーから切断"""
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)
        if self._stdio_client:
            await self._stdio_client.__aexit__(None, None, None)

        # モデルをアンロードしてメモリ解放
        print("モデルをアンロード中...")
        self.unload_model()

    async def _execute_tool(self, name: str, args: dict) -> str:
        """MCPツールを実行"""
        if self.mcp_session:
            result = await self.mcp_session.call_tool(name, args)
            return result.content[0].text
        else:
            return json.dumps({"error": "MCP not connected"})

    # -------------------------------------------------------------------------
    # ツール検出（Hybridアプローチ）
    # -------------------------------------------------------------------------

    # Vision機能用キーワード（MCP非依存、Gemma 3マルチモーダル）
    VISION_KEYWORDS = [
        "何が見える", "何が映って", "説明して", "見て", "画像を見て",
        "写真を見て", "これは何", "何ですか", "教えて"
    ]

    # キーワードマッピング（拡張版）
    # 優先度順（上位ほど優先）
    TOOL_KEYWORDS = {
        # 高優先度: 特定的なキーワード
        "check_camera": [
            "カメラ", "撮影", "状態確認", "動いて", "接続", "起動", "カメラ確認"
        ],
        # 低優先度: 汎用的なキーワード
        "list_images": [
            "画像一覧", "画像を見", "リスト", "ファイル一覧", "中身を確認",
            "フォルダの中", "ディレクトリ", "何がある", "画像ファイル"
        ],
        "analyze_frame": [
            "明るさ", "コントラスト", "分析", "フレーム分析"
        ],
        "detect_objects": [
            "物体検出", "何が映", "検出", "認識", "YOLO", "オブジェクト"
        ],
        "check_system_resources": [
            "メモリ", "リソース", "GPU", "システム", "使用状況"
        ],
        "check_model_files": [
            "モデルファイル", "モデル確認", "ファイル存在"
        ],
        "evaluate_quality": [
            "品質", "ブレ", "露出", "クオリティ", "使える"
        ],
        "segment_image": [
            "セグメント", "領域", "道路検出", "セグメンテーション"
        ],
        "get_image_info": [
            "画像情報", "メタ情報", "サイズ", "解像度"
        ],
    }

    def _detect_tool(self, text: str) -> Optional[str]:
        """キーワードマッチングでツール検出（スコアベース）"""
        best_tool = None
        best_score = 0

        for tool_name, keywords in self.TOOL_KEYWORDS.items():
            if tool_name not in self.tools:  # 登録済みツールのみ
                continue

            score = 0
            for kw in keywords:
                if kw in text:
                    # キーワード長が長いほど高スコア（より特定的）
                    score += len(kw)

            if score > best_score:
                best_score = score
                best_tool = tool_name

        return best_tool

    def _detect_vision_request(self, text: str) -> bool:
        """Vision（画像説明）リクエストを検出"""
        return any(kw in text for kw in self.VISION_KEYWORDS)

    # -------------------------------------------------------------------------
    # 引数抽出（改善版 - 90%目標）
    # -------------------------------------------------------------------------

    def _extract_args_for_tool(self, tool: str, text: str) -> Dict[str, Any]:
        """ツールに応じた引数抽出（改善版）"""

        if tool == "list_images":
            return self._extract_folder_arg(text)

        elif tool in ["analyze_frame", "evaluate_quality", "segment_image",
                      "get_image_info", "detect_objects"]:
            return self._extract_image_path_arg(text)

        return {}

    def _extract_folder_arg(self, text: str) -> Dict[str, Any]:
        """フォルダ引数を抽出"""

        # パターン1: 「○○フォルダ」「○○ディレクトリ」
        folder_match = re.search(r'(\w+)(?:フォルダ|ディレクトリ)', text)
        if folder_match:
            return {"folder": folder_match.group(1)}

        # パターン2: 「○○の中身」「○○の画像」
        of_match = re.search(r'(\w+)(?:の中身|の画像|の中|にある)', text)
        if of_match:
            return {"folder": of_match.group(1)}

        # パターン3: 絶対パス /home/jetson/... ~/...
        abs_path = re.search(r'(/[\w/.-]+|~/[\w/.-]+)', text)
        if abs_path:
            path = abs_path.group(1)
            if path.startswith("~"):
                path = str(Path.home() / path[2:])
            return {"folder": path}

        # パターン4: 相対パス ./images, data/raw
        rel_path = re.search(r'(\.?/[\w/.-]+|\w+/\w+)', text)
        if rel_path:
            return {"folder": rel_path.group(1)}

        # パターン5: クォート内 「〜」"〜"
        quote_match = re.search(r'[「「"\']([\w/.-]+)[」」"\']', text)
        if quote_match:
            return {"folder": quote_match.group(1)}

        # パターン6: セッション名 session_YYYYMMDD_HHMMSS
        session_match = re.search(r'(session_\d{8}_\d{6})', text)
        if session_match:
            return {"folder": session_match.group(1)}

        # パターン7: 一般的なフォルダ名
        common_folders = ["data", "images", "output", "raw", "raw_images",
                         "frames", "captures", "annotated", "train", "test"]
        for folder in common_folders:
            if folder in text.lower():
                return {"folder": folder}

        # デフォルト
        return {"folder": "data"}

    def _extract_image_path_arg(self, text: str) -> Dict[str, Any]:
        """画像パス引数を抽出"""

        # パターン1: 完全なファイルパス（拡張子付き）
        full_path = re.search(
            r'((?:/[\w.-]+)+\.(?:jpg|jpeg|png|bmp))|'
            r'((?:~/[\w.-]+)+\.(?:jpg|jpeg|png|bmp))|'
            r'([\w.-]+/[\w.-]+\.(?:jpg|jpeg|png|bmp))|'
            r'([\w.-]+\.(?:jpg|jpeg|png|bmp))',
            text, re.IGNORECASE
        )
        if full_path:
            path = next(g for g in full_path.groups() if g)
            if path.startswith("~"):
                path = str(Path.home() / path[2:])
            return {"image_path": path}

        # パターン2: クォート内
        quote_match = re.search(r'[「「"\']([\w/.-]+\.(?:jpg|jpeg|png|bmp))[」」"\']', text, re.IGNORECASE)
        if quote_match:
            return {"image_path": quote_match.group(1)}

        # パターン3: frame_XXX パターン
        frame_match = re.search(r'(frame_?\d+)', text, re.IGNORECASE)
        if frame_match:
            return {"image_path": f"{frame_match.group(1)}.jpg"}

        # パターン4: img_XXX パターン
        img_match = re.search(r'(img_?\d+)', text, re.IGNORECASE)
        if img_match:
            return {"image_path": f"{img_match.group(1)}.jpg"}

        return {"image_path": ""}

    # -------------------------------------------------------------------------
    # 入力処理
    # -------------------------------------------------------------------------

    def process(self, user_input: str) -> AgentResponse:
        """ユーザー入力を処理（Hybridアプローチ）"""

        # Vision（画像説明）リクエストを検出
        if self._detect_vision_request(user_input):
            # 画像パスを抽出
            args = self._extract_image_path_arg(user_input)
            return AgentResponse(
                tool_call=ToolCall(name="_vision_describe", args=args),
                reply=None,
                raw_output=f"Vision request detected"
            )

        # キーワードでツール検出
        detected_tool = self._detect_tool(user_input)

        if detected_tool:
            args = self._extract_args_for_tool(detected_tool, user_input)
            return AgentResponse(
                tool_call=ToolCall(name=detected_tool, args=args),
                reply=None,
                raw_output=f"Keyword match: {detected_tool}"
            )

        # ツール不要 → 会話応答
        return AgentResponse(
            tool_call=None,
            reply=self._generate_chat_reply(user_input),
            raw_output="No tool matched"
        )

    def _generate_chat_reply(self, user_input: str) -> str:
        """通常の会話応答を生成"""
        prompt = f"""あなたはYANA（Your Autonomous Navigation Assistant）、JetRacer自律走行プロジェクトのアシスタントです。
親切に簡潔に日本語で回答してください。
自己紹介を求められたら「"Your Autonomous Navigation Assistant"、やなです。JetRacer自律走行プロジェクトのアシスタントです。」と答えてください。

ユーザー: {user_input}
YANA:"""
        return self._call_ollama(prompt)

    def _explain_result(self, tool_name: str, result: Any) -> str:
        """ツール実行結果を自然言語で説明"""
        result_str = json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, (dict, list)) else str(result)

        prompt = f"""以下のツール実行結果をユーザーに分かりやすく日本語で説明してください。

ツール: {tool_name}
結果:
{result_str}

簡潔に説明（3-4文以内）:"""
        return self._call_ollama(prompt)

    # -------------------------------------------------------------------------
    # 実行
    # -------------------------------------------------------------------------

    async def run(self, user_input: str) -> str:
        """ユーザー入力を処理し、必要ならツールを実行"""
        response = self.process(user_input)

        if response.tool_call:
            tool_name = response.tool_call.name
            args = response.tool_call.args

            # コールバック
            if self.on_tool_call:
                self.on_tool_call(tool_name, args)

            # Vision（画像説明）の特別処理
            if tool_name == "_vision_describe":
                image_path = args.get("image_path", "")
                if not image_path:
                    # 最新のカメラ画像を使用
                    image_path = str(Path(__file__).parent.parent / "tmp" / "yana_startup_frame.jpg")
                print(f"  [Vision] describe_image({image_path})")
                return self.describe_image(image_path, user_input)

            print(f"  [Tool] {tool_name}({args})")

            try:
                result_str = await self._execute_tool(tool_name, args)
                result = json.loads(result_str)
                print(f"  [Result] {result_str[:100]}...")

                # 結果を自然言語で説明
                return self._explain_result(tool_name, result)
            except Exception as e:
                logger.error(f"ツール実行エラー: {e}")
                return f"ツール実行中にエラーが発生しました: {e}"

        return response.reply or "すみません、よく分かりませんでした。"

    async def run_with_tools(self, user_input: str) -> AsyncGenerator[str, None]:
        """ストリーミング対応の実行（GUI互換用）"""
        result = await self.run(user_input)
        yield result

    # -------------------------------------------------------------------------
    # 起動シーケンス
    # -------------------------------------------------------------------------

    async def startup(self) -> AsyncGenerator[str, None]:
        """起動シーケンスを実行"""

        # セルフチェック実行
        check_results = await self._run_startup_checks()

        # 結果をLLMでコメント生成
        comment = self._build_startup_comment(check_results)
        yield comment

    async def _run_startup_checks(self) -> dict:
        """起動時のセルフチェック"""
        results = {}

        # カメラチェック
        try:
            print("  [Tool] check_camera({})")
            camera_result = await self._execute_tool("check_camera", {})
            results["camera"] = json.loads(camera_result)
            print(f"  [Result] {camera_result[:80]}...")

            if results["camera"].get("connected"):
                frame_path = results["camera"].get("frame_path")
                if frame_path:
                    # 明るさ分析
                    print(f"  [Tool] analyze_frame({{image_path: {frame_path}}})")
                    analysis = await self._execute_tool("analyze_frame", {"image_path": frame_path})
                    results["analysis"] = json.loads(analysis)
                    print(f"  [Result] {analysis[:80]}...")

                    # Gemma 3 Vision で画像分析
                    print(f"  [Vision] describe_image({frame_path})")
                    vision_desc = self.describe_image(frame_path, "この画像に何が映っていますか？簡潔に説明してください。")
                    results["vision"] = vision_desc
                    print(f"  [Result] {vision_desc[:80]}...")
        except Exception as e:
            results["camera"] = {"connected": False, "error": str(e)}

        # リソースチェック
        try:
            print("  [Tool] check_system_resources({})")
            resources = await self._execute_tool("check_system_resources", {})
            results["resources"] = json.loads(resources)
            print(f"  [Result] {resources[:80]}...")
        except Exception as e:
            results["resources"] = {"error": str(e)}

        # モデルチェック
        try:
            print("  [Tool] check_model_files({})")
            models = await self._execute_tool("check_model_files", {})
            results["models"] = json.loads(models)
            print(f"  [Result] {models[:80]}...")
        except Exception as e:
            results["models"] = {"error": str(e)}

        return results

    def _build_startup_comment(self, results: dict) -> str:
        """起動チェック結果からコメント生成"""
        prompt = """あなたはYANA、JetRacerのアシスタントです。
システム起動時のセルフチェック結果を報告してください。

まず「"Your Autonomous Navigation Assistant"、やな、起動しました。」と挨拶してから、
以下のチェック結果を簡潔に報告してください。

チェック結果:
"""
        # カメラ
        camera = results.get("camera", {})
        if camera.get("connected"):
            prompt += f"- カメラ: OK ({camera.get('resolution', '不明')})\n"
        else:
            prompt += f"- カメラ: NG ({camera.get('error', '接続エラー')})\n"

        # 明るさ
        analysis = results.get("analysis", {})
        if analysis:
            brightness = analysis.get("brightness", 0)
            prompt += f"- 明るさ: {brightness:.0f}\n"

        # Vision分析結果
        vision = results.get("vision", "")
        if vision:
            prompt += f"- カメラ映像: {vision[:100]}...\n" if len(vision) > 100 else f"- カメラ映像: {vision}\n"

        # リソース
        resources = results.get("resources", {})
        if resources and not resources.get("error"):
            free_mb = resources.get("memory_free_mb", 0)
            prompt += f"- メモリ: {free_mb}MB空き\n"

        # モデル
        models = results.get("models", {})
        if models and not models.get("error"):
            llm_ok = models.get("llm", {}).get("exists", False)
            prompt += f"- モデル: {'OK' if llm_ok else 'NG'}\n"

        prompt += "\n簡潔に報告し、最後に「何かお手伝いできることはありますか？」で締めてください。"

        return self._call_ollama(prompt)


# =============================================================================
# メイン（CLIテスト用）
# =============================================================================

async def main():
    """CLIテスト"""
    agent = YANAAgent()

    # Ollama接続確認
    if not agent.check_ollama_connection():
        print("ERROR: Ollamaに接続できません")
        print("ollama serve を実行してください")
        return

    print(f"Ollama接続OK: {agent.model}")

    try:
        await agent.connect_mcp()

        print("\n--- 起動シーケンス ---")
        async for chunk in agent.startup():
            print(f"\nYANA: {chunk}")

        print("\n" + "="*50)
        print("対話モード（quitで終了）")
        print("="*50)

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue

                response = await agent.run(user_input)
                print(f"\nYANA: {response}")

            except KeyboardInterrupt:
                break

    finally:
        await agent.disconnect_mcp()
        print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
