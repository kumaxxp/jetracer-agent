# YANA Phase 2 統合仕様書 v1.1

**作成日**: 2025年12月29日  
**プロジェクト**: YANA (Your Autonomous Navigation Assistant)  
**目的**: JetRacer強化計画の次段階への移行

---

## 1. エグゼクティブサマリー

### 1.1 現在の状態

| 項目 | 状態 |
|------|------|
| ハードウェア | ✅ マイク・IMU・LiDAR固定完了 |
| 音声取得 | ✅ 実験成功、取得可能を確認 |
| リポジトリ | ✅ yana-brain, jetracer-agent 存在 |
| 要素技術調査 | ✅ NanoSAM, DA3, FunctionGemma 調査完了 |

### 1.2 目指すゴール

```mermaid
flowchart TB
    subgraph YANA["YANA 最終形態"]
        subgraph Race["レース走行モード"]
            MoE["MoE (resnet18)<br/>高速・軽量"]
            Acoustic["音響状態推定<br/>先行指標活用"]
            Steering["ステアリング制御"]
            Throttle["スロットル制御<br/>低速安定走行"]
            MoE --> Steering
            Acoustic --> Throttle
        end
        
        subgraph DataCollection["データ収集モード"]
            NanoSAM["NanoSAM<br/>セグメンテーション"]
            RoadDetect["走行可能領域検出"]
            AutoDrive["自動走行"]
            AutoAnnot["自動アノテーション"]
            NanoSAM --> RoadDetect
            RoadDetect --> AutoDrive
            RoadDetect --> AutoAnnot
        end
    end
```

### 1.3 開発パイプライン全体像

```mermaid
gantt
    title YANA Phase 2 開発スケジュール
    dateFormat  YYYY-MM-DD
    section Phase 2-A
    NanoSAM動作確認     :a1, 2025-01-06, 3d
    音響特徴抽出実験    :a2, after a1, 3d
    FunctionGemma実験   :a3, after a2, 4d
    結果分析・方針調整  :a4, after a3, 4d
    section Phase 2-B
    音響スロットル制御  :b1, after a4, 4d
    NanoSAM統合        :b2, after b1, 4d
    MCP統合            :b3, after b2, 4d
    低速走行テスト      :b4, after b3, 2d
    section Phase 2-C
    自動データ収集      :c1, after b4, 4d
    データ収集実行      :c2, after c1, 4d
    品質検証・整理      :c3, after c2, 4d
    追加収集           :c4, after c3, 2d
    section Phase 2-D
    MoE学習            :d1, after c4, 4d
    TensorRT変換       :d2, after d1, 3d
    レースモードテスト  :d3, after d2, 3d
    パラメータ調整      :d4, after d3, 4d
```

---

## 2. システムアーキテクチャ

### 2.1 リポジトリ構成

```mermaid
flowchart TB
    subgraph PC["PC (Windows/Ubuntu)"]
        subgraph YanaBrain["yana-brain"]
            DA3["DA3処理<br/>3Dマップ生成"]
            MoETrain["MoE学習<br/>訓練パイプライン"]
            AcousticTrain["音響モデル学習<br/>状態分類器訓練"]
            DataMgmt["データ管理<br/>収集データの整理・検証"]
        end
    end
    
    subgraph Jetson["Jetson Orin Nano Super"]
        subgraph JetracerAgent["jetracer-agent"]
            Sensors["センサー統合<br/>Camera, IMU, LiDAR, Mic"]
            NanoSAMInf["NanoSAM推論<br/>セグメンテーション"]
            AcousticEst["音響推定<br/>状態推定・スロットル制御"]
            MoEInf["MoE推論<br/>ステアリング制御"]
            MCPServer["MCP Server<br/>ツール提供"]
            FuncGemma["FunctionGemma<br/>ローカルLLMエージェント"]
        end
    end
    
    PC <-->|"SSH / HTTP API<br/>モデル転送"| Jetson
```

### 2.2 jetracer-agent 構成（計画）

```mermaid
flowchart LR
    subgraph Sensors["sensors/"]
        camera["camera.py"]
        imu["imu_bno055.py"]
        lidar["lidar_vl53l7cx.py"]
        mic["microphone.py"]
    end
    
    subgraph Perception["perception/"]
        nanosam["nanosam_segmentation.py"]
        road["road_detector.py"]
        acoustic["acoustic_observer.py"]
    end
    
    subgraph Control["control/"]
        steering["steering_controller.py"]
        throttle["throttle_controller.py"]
        safety["safety_monitor.py"]
    end
    
    subgraph Inference["inference/"]
        moe["moe_inference.py"]
        trt["tensorrt_utils.py"]
    end
    
    subgraph Agent["agent/"]
        funcgemma["function_gemma.py"]
        mcp["mcp_server.py"]
    end
    
    Sensors --> Perception
    Perception --> Control
    Inference --> Control
    Agent --> Control
```

### 2.3 yana-brain 構成（計画）

```mermaid
flowchart LR
    subgraph Mapping["mapping/"]
        da3["da3_processor.py"]
        wallmap["wall_map_generator.py"]
    end
    
    subgraph Training["training/"]
        moe_train["moe_trainer.py"]
        acoustic_train["acoustic_trainer.py"]
        augment["data_augmentation.py"]
    end
    
    subgraph DataMgmt["data_management/"]
        dataset["dataset_builder.py"]
        quality["quality_checker.py"]
        validator["annotation_validator.py"]
    end
    
    subgraph Export["export/"]
        onnx["onnx_exporter.py"]
        tensorrt["tensorrt_builder.py"]
    end
    
    Mapping --> Training
    DataMgmt --> Training
    Training --> Export
```

---

## 3. Phase 2-A: 要素実験（2週間）

### 3.1 実験1: NanoSAM動作確認

**目的**: Jetson上でNanoSAMが実用的な速度で動作することを確認

```mermaid
flowchart LR
    Install["NanoSAM<br/>インストール"] --> TRT["TensorRT<br/>エンジン生成"]
    TRT --> Test["動作テスト"]
    Test --> Bench["ベンチマーク<br/>計測"]
```

**手順**:
```bash
# 1. NanoSAMインストール
cd ~/jetracer-agent
git clone https://github.com/NVIDIA-AI-IOT/nanosam
cd nanosam
pip install -e . --break-system-packages

# 2. TensorRTエンジン生成
python scripts/export_sam_encoder.py
python scripts/export_sam_decoder.py

# 3. 動作テスト
python scripts/demo.py --image test.jpg
```

**検証項目**:
| 項目 | 目標値 | 測定方法 |
|------|--------|---------|
| 推論時間 | < 50ms | time.perf_counter() |
| メモリ使用量 | < 1GB | tegrastats |
| セグメンテーション品質 | 目視確認 | 走行可能領域の検出精度 |

**成果物**:
- `perception/nanosam_segmentation.py`
- ベンチマーク結果レポート

### 3.2 実験2: 音響特徴抽出

**目的**: モーター・サーボ音からMFCC特徴量を抽出し、状態との相関を確認

```mermaid
flowchart LR
    Record["各状態で<br/>録音"] --> Extract["MFCC<br/>特徴抽出"]
    Extract --> Visualize["スペクトログラム<br/>可視化"]
    Visualize --> Compare["状態間<br/>比較分析"]
```

**収集する状態**:
```mermaid
stateDiagram-v2
    [*] --> STOPPED: 電源ON
    STOPPED --> IDLE: PWM出力開始
    IDLE --> STARTING: 動き出し
    STARTING --> RUNNING: 定常走行
    RUNNING --> SPINNING: グリップ喪失
    RUNNING --> STOPPED: 停止
    SPINNING --> RUNNING: グリップ回復
```

**検証項目**:
| 状態 | 確認ポイント |
|------|-------------|
| 停止 vs 動作 | エネルギーレベルの差 |
| 低速 vs 高速 | 基本周波数の違い |
| 空転 vs 接地 | スペクトル形状の差 |
| 動き出し前兆 | 時系列変化パターン |

**成果物**:
- 各状態のMFCC可視化
- `sensors/microphone.py`
- `perception/acoustic_observer.py` (基礎版)

### 3.3 実験3: FunctionGemma適用

**目的**: FunctionGemmaがJetson上でMCPツール選択を実行できることを確認

```mermaid
flowchart LR
    Input["ユーザー入力<br/>自然言語"] --> FuncGemma["FunctionGemma<br/>ツール選択"]
    FuncGemma --> ToolCall["MCP Tool<br/>呼び出し"]
    ToolCall --> Result["結果<br/>返却"]
```

**テストケース**:
| 入力 | 期待されるツール |
|------|----------------|
| 「前に進んで」 | set_throttle |
| 「カメラの画像を見せて」 | get_camera_frame |
| 「データ収集を始めて」 | start_recording |
| 「今の姿勢を教えて」 | get_imu_data |

**検証項目**:
| 項目 | 目標値 |
|------|--------|
| ツール選択正確度 | > 80% |
| 応答時間 | < 1秒 |
| メモリ使用量 | < 2GB |

**成果物**:
- `agent/function_gemma.py`
- ツール選択精度レポート

---

## 4. Phase 2-B: 基盤構築（2週間）

### 4.1 音響ベース低速走行システム

**目標**: 音響フィードバックで安定した低速走行を実現

```mermaid
flowchart TB
    subgraph Input["入力"]
        Mic["マイク<br/>(16kHz)"]
    end
    
    subgraph Processing["処理"]
        Feature["特徴抽出<br/>(MFCC等)"]
        Classify["状態分類<br/>(軽量CNN)"]
    end
    
    subgraph Output["出力"]
        State["音響状態<br/>motor_state<br/>speed_estimate<br/>starting_likelihood"]
    end
    
    subgraph Controller["ヒステリシス制御"]
        Stopped["STOPPED"]
        Starting["STARTING"]
        Running["RUNNING"]
    end
    
    subgraph Actuator["アクチュエータ"]
        ESC["PWM出力<br/>(ESC)"]
    end
    
    Mic --> Feature --> Classify --> State
    State --> Controller
    Stopped -->|"起動PWM"| Starting
    Starting -->|"PWM降下"| Running
    Running -->|"停止検出"| Stopped
    Controller --> ESC
```

**状態遷移詳細**:

```mermaid
stateDiagram-v2
    [*] --> STOPPED
    
    STOPPED --> STARTING: start() / 起動PWM出力
    
    STARTING --> STARTING: motor_state==STOPPED / PWM++
    STARTING --> RUNNING: motor_state==STARTING or RUNNING / PWM降下
    
    RUNNING --> RUNNING: PID制御で速度維持
    RUNNING --> STOPPED: motor_state==STOPPED / 停止検出
    
    note right of STARTING
        静止摩擦を超えるため
        高めのPWMで蹴り出す
    end note
    
    note right of RUNNING
        動摩擦領域で
        低いPWMで維持
    end note
```

**実装タスク**:
| タスク | ファイル | 優先度 |
|--------|---------|--------|
| マイク入力モジュール | `sensors/microphone.py` | 高 |
| MFCC特徴抽出 | `perception/acoustic_observer.py` | 高 |
| 状態分類器（ルールベース初版） | `perception/acoustic_observer.py` | 高 |
| ヒステリシス制御 | `control/throttle_controller.py` | 高 |
| 学習データ収集ツール | `data_collection/acoustic_recorder.py` | 中 |

### 4.2 NanoSAMセグメンテーション統合

**目標**: リアルタイムで走行可能領域を検出

```mermaid
flowchart LR
    Camera["カメラ<br/>フレーム"] --> Encoder["NanoSAM<br/>Encoder"]
    Encoder --> Decoder["NanoSAM<br/>Decoder"]
    Decoder --> Mask["セグメンテーション<br/>マスク"]
    Mask --> RoadExtract["走行可能領域<br/>抽出"]
    RoadExtract --> Steering["ステアリング<br/>推奨値計算"]
```

**ステアリング計算ロジック**:

```mermaid
flowchart TB
    Mask["走行可能領域<br/>マスク"] --> ROI["下半分を<br/>ROIとして抽出"]
    ROI --> Centroid["走行可能領域の<br/>重心計算"]
    Centroid --> Offset["中心からの<br/>オフセット計算"]
    Offset --> Steering["ステアリング値<br/>-1.0 〜 1.0"]
```

### 4.3 MCP統合

**目標**: FunctionGemmaがMCPツールを呼び出せるようにする

```mermaid
flowchart TB
    subgraph MCPServer["MCP Server (jetracer-agent)"]
        subgraph SensorTools["センサーツール"]
            get_camera["get_camera_frame()"]
            get_imu["get_imu_data()"]
            get_lidar["get_lidar_data()"]
            get_acoustic["get_acoustic_state()"]
        end
        
        subgraph ControlTools["制御ツール"]
            set_throttle["set_throttle(value)"]
            set_steering["set_steering(value)"]
            emergency["emergency_stop()"]
        end
        
        subgraph DataTools["データ収集ツール"]
            start_rec["start_recording(name)"]
            stop_rec["stop_recording()"]
        end
        
        subgraph PerceptionTools["認識ツール"]
            segment["segment_current_view()"]
        end
    end
    
    FuncGemma["FunctionGemma"] <-->|"MCP Protocol"| MCPServer
```

---

## 5. Phase 2-C: 統合開発（2週間）

### 5.1 自動データ収集システム

**目標**: NanoSAM + 音響制御で自動的に走行し、学習データを収集

```mermaid
flowchart TB
    subgraph MainLoop["10Hz メインループ"]
        direction TB
        
        Step1["1. センサーデータ取得"]
        Step2["2. NanoSAMセグメンテーション"]
        Step3["3. ステアリング決定"]
        Step4["4. スロットル決定"]
        Step5["5. データ記録"]
        Step6["6. 制御出力"]
        
        Step1 --> Step2 --> Step3 --> Step4 --> Step5 --> Step6
    end
    
    subgraph Sensors["センサー入力"]
        Camera["カメラ"]
        IMU["IMU"]
        LiDAR["LiDAR"]
        Mic["マイク"]
    end
    
    subgraph DataRecord["記録データ"]
        Image["画像 +<br/>ステアリング値"]
        Audio["音響特徴量 +<br/>状態ラベル"]
        Sync["全センサー<br/>同期データ"]
    end
    
    subgraph Output["制御出力"]
        PWM_S["PWM<br/>(steering)"]
        PWM_T["PWM<br/>(throttle)"]
    end
    
    Sensors --> Step1
    Step5 --> DataRecord
    Step6 --> Output
```

**自動アノテーションフロー**:

```mermaid
flowchart LR
    Frame["カメラ<br/>フレーム"] --> NanoSAM["NanoSAM"]
    NanoSAM --> Mask["走行可能<br/>領域マスク"]
    
    Mask --> SteeringCalc["ステアリング<br/>計算"]
    SteeringCalc --> ActualSteering["実際の<br/>ステアリング出力"]
    
    ActualSteering --> Label["ラベル"]
    Frame --> Dataset["データセット"]
    Mask --> Dataset
    Label --> Dataset
```

### 5.2 品質検証パイプライン

**yana-brain側での検証フロー**:

```mermaid
flowchart TB
    Session["セッション<br/>データ"] --> Load["フレーム<br/>読み込み"]
    
    Load --> BlurCheck["ブレ<br/>チェック"]
    Load --> SteeringCheck["ステアリング値<br/>妥当性"]
    Load --> RoadCheck["走行可能領域<br/>存在確認"]
    
    BlurCheck -->|"NG"| Reject["除外"]
    SteeringCheck -->|"NG"| Reject
    RoadCheck -->|"NG"| Reject
    
    BlurCheck -->|"OK"| Valid["有効フレーム"]
    SteeringCheck -->|"OK"| Valid
    RoadCheck -->|"OK"| Valid
    
    Valid --> QualityReport["品質レポート<br/>出力"]
```

---

## 6. Phase 2-D: レース準備（2週間）

### 6.1 MoE学習

**Mixture of Experts アーキテクチャ**:

```mermaid
flowchart TB
    Input["入力画像"] --> Gating["Gating Network<br/>状況判定"]
    Input --> Expert1["Expert 1<br/>直線区間"]
    Input --> Expert2["Expert 2<br/>緩いコーナー"]
    Input --> Expert3["Expert 3<br/>急カーブ"]
    
    Gating --> Weights["重み<br/>w1, w2, w3"]
    
    Expert1 --> Mix["加重平均"]
    Expert2 --> Mix
    Expert3 --> Mix
    Weights --> Mix
    
    Mix --> Output["ステアリング<br/>出力"]
```

**学習パイプライン**:

```mermaid
flowchart LR
    Data["収集データ"] --> Cluster["ステアリング値で<br/>クラスタリング"]
    Cluster --> TrainExperts["各Expert<br/>個別学習"]
    TrainExperts --> TrainGating["Gating Network<br/>学習"]
    TrainGating --> Export["ONNX<br/>エクスポート"]
    Export --> TRT["TensorRT<br/>変換"]
    TRT --> Deploy["Jetsonへ<br/>デプロイ"]
```

### 6.2 レース走行モード

```mermaid
flowchart TB
    subgraph RaceLoop["レースモードループ"]
        Camera["カメラ"] --> MoE["MoE推論"]
        MoE --> Steering["ステアリング<br/>制御"]
        
        Mic["マイク"] --> Acoustic["音響状態<br/>推定"]
        Acoustic --> Throttle["スロットル<br/>制御"]
        
        Acoustic --> Safety["安全<br/>チェック"]
        Safety -->|"異常検出"| EmergencyStop["緊急停止"]
        
        Steering --> PWM["PWM出力"]
        Throttle --> PWM
    end
```

---

## 7. センサー仕様

### 7.1 現在のハードウェア構成

| センサー | 型番 | インターフェース | サンプリング |
|---------|------|-----------------|-------------|
| カメラ | CSI Camera | MIPI CSI | 15 FPS |
| IMU | BNO055 | I2C | 100 Hz |
| LiDAR | VL53L7CX | I2C | 10 Hz |
| マイク | USB Microphone | USB/ALSA | 16 kHz |

### 7.2 センサー配置

```mermaid
flowchart TB
    subgraph JetRacer["JetRacer 構成"]
        subgraph Front["前方"]
            Camera["📷 Camera<br/>前方向き"]
            SteerServo["Steering<br/>Servo"]
        end
        
        subgraph Center["中央 (Jetson上)"]
            IMU["🔲 IMU<br/>BNO055"]
            Mic["🎤 Mic<br/>USB"]
        end
        
        subgraph Rear["後方"]
            Motor["Motor<br/>+ ESC"]
            LiDAR["LiDAR<br/>VL53L7CX<br/>下向き"]
        end
    end
    
    Front --- Center --- Rear
```

---

## 8. パフォーマンス目標

### 8.1 推論速度

| コンポーネント | 目標 | 優先度 |
|---------------|------|--------|
| NanoSAM | < 50ms | 高 |
| 音響特徴抽出 | < 10ms | 高 |
| 音響分類 | < 5ms | 高 |
| MoE (resnet18) | < 30ms | 中 |
| FunctionGemma | < 500ms | 低 |

### 8.2 メモリ配分（8GB）

```mermaid
pie title メモリ配分 (8GB)
    "OS/システム" : 1.5
    "NanoSAM (TensorRT)" : 0.5
    "音響処理 (librosa)" : 0.2
    "MoE (TensorRT)" : 0.3
    "FunctionGemma (Q4)" : 1.5
    "カメラ/バッファ" : 0.5
    "余裕" : 3.5
```

### 8.3 制御ループ

| モード | ループ周波数 | 備考 |
|--------|-------------|------|
| データ収集 | 10 Hz | NanoSAM + 音響制御 |
| レース走行 | 30 Hz | MoE + 音響制御 |

---

## 9. 実装スケジュール

### Week 1-2: Phase 2-A（要素実験）

```mermaid
gantt
    title Phase 2-A 詳細スケジュール
    dateFormat  YYYY-MM-DD
    section NanoSAM
    インストール        :a1, 2025-01-06, 1d
    TensorRT変換       :a2, after a1, 1d
    動作テスト         :a3, after a2, 1d
    section 音響
    録音環境構築       :b1, after a3, 1d
    各状態録音         :b2, after b1, 1d
    MFCC分析          :b3, after b2, 1d
    section FunctionGemma
    モデル導入         :c1, after b3, 2d
    ツール選択テスト   :c2, after c1, 2d
    section まとめ
    結果分析           :d1, after c2, 2d
    方針調整           :d2, after d1, 2d
```

### Week 3-4: Phase 2-B（基盤構築）

| 日程 | タスク | 担当 |
|------|--------|------|
| Day 1-4 | 音響スロットル制御基盤 | jetracer-agent |
| Day 5-8 | NanoSAM統合 | jetracer-agent |
| Day 9-12 | MCP統合 | jetracer-agent |
| Day 13-14 | 低速走行テスト | jetracer-agent |

### Week 5-6: Phase 2-C（統合開発）

| 日程 | タスク | 担当 |
|------|--------|------|
| Day 1-4 | 自動データ収集システム | jetracer-agent |
| Day 5-8 | データ収集実行（5-10周） | jetracer-agent |
| Day 9-12 | 品質検証・データ整理 | yana-brain |
| Day 13-14 | 問題修正・追加収集 | 両方 |

### Week 7-8: Phase 2-D（レース準備）

| 日程 | タスク | 担当 |
|------|--------|------|
| Day 1-4 | MoE学習 | yana-brain |
| Day 5-7 | TensorRT変換・デプロイ | yana-brain → jetracer-agent |
| Day 8-10 | レースモードテスト | jetracer-agent |
| Day 11-14 | パラメータチューニング | 両方 |

---

## 10. リスクと対策

```mermaid
flowchart LR
    subgraph Risks["リスク"]
        R1["NanoSAMが遅い"]
        R2["音響分類精度が低い"]
        R3["FunctionGemmaが不安定"]
        R4["メモリ不足"]
        R5["データ品質が低い"]
    end
    
    subgraph Mitigations["対策"]
        M1["解像度下げる<br/>キャッシュ活用"]
        M2["ルールベース<br/>フォールバック"]
        M3["MCPツール<br/>直接呼び出し"]
        M4["同時ロード<br/>モデル数制限"]
        M5["品質チェック強化<br/>手動フィルタ"]
    end
    
    R1 --> M1
    R2 --> M2
    R3 --> M3
    R4 --> M4
    R5 --> M5
```

---

## 11. 成功基準

### Phase 2-A 完了条件
- [ ] NanoSAMが50ms以下で推論
- [ ] 音響特徴量と状態の相関確認
- [ ] FunctionGemmaがツール選択80%以上

### Phase 2-B 完了条件
- [ ] 音響制御で低速一定走行（5m以上）
- [ ] NanoSAMで走行可能領域を正しく検出
- [ ] MCPツール呼び出しが動作

### Phase 2-C 完了条件
- [ ] 自動走行で1周完走
- [ ] 1000フレーム以上のデータ収集
- [ ] データ品質80%以上

### Phase 2-D 完了条件
- [ ] MoEモデルの学習完了
- [ ] レースモードで3周完走
- [ ] 衝突なしで走行

---

## 12. 改訂履歴

| バージョン | 日付 | 内容 |
|-----------|------|------|
| v1.0 | 2025-12-29 | 初版作成 |
| v1.1 | 2025-12-29 | 図をmermaid形式に変更 |

---

## 付録A: 参考文献・リンク

- [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam)
- [DA3](https://github.com/facebookresearch/DA3)
- [FunctionGemma](https://huggingface.co/google/gemma-2-2b-function-calling)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [librosa](https://librosa.org/)
