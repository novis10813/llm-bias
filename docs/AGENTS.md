# `docs/` guidance

本目錄保存政策、canonical workflow、artifact contract、研究方法與結果。上層解析規則
見 root `AGENTS.md`；文件分類、source of truth 與維護順序見
[`documentation-system.md`](documentation-system.md)。

## 文件慣例

- 每個 active experiment 放在 `docs/<experiment-name>/`。單一版本實驗包含 `proposal.md`
  與對應 report；多版本或分階段演進時，統一使用版本化檔名（`proposal-v1.md`、`proposal-v2.md`
  等），目錄下不保留無版本號的 `proposal.md`，避免語意混淆。
- 不同核心假說、因果機制或不同研究階段（例如定位 vs 歸因 vs 干預）應獨立立案或拆分文件，
  嚴禁將多階段研究路線混裝在單一 proposal 中。
- 各版本 workflow 的命令、參數、artifact layout 與 interpretation limits 只在該版本
  對應的 `proposal-vN.md` 維護，版本間不可相互覆蓋或回填假設；README 與 AGENTS 只提供
  摘要和連結。
- 實驗結果要附 run ID 或 artifact path，並標明 discovery、calibration、held-out、
  diagnostic 或 formal status。不要用新結果覆寫舊 run 的歷史紀錄；一旦
  formal run 執行完畢，對應的 `proposal-vN.md` 實質邏輯即刻凍結，禁止事後原地修改。
- Direction source、primary outcome、controls 或 gate 改變時，依
  [`documentation-system.md#experiment-versioning`](documentation-system.md#experiment-versioning)
  建立 version index、versioned proposal 與 versioned report；每版明寫
  implemented/proposed/evidence status。
- `proposal/` 只保存跨實驗 research program 與 roadmap；code 移入 archive 後，要同步
  更新 roadmap 的 code/protocol status。
- `archive/` 文件描述 frozen workflows。Archived CLI 不可寫成 active entry point；
  restore steps 統一連到 [`../archive/README.md`](../archive/README.md)。
- 圖表放 `assets/` 時，在來源 workflow 文件記錄 renderer、input run 與可重建方式。
- Markdown 連結使用相對路徑；修改或搬移文件後，檢查所有 inbound links。

Shared core 的 ownership 與 compatibility map 見
[`shared-experiment-core.md`](shared-experiment-core.md)；research operators 與 renderers
的詳細對照見 [`research-scripts.md`](research-scripts.md)；artifact lifecycle 見
[`artifact-contract.md`](artifact-contract.md)。

## 實驗目錄與版本化規格

多版本或分階段的研究線，在 `docs/<experiment-topic>/` 下採用以下檔案結構：

```text
docs/<experiment-topic>/
├── README.md               # 版本路由與矩陣（列出各版 status、primary outcome、evidence 與停止原因）
├── proposal-v1.md          # 凍結的 V1 協議（完成後禁止原地修改實質邏輯）
├── report-v1.md            # V1 實驗報告（記錄 run ID、完整數值、成功或失敗結論）
├── proposal-v2.md          # 若觸發版本分立條件，建立獨立文件
└── report-v2.md            # V2 實驗報告
```

### Proposal 必備章節與契約要素

每份 `proposal-vN.md` 必須具備以下章節與明確契約，不可省略：

1. **核心假說與文獻邊界（Scientific Question & Literature Boundary）**：
   - 明確標註方法參考的文獻出處（論文名稱、演算法、定理或任務設定）。
   - 附「文獻原始設定」與「本專案適應性修改（Adaptations）」的差異對照表，載明修改可能引入的理論風險與邊界限制。
2. **預期 Input / Output 契約**：
   - **Input**：精確記錄依賴的檔案路徑、預期欄位、型別、Tokenizer 條件與 upstream artifact hashes。
   - **Output**：產出的 compact JSON/JSONL 格式與 schema，禁止保存未聚合的 raw tensors/activations/KV caches，明定數值欄位必須為 finite float。
   - **CLI 契約 1:1 綁定**：Proposal 內記載的可執行命令，必須精準對應所屬實驗階段的專屬 Subcommand，嚴禁在文檔中寫入帶有跨階段條件分支、或需讀者自行挑選互斥參數的模糊指令。
3. **邊界情況與防禦性行為（Edge Cases & Fail-Safe Policies）**：
   - **退化條件**：明確定義何種數值或分佈屬 degenerate/degraded，以及退化時的 fallback 對照規則。
   - **控制組缺失**：若僅單一實體通過篩選或缺乏配對對照，定義系統回退行為（如報錯中斷或採用預設基準）。
   - **序列覆蓋**：序列 position 分割必須在邏輯上保證互斥且 100% 覆蓋目標區間。
   - **數值容差**：數值判定門檻（如 additivity tolerance）必須根據模型 precision（bf16/fp32）給出明確公式或底限。
4. **版本分立觸發條件（Version Break Triggers）**：
   - 當以下任一要素發生變更時，**必須建立新的 `proposal-v(N+1).md` 與 `report-v(N+1).md`，嚴禁原地修改既有文件**：
     - ① Prompt 構造方式或 prompt family 變更（例如固定 Header 轉自然句 Frames）；
     - ② 核心評估指標（Estimand）或 direction source 變更；
     - ③ 合格門檻（Gates / Thresholds）的判定規則或篩選條件調整；
     - ④ 控制組（Control family）的構造邏輯改變。
5. **強制端到端 Preflight 要求**：
   - 在啟動任何 formal discovery/calibration/test run 前，必須先以真實模型與真實 tokenizer 執行至少 1 筆 prompt 的完整端到端 smoke run（涵蓋所有 hook、controls、downstream 模組與 analyze 摘要），確認未拋出例外且輸出符合 schema 後，方可執行 formal run。

## Instruction Index

目前 `docs/` 的直接子目錄沒有 `AGENTS.md`。實驗目錄沿用本檔的 proposal/report 規則；
若單一目錄出現超過本檔一兩句能覆蓋的獨立 versioning 或 artifact 規則，再於該目錄新增
`AGENTS.md`，並只回來更新本節。

## Verification

修改 root guidance 或 workflow boundary 時執行：

```bash
uv run pytest -q tests/test_workflow_boundaries.py
```

修改 command block 時，對照 `pyproject.toml` entry points 與對應 CLI parser；修改相對
連結後，從該 Markdown 檔案所在目錄解析目標，不能只從 repository root 判斷。
