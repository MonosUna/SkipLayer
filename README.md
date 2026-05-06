# SkipLayer — обучение LLM с пропуском слоёв через aligner

Standalone-фреймворк для **layer-skip**-обучения LLM. На каждом forward-pass
один (на трейне) или несколько (на инференсе) decoder-слоёв заменяются на
обучаемый **aligner** (MLP, который аппроксимирует выход настоящего слоя).
Loss — обычная cross-entropy на финальных логитах, поэтому градиент течёт в
aligner через хвост сети.

Подробное описание того, что было перенесено и как это устроено — см.
[`result.md`](result.md).

Быстрый старт:

```bash
cd SkipLayer
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .

COMET_API_KEY=... \
python -m src.scripts.train logger.workspace="your_workspace"
```
