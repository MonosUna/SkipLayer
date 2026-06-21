# SkipLayer — ускорение инференса LLM через пропуск слоёв

Фреймворк для исследования **layer-skip**-инференса больших языковых моделей.
Цель — ускорить авторегрессионную генерацию, выборочно пропуская часть
decoder-слоёв, и компенсировать пропуск без потери качества.

Основной метод — **training-free EMA-компенсация**: веса предобученной модели
заморожены, метод работает только во время генерации и не требует дообучения.
Он состоит из четырёх независимых компонентов:

- **защищённая зона** — первые и последние слои никогда не пропускаются;
- **отбор кандидатов** на пропуск по динамической мере важности
  $I_l = \|\Delta_l\| / \|h\|$;
- **EMA-компенсация** скрытого состояния через скользящее экспоненциальное
  среднее приращений $\Delta_l = h_{\text{out}} - h_{\text{in}}$;
- **заполнение KV-кэша** пропущенного слоя копированием $(k, v)$ соседнего
  честно вычисленного слоя (simple KV propagate).

Для сравнения реализованы random-skip-бейзлайны и обучаемые **aligner**'ы
(MLP, который аппроксимирует выход настоящего слоя; обучается обычной
cross-entropy на финальных логитах). Все компоненты — отдельные модули, которые
включаются и отключаются в ablation-режиме через конфиги Hydra.

Модель по умолчанию — `Qwen3-4B-Instruct-2507`, оценка на тестовой части
`trivia_qa`. Метрика качества — per-step accuracy относительно полной модели
(forced decoding), см. [`src/metrics/skip_metrics.py`](src/metrics/skip_metrics.py:1).

## Быстрый старт

```bash
cd SkipLayer
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Запуск EMA-метода (инференс, без обучения):

```bash
COMET_API_KEY=... \
python -m src.scripts.train --config-name=ema_skip logger.workspace="your_workspace"
```

Random-skip-бейзлайн и обучение aligner'а:

```bash
python -m src.scripts.train --config-name=baseline
python -m src.scripts.train --config-name=layer_skip
```

## Эксперименты

Все запуски из работы собраны в [`exps.sh`](exps.sh:1):

```bash
bash exps.sh list            # список доступных экспериментов
bash exps.sh exp_ema_default # один эксперимент
bash exps.sh all             # все по очереди
bash exps.sh aggregate       # сводная таблица метрик в outputs/summary.csv
```
