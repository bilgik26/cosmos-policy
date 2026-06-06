# RoboCasa Target Datasets (target / human)

`download_datasets.py` でダウンロードできる `split=target, source=human` のデータセット一覧。  
`TARGET_TASKS` 定義（`robocasa/utils/dataset_registry.py` L.2802–2858）に基づく。

---

## 1. Atomic Seen（18 タスク）

pretraining にも使用された atomic タスク。

| タスク名 | データセットパス | horizon |
|---|---|---:|
| CloseBlenderLid | `v1.0/target/atomic/CloseBlenderLid/20250822` | 900 |
| CloseFridge | `v1.0/target/atomic/CloseFridge/20250816` | 900 |
| CloseToasterOvenDoor | `v1.0/target/atomic/CloseToasterOvenDoor/20250818` | 450 |
| CoffeeSetupMug | `v1.0/target/atomic/CoffeeSetupMug/20250813` | 600 |
| NavigateKitchen | `v1.0/target/atomic/NavigateKitchen/20250821` | 450 |
| OpenCabinet | `v1.0/target/atomic/OpenCabinet/20250813` | 1050 |
| OpenDrawer | `v1.0/target/atomic/OpenDrawer/20250816` | 750 |
| OpenStandMixerHead | `v1.0/target/atomic/OpenStandMixerHead/20250818` | 450 |
| PickPlaceCounterToCabinet | `v1.0/target/atomic/PickPlaceCounterToCabinet/20250811` | 750 |
| PickPlaceCounterToStove | `v1.0/target/atomic/PickPlaceCounterToStove/20250818` | 600 |
| PickPlaceDrawerToCounter | `v1.0/target/atomic/PickPlaceDrawerToCounter/20250820` | 750 |
| PickPlaceSinkToCounter | `v1.0/target/atomic/PickPlaceSinkToCounter/20250813` | 900 |
| PickPlaceToasterToCounter | `v1.0/target/atomic/PickPlaceToasterToCounter/20250817` | 600 |
| SlideDishwasherRack | `v1.0/target/atomic/SlideDishwasherRack/20250820` | 450 |
| TurnOffStove | `v1.0/target/atomic/TurnOffStove/20250812` | 750 |
| TurnOnElectricKettle | `v1.0/target/atomic/TurnOnElectricKettle/20250817` | 450 |
| TurnOnMicrowave | `v1.0/target/atomic/TurnOnMicrowave/20250813` | 450 |
| TurnOnSinkFaucet | `v1.0/target/atomic/TurnOnSinkFaucet/20250812` | 600 |

ダウンロードコマンド例:
```bash
python robocasa/scripts/download_datasets.py \
  --split target --source human --task_type atomic \
  --tasks CloseBlenderLid CloseFridge ...
```

---

## 2. Composite Seen（16 タスク）

pretraining にも使用された composite タスク。

| タスク名 | データセットパス | horizon |
|---|---|---:|
| DeliverStraw | `v1.0/target/composite/DeliverStraw/20250813` | 2550 |
| GetToastedBread | `v1.0/target/composite/GetToastedBread/20250812` | 3000 |
| KettleBoiling | `v1.0/target/composite/KettleBoiling/20250814` | 1500 |
| LoadDishwasher | `v1.0/target/composite/LoadDishwasher/20250811` | 1800 |
| PackIdenticalLunches | `v1.0/target/composite/PackIdenticalLunches/20250815` | 3900 |
| PreSoakPan | `v1.0/target/composite/PreSoakPan/20250809` | 2400 |
| PrepareCoffee | `v1.0/target/composite/PrepareCoffee/20250812` | 1800 |
| RinseSinkBasin | `v1.0/target/composite/RinseSinkBasin/20250816` | 1350 |
| ScrubCuttingBoard | `v1.0/target/composite/ScrubCuttingBoard/20250816` | 1200 |
| SearingMeat | `v1.0/target/composite/SearingMeat/20250812` | 4350 |
| SetUpCuttingStation | `v1.0/target/composite/SetUpCuttingStation/20250817` | 2400 |
| StackBowlsCabinet | `v1.0/target/composite/StackBowlsCabinet/20250815` | 2100 |
| SteamInMicrowave | `v1.0/target/composite/SteamInMicrowave/20250814` | 2100 |
| StirVegetables | `v1.0/target/composite/StirVegetables/20250814` | 2400 |
| StoreLeftoversInBowl | `v1.0/target/composite/StoreLeftoversInBowl/20250813` | 2550 |
| WashLettuce | `v1.0/target/composite/WashLettuce/20250814` | 1650 |

ダウンロードコマンド例:
```bash
python robocasa/scripts/download_datasets.py \
  --split target --source human \
  --tasks DeliverStraw GetToastedBread KettleBoiling ...
```

---

## 3. Composite Unseen（16 タスク）

pretraining には含まれない、汎化評価用の composite タスク。

| タスク名 | データセットパス | horizon |
|---|---|---:|
| ArrangeBreadBasket | `v1.0/target/composite/ArrangeBreadBasket/20250809` | 4350 |
| ArrangeTea | `v1.0/target/composite/ArrangeTea/20250812` | 2250 |
| BreadSelection | `v1.0/target/composite/BreadSelection/20250815` | 1950 |
| CategorizeCondiments | `v1.0/target/composite/CategorizeCondiments/20250814` | 1650 |
| CuttingToolSelection | `v1.0/target/composite/CuttingToolSelection/20250814` | 1200 |
| GarnishPancake | `v1.0/target/composite/GarnishPancake/20250815` | 2700 |
| GatherTableware | `v1.0/target/composite/GatherTableware/20250815` | 2250 |
| HeatKebabSandwich | `v1.0/target/composite/HeatKebabSandwich/20250813` | 2700 |
| MakeIceLemonade | `v1.0/target/composite/MakeIceLemonade/20250813` | 3000 |
| PanTransfer | `v1.0/target/composite/PanTransfer/20250817` | 1800 |
| PortionHotDogs | `v1.0/target/composite/PortionHotDogs/20250816` | 2250 |
| RecycleBottlesByType | `v1.0/target/composite/RecycleBottlesByType/20250812` | 2850 |
| SeparateFreezerRack | `v1.0/target/composite/SeparateFreezerRack/20250815` | 2400 |
| WaffleReheat | `v1.0/target/composite/WaffleReheat/20250817` | 4050 |
| WashFruitColander | `v1.0/target/composite/WashFruitColander/20250811` | 3150 |
| WeighIngredients | `v1.0/target/composite/WeighIngredients/20250812` | 3000 |

ダウンロードコマンド例:
```bash
python robocasa/scripts/download_datasets.py \
  --split target --source human \
  --tasks ArrangeBreadBasket ArrangeTea BreadSelection ...
```

---

## まとめ

| カテゴリ | タスク数 | 用途 |
|---|---:|---|
| Atomic Seen | 18 | 単一操作タスク（pretraining と共通） |
| Composite Seen | 16 | 複合タスク（pretraining と共通） |
| Composite Unseen | 16 | 複合タスク（汎化評価専用） |
| **合計** | **50** | `target50` として一括参照可能 |

全50タスクまとめてダウンロード:
```bash
python robocasa/scripts/download_datasets.py --split target --source human
```
