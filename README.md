# 🏀 HoopVision

Celem tego projektu jest stworzenie aplikacji, która pozwoli zawodnikom na **analizę oddanych przez siebie rzutów**.  
Za pomocą kamery ustawionej tak, aby obejmowała zarówno zawodnika, jak i kosz, system będzie **śledził piłkę i obręcz**, a następnie **wykrywał moment, w którym piłka przelatuje przez obręcz**.  
Na tej podstawie aplikacja obliczy **procent celności z oddanych rzutów**.

---

## 🌿 Zasady pracy z Git

💡 **Zasada:**
- Nigdy nie commitujemy bezpośrednio do `main`!
- Każda funkcja, poprawka lub eksperyment ma **swój osobny branch**.

---

## 🔹 Branch per feature / fix

Każdy branch powinien opisywać **konkretny cel** – funkcję, zadanie lub poprawkę.  
Nazwy piszemy **małymi literami**, używając **myślników (`-`)** jako separatorów.

---

## 🔹 Główne branche

| Branch | Przeznaczenie | Zasady |
|--------|----------------|--------|
| **`main`** | Stabilna wersja MVP | Tylko zatwierdzone release’y po code review |
| **`dev`** | Aktywny rozwój i integracja | Merge tylko po przeglądzie kodu |

💡 **Nigdy nie commitujemy bezpośrednio do `main`!**  
Każda funkcja, poprawka lub eksperyment ma swój osobny branch.

---

## 📋 Typy branchy

| Typ | Zastosowanie | Przykład |
|------|---------------|----------|
| `feature` | Nowa funkcja / moduł | `feature/ball-tracker` |
| `fix` | Poprawka błędu | `fix/tracker-stability` |
| `refactor` | Porządki w kodzie (bez zmiany funkcjonalności) | `refactor/overlay-cleanup` |
| `test` | Testy i eksperymenty | `test/kalman-filter` |

---

## 🧩 Zasady commitów

Każdy commit powinien być:
- **mały** (dotyczyć jednej zmiany),
- **czytelny**,
- **opisowy**.


## 📁 Struktura projektu

```bash
hoopvision/
├── models/
│   └── hoopvision.pt          # wytrenowany model YOLOv8
├── src/
│   ├── detector.py            # moduł detekcji obiektów (piłka, kosz)
│   ├── tracker.py             # śledzenie piłki i analiza trajektorii
│   ├── logic.py               # logika wykrywania trafień
│   ├── overlay.py             # nakładka graficzna na podgląd kamery
│   └── main.py                # punkt wejścia MVP
└── data/
    └── results.csv            # zapis wyników i trafień