# HoopVision

Celem tego projektu jest stworzenie aplikacji która pozwoli zawodnikom na analizę oddanych przez siebie rzutów.
Za pomocą kamery, ustawionej tak aby obejmowała zarówno zawodnika oraz kosz, system będzie przechwytywał piłkę oraz obręcz, a następnie wykrywał moment kiedy piłka przelatuje przez obręcz.
Na tej podstawie będzie obliczał procent celności z oddanych rzutów.

💡 Zasada:
	•	Nigdy nie commitujemy bezpośrednio do main!
	•	Każda funkcja, poprawka lub eksperyment ma swój osobny branch.

🔹 Branch per feature / fix
Każdy branch powinien opisywać konkretny cel – funkcję, zadanie lub poprawkę.
Nazwy piszemy małymi literami, używając myślników (-) jako separatorów.

🔹 Główne branche:
    main - stabilna wersja MVP, tylko zatwierdzone release’y po code review
    dev - aktywny rozwój i integracjamerge tylko po przeglądzie kodu, zawiera najnowsze zmiany z zespołu
	•	Nigdy nie commitujemy bezpośrednio do main!
	•	Każda funkcja, poprawka lub eksperyment ma swój osobny branch.
    
📋 Typy branchy:
    /feature - nowa funkcja/modul
    /fix - poprawka błędów
    /refactor - poprawki bez zmiany funkcjonalnosci(clean code)
    /test - testy

Każdy commit powinien być mały, czytelny i opisowy.

Struktura projektu
hoopvision/
├── models/hoopvision.pt
├── src/
│   ├── detector.py
│   ├── tracker.py
│   ├── logic.py
│   ├── overlay.py
│   └── main.py
└── data/results.csv
